# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
Single-hop retrieval evaluation

## Use the unified model (trained with both hotpotQA and NQ)


python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-val-simplified.txt index/psg100_unified.npy index/psgs_w100_id2doc.json logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 1000 --shared-encoder --model-name bert-base-uncased --unified --save-pred nq-val-filtered-top50.txt --topk 50


# DPR shared-encoder baseline bsz256
python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/psg100_dpr_shared_baseline.npy index/psgs_w100_id2doc.json logs/08-23-2020/nq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --max-q-len 50 --save-pred nq-test-dpr-shared-b256-res.txt  

# shared encoder on merged corpus
python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/merged_all_single_only.npy index/merged_all_id2doc.json logs/08-23-2020/nq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --max-q-len 50

# to get negatives from DPR shared baseline
python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-val-simplified.txt index/psg100_dpr_shared_baseline.npy index/psgs_w100_id2doc.json logs/08-25-2020/wq_mhop_1_shared_dpr_neg_from_scratch-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --save-pred nq-val-shared-dpr-top100.txt --topk 100 

python eval_retrieval.py /private/home/xwhan/data/WebQ/WebQuestions-test.txt index/psg100_mhop_wq_1_from_baseline.npy index/psgs_w100_id2doc.json  logs/08-26-2020/wq_mhop_1_shared_dpr_neg_from_scratch-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --model-name bert-base-uncased --shared-encoder --save-pred wq-test-res-type1.txt


python eval_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/merged_all.npy index/merged_all_id2doc.json logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 1000 --shared-encoder --model-name bert-base-uncased --unified



"""

import numpy as np
import json
import faiss
import argparse
import logging
import torch
from tqdm import tqdm

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import defaultdict

from utils.utils import load_saved, move_to_cuda, para_has_answer
from utils.basic_tokenizer import SimpleTokenizer

from transformers import AutoConfig, AutoTokenizer
from models.retriever import BertRetrieverSingle, RobertaRetrieverSingle
from models.unified_retriever import UnifiedRetriever, BertNQRetriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

PROCESS_TOK = None
def init():
    global PROCESS_TOK
    PROCESS_TOK = SimpleTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

def get_score(answer_doc, topk=20):
    """Search through all the top docs to see if they have the answer."""
    question, answer, docs = answer_doc
    top5doc_covered = 0
    global PROCESS_TOK
    topkpara_covered = []
    for p in docs:
        topkpara_covered.append(int(para_has_answer(answer, p["title"] + " " + p["text"], PROCESS_TOK)))

    return {
        "5": int(np.sum(topkpara_covered[:5]) > 0),
        "10": int(np.sum(topkpara_covered[:10]) > 0),
        "20": int(np.sum(topkpara_covered[:20]) > 0),
        "50": int(np.sum(topkpara_covered[:50]) > 0),
        "100": int(np.sum(topkpara_covered[:100]) > 0),
        "covered": topkpara_covered
    }


def add_marker_q(tokenizer, q):
    q_toks = tokenizer.tokenize(q)
    return ['[unused0]'] + q_toks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--max-q-len', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save-pred", default="", type=str)
    parser.add_argument("--unified", action="store_true", help="test with unified trained model")
    args = parser.parse_args()

    logger.info(f"Loading questions")
    qas = [json.loads(line) for line in open(args.raw_data).readlines()]
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in qas]
    answers = [item["answer"] for item in qas]

    logger.info(f"Loading index")
    d = 768
    xb = np.load(args.indexpath).astype('float32')
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    if args.gpu:    
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 1, index)
    # logger.info(f"Building GPU index")
    # co = faiss.GpuMultipleClonerOptions()
    # co.useFloat16 = True
    # co.shards = True
    # index = faiss.index_cpu_to_gpus_list(index, co, [1,2,3,4,5,6,7])
    # index.add(xb)

    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.unified:
        model = UnifiedRetriever(bert_config, args)
    elif "roberta" in args.model_name:
        model = RobertaRetrieverSingle(bert_config, args)
    else:
        model = BertRetrieverSingle(bert_config, args)
    
    model = load_saved(model, args.model_path, exact=False)
    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info(f"Loading corpus")
    id2doc = json.load(open(args.corpus_dict))
    logger.info(f"Corpus size {len(id2doc)}")

    retrieved_results = []
    retrieved_docids = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ans = answers[b_start:b_start + args.batch_size]

            # test retrieval model with marker
            # batch_q_toks = [add_marker_q(tokenizer, q) for q in batch_q]
            # batch_q_encodes = tokenizer.batch_encode_plus(batch_q_toks, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt", is_pretokenized=True)

            batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt", is_pretokenized=True)        

            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            D, I = index.search(q_embeds_numpy, args.topk)
            for b_idx in range(len(batch_q)):
                topk_docs = [{"title": id2doc[str(doc_id)][0],"text": id2doc[str(doc_id)][1]} for doc_id in I[b_idx]]
                retrieved_results.append(topk_docs)
                retrieved_docids.append([str(doc_id) for doc_id in I[b_idx]])

    answers_docs = list(zip(questions, answers, retrieved_results))
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init
    )
    get_score_partial = partial(
         get_score, topk=args.topk)
    results = processes.map(get_score_partial, answers_docs)

    if args.save_pred != "":
        to_save = []
        for inputs, metrics, topk_ids in zip(answers_docs, results, retrieved_docids):
            q, ans, topk_doc = inputs
            topk_covered = metrics["covered"]
            assert len(topk_doc) == len(topk_covered)
            assert len(topk_doc) == len(topk_ids)
            to_save.append({
                "question": q,
                "ans": ans,
                "topk": list(zip(topk_doc, topk_covered)),
                "topkdocs": topk_doc,
                "metrics": metrics,
                "topk_ids": topk_ids
            })
        print(f"Saving {len(to_save)} instances...")
        with open("/private/home/xwhan/data/nq-dpr/results/" + args.save_pred, "w") as g:
            for l in to_save:
                g.write(json.dumps(l) + "\n")

    aggregate = defaultdict(list)
    for r in results:
        for k, v in r.items():
            aggregate[k].append(v)

    for k in aggregate:
        results = aggregate[k]
        print('Top {} Recall for {} QA pairs: {} ...'.format(
            k, len(results), np.mean(results)))
