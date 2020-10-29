"""
Single-hop retrieval evaluation


python eval_unified_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/merged_all.npy index/merged_all_id2doc.json logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 100 --shared-encoder --model-name bert-base-uncased --unified --topk 20


python eval_unified_retrieval.py /private/home/xwhan/data/hotpot/hotpot_qas_val.json index/merged_all.npy index/merged_all_id2doc.json logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 50 --shared-encoder --model-name bert-base-uncased --unified --topk 20


# with newest model
python eval_unified_retrieval.py /private/home/xwhan/data/hotpot/hotpot_qas_val.json index/merged_all_retrained_no_period.npy index/merged_all_id2doc.json logs/09-15-2020/unified_bert_no_period-seed16-bsz128-fp16True-lr2e-05-decay0.0-adamTrue/checkpoint_best.pt --batch-size 50 --shared-encoder --model-name bert-base-uncased --unified --topk 20

python eval_unified_retrieval.py /private/home/xwhan/data/nq-dpr/nq-test-qas.txt index/merged_all.npy index/merged_all_id2doc.json logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 100 --shared-encoder --model-name bert-base-uncased --unified --topk 20

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
from models.unified_retriever import UnifiedRetriever

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
        topkpara_covered.append(int(para_has_answer(answer, p, PROCESS_TOK)))

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
    parser.add_argument('--max-q-sp-len', type=int, default=350)
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
    qas = [_ for _ in qas if _["answer"][0] not in ["yes", "no"]]
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in qas]
    answers = [item["answer"] for item in qas]

    logger.info(f"Loading index")
    d = 768
    xb = np.load(args.indexpath).astype('float32')
    index = faiss.IndexFlatIP(d)
    index.add(xb)


    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = UnifiedRetriever(bert_config, args)
    
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
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ans = answers[b_start:b_start + args.batch_size]
            bsize = len(batch_q)

            # test retrieval model with marker
            # batch_q_toks = [add_marker_q(tokenizer, q) for q in batch_q]
            batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")    

            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            D, I = index.search(q_embeds_numpy, args.topk)

            # 2hop search
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):
                    doc = id2doc[str(doc_id)][1]
                    query_pairs.append((batch_q[b_idx], doc))

            batch_q_sp_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_sp_encodes = move_to_cuda(dict(batch_q_sp_encodes))
            q_sp_embeds, stop_logits = model.encode_qsp(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], batch_q_sp_encodes.get("token_type_ids", None))

            next_hop = stop_logits.argmax(dim=1).view(bsize, args.topk)[:,:5].cpu().numpy().sum(1) > (min(5, args.topk) / 2)

            if np.sum(next_hop) > 0:
                q_sp_embeds = q_sp_embeds.contiguous().cpu().numpy()
                D_, I_ = index.search(q_sp_embeds, args.topk)
                D_ = D_.reshape(bsize, args.topk, args.topk)
                I_ = I_.reshape(bsize, args.topk, args.topk)
                path_scores = np.expand_dims(D, axis=2) + D_

            for idx in range(bsize):
                topk_docs = []
                if not next_hop[idx]:
                    for doc_id in I[idx]:
                        topk_docs.append(id2doc[str(doc_id)][0] + " " +  id2doc[str(doc_id)][1])
                else:
                    search_scores = path_scores[idx]
                    ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], (args.topk, args.topk))).transpose()

                    for _ in range(args.topk):
                        path_ids = ranked_pairs[_]
                        hop_1_id = I[idx, path_ids[0]]
                        hop_2_id = I_[idx, path_ids[0], path_ids[1]]

                        concat_p = id2doc[str(hop_1_id)][0] + " " + id2doc[str(hop_1_id)][1] + " " + id2doc[str(hop_2_id)][0] + " " + id2doc[str(hop_2_id)][1]
                        topk_docs.append(concat_p)
                assert len(topk_docs) == args.topk
                retrieved_results.append(topk_docs)
                        

    answers_docs = list(zip(questions, answers, retrieved_results))
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init
    )
    get_score_partial = partial(
         get_score, topk=args.topk)
    results = processes.map(get_score_partial, answers_docs)


    aggregate = defaultdict(list)
    for r in results:
        for k, v in r.items():
            aggregate[k].append(v)

    for k in aggregate:
        results = aggregate[k]
        print('Top {} Recall for {} QA pairs: {} ...'.format(
            k, len(results), np.mean(results)))
