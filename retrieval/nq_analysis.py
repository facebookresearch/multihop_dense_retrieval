"""
python nq_analysis.py index/psg100_nqmhop.npy index/psgs_w100_id2doc.json logs/08-24-2020/nq_mhop_2_marker_shared_dpr_neg-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt/checkpoint_best.pt --batch-size 100 --shared-encoder --model-name bert-base-uncased --topk 10 --beam-size 10

python nq_analysis.py index/psg100_mhop_add_marker.npy index/psgs_w100_id2doc.json logs/08-22-2020/nq_mhop_add_marker_type2-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 20 --shared-encoder --model-name bert-base-uncased --topk 50 --beam-size 50


python nq_analysis.py index/psg100_mhop_wq_1_from_baseline.npy index/psgs_w100_id2doc.json  logs/08-26-2020/wq_mhop_1_shared_dpr_neg_from_scratch-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 200 --shared-encoder --model-name bert-base-uncased --topk 10 --beam-size 10
"""

import json
import logging
import argparse

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.mhop_retriever import MhopRetriever, RobertaRetriever
from models.retriever import RobertaRetrieverSingle
from models.unified_retriever import UnifiedRetriever, BertNQRetriever
from utils.basic_tokenizer import SimpleTokenizer
from utils.utils import (load_saved, para_has_answer, move_to_cuda)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def get_errors():
    """
    get those error cases of NQ DPR model
    """
    nq_top100_results = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/results/wq-test-res-type1.txt")]

    top20_errors = []
    for item in nq_top100_results:
        if item["metrics"]["50"] == 0:
            top20_errors.append(item)

    print(len(top20_errors))
    with open("/private/home/xwhan/data/nq-dpr/results/wq-test-type2-scratch-res-top50-errors.txt", "w") as g:
        for item in top20_errors:
            g.write(json.dumps(item) + "\n")

def add_marker_q(tokenizer, q, marker):
    q_toks = tokenizer.tokenize(q)
    return [marker] + q_toks

def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=20, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--max-c-len', type=int, default=300)
    parser.add_argument('--max-q-len', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
    args = parser.parse_args()

    logger.info("Loading data...")
    all_data = open(f"/private/home/xwhan/data/nq-dpr/results/wq-test-type2-scratch-res-top{args.topk}-errors.txt").readlines()

    ds_items = [json.loads(_.strip()) for _ in all_data]

    logger.info("Building index...")
    d = 768
    xb = np.load(args.indexpath).astype('float32')
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    title2doc = {item[0]:item[1] for item in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")

    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # model = UnifiedRetriever(bert_config, args)
    # model = RobertaRetrieverSingle(bert_config, args)
    model = BertNQRetriever(bert_config, args)

    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    questions = [_["question"] for _ in ds_items]
    answers = [_["ans"] for _ in ds_items]
    retrieved_results = [_["topkdocs"][:args.topk] for _ in ds_items]
    retrieved_docids = [_["topk_ids"][:args.topk] for _ in ds_items]

    corrected_cases = []
    ans_recalls = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ans = answers[b_start:b_start + args.batch_size]
            bsize = len(batch_q)

            topk_1hop = retrieved_results[b_start:b_start + args.batch_size]
            docids_1hop = retrieved_docids[b_start:b_start + args.batch_size]

            query_pairs = []
            for q, topk in zip(batch_q, topk_1hop):
                for doc in topk:
                    query_pairs.append((q, doc["title"] + " [SEP] " + doc["text"].strip()))

            # batch_q_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")

            # add marker to q
            tokenized_query_pairs = []
            for q, p in query_pairs:
                tokenized_query_pairs.append((add_marker_q(tokenizer, q, '[unused1]'), tokenizer.tokenize(p)))
    
            batch_q_encodes = tokenizer.batch_encode_plus(tokenized_query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            D, I = index.search(q_embeds_numpy, args.beam_size)

            # add scores from 1st hop
            # D = D.reshape(bsize, args.topk, args.beam_size) + np.expand_dims(D_1, axis=2)
            
            D = D.reshape(bsize, args.topk*args.beam_size)
            I = I.reshape(bsize, args.topk*args.beam_size)

            for idx in range(bsize):
                existing_doc_ids = set(docids_1hop[idx])
                assert len(existing_doc_ids) == args.topk


                docs_2hop = []
                ans_recall = []

                ids_and_scores = list(zip(I[idx], D[idx]))
                ids_and_scores.sort(key=lambda x:x[1], reverse=True)

                count = 0
                for doc_id, score in ids_and_scores:
                    if str(doc_id) not in existing_doc_ids:
                        existing_doc_ids.add(str(doc_id))
                        docs_2hop.append({"title": id2doc[str(doc_id)][0], "text": id2doc[str(doc_id)][1]})
                        ans_recall.append(int(para_has_answer(batch_ans[idx], docs_2hop[-1]["title"] + " " + docs_2hop[-1]["text"], simple_tokenizer)))
                        count += 1
                        if count == args.topk:
                            break


                if np.sum(ans_recall) > 0:
                    corrected_cases.append(
                        {
                            "question": batch_q[idx],
                            "answer": batch_ans[idx],
                        }
                    )

                ans_recalls.append(int(np.sum(ans_recall) > 0))

    with open("/private/home/xwhan/data/nq-dpr/analysis/nq_recovered_cases_top50_errors.json", "w") as g:
        for _ in corrected_cases:
            g.write(json.dumps(_) + "\n")

    assert len(ans_recalls) == len(questions)
    print(f"Top{args.topk} 2hop, answer recall {np.mean(ans_recalls)}, count {len(ans_recalls)}")

if __name__ == '__main__':
    analyze()
    # get_errors()