"""


# single-hop baseline for hotpotQA
python single_hop_hq_baseline.py /private/home/xwhan/data/hotpot/hotpot_qas_val.json index/abstracts_single.npy index/hotpotQA_corpus_dict.json logs/09-10-2020/hotpot_single-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-roberta-base/checkpoint_best.pt --batch-size 50 --topk 20 --shared-encoder --model-name roberta-base --gpu


"""
import argparse
import collections
import json
import logging
from os import path
import time

import faiss

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.mhop_retriever import MhopRetriever, RobertaRetriever
from models.unified_retriever import UnifiedRetriever
from utils.basic_tokenizer import SimpleTokenizer
from utils.utils import (load_saved, move_to_cuda, para_has_answer)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def convert_hnsw_query(query_vectors):
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=2, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-len', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--save-index', action="store_true")
    parser.add_argument('--only-eval-ans', action="store_true")
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument("--unified", action="store_true")
    parser.add_argument('--hnsw', action="store_true")
    args = parser.parse_args()
    
    logger.info("Loading data...")
    ds_items = [json.loads(_) for _ in open(args.raw_data).readlines()]

    logger.info("Building index...")
    d = 768
    xb = np.load(args.indexpath).astype('float32')
    print(xb.shape)

    index = faiss.IndexFlatIP(d)
    index.add(xb)
    if args.gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 1, index)

    
    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    if isinstance(id2doc["0"], list):
        id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}
    # title2text = {v[0]:v[1] for v in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")
    
    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.unified:
        model = UnifiedRetriever(bert_config, args)
    elif "roberta" in args.model_name:
        model = RobertaRetriever(bert_config, args)
    else:
        model = MhopRetriever(bert_config, args)
    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info("Encoding questions and searching")
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    metrics = []
    retrieval_outputs = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ann = ds_items[b_start:b_start + args.batch_size]
            bsize = len(batch_q)

            batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))

            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            D, I = index.search(q_embeds_numpy, args.topk)

            for idx in range(bsize):
                retrieved_titles = []
                for _, doc_id in enumerate(I[idx]):
                    retrieved_titles.append(id2doc[str(doc_id)]["title"])
                
                sp = batch_ann[idx]["sp"]
                assert len(set(sp)) == 2
                type_ = batch_ann[idx]["type"]
                question = batch_ann[idx]["question"]
                p_recall, p_em = 0, 0
                sp_covered = [sp_title in retrieved_titles for sp_title in sp]
                if np.sum(sp_covered) > 0:
                    p_recall = 1
                if np.sum(sp_covered) == len(sp_covered):
                    p_em = 1

                metrics.append({
                "question": question,
                "p_recall": p_recall,
                "p_em": p_em,
                "type": type_,
                })

    logger.info(f"Evaluating {len(metrics)} samples...")
    type2items = collections.defaultdict(list)
    for item in metrics:
        type2items[item["type"]].append(item)

    logger.info(f'Avg PR: {np.mean([m["p_recall"] for m in metrics])}')
    logger.info(f'Avg P-EM: {np.mean([m["p_em"] for m in metrics])}')
    for t in type2items.keys():
        logger.info(f"{t} Questions num: {len(type2items[t])}")
        logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in type2items[t]])}')
        logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in type2items[t]])}')
