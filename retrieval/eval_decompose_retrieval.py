
import argparse
from ast import parse
import collections
import csv
import json
import logging
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=2, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-len', type=int, default=70)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--max-c-len', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size-1', type=int, default=5)
    parser.add_argument('--beam-size-2', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--save-index', action="store_true")
    parser.add_argument('--only-eval-ans', action="store_true")
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
    args = parser.parse_args()
    
    logger.info("Loading data...")

    all_data = open(args.raw_data).readlines()
    ds_items = [json.loads(_.strip()) for _ in all_data]

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
    title2doc = {item[0]:item[1] for item in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")
    
    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = UnifiedRetriever(bert_config, args)
    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info("Encoding questions and searching")
    question_pairs = [(_["subQ_1"], _["subQ_2"]) for _ in ds_items]
    gold_passage = [[_["sp"][0]["title"], _["sp"][1]["title"]] for _ in ds_items]
    question_ids = [_["id"] for _ in ds_items]
    questions = [_["orig_q"][:-1] if _["orig_q"].endswith("?") else _["orig_q"] for _ in ds_items]

    metrics = []
    retrieval_outputs = []
    for b_start in tqdm(range(0, len(question_pairs), args.batch_size)):
        with torch.no_grad():
            batch_q = question_pairs[b_start:b_start + args.batch_size]
            batch_ann = gold_passage[b_start:b_start + args.batch_size]
            bsize = len(batch_q)
            batch_ids = question_ids[b_start:b_start + args.batch_size]
            batch_origq = questions[b_start:b_start + args.batch_size]

            batch_q_encodes = tokenizer.batch_encode_plus([" ".join(_[0].split()) for _ in batch_q], max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            # batch_q_encodes = tokenizer.batch_encode_plus(batch_origq, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy() 
            D, I = index.search(q_embeds_numpy, args.beam_size_1)

            # 2 hop search
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):
                    doc = id2doc[str(doc_id)][1]
                    # query_pairs.append((" ".join(batch_q[b_idx]), doc))
                    query_pairs.append(batch_q[b_idx][1])

            # batch_q_sp_encodes = tokenizer.batch_encode_plus([" ".join(_[1].split()) for _ in batch_q], max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_sp_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_sp_encodes = move_to_cuda(dict(batch_q_sp_encodes))
            q_sp_embeds = model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], batch_q_sp_encodes.get("token_type_ids", None))
            q_sp_embeds = q_sp_embeds.contiguous().cpu().numpy()
            
            D_, I_ = index.search(q_sp_embeds, args.beam_size_2)
            D_ = D_.reshape(bsize, args.beam_size_1, args.beam_size_2)
            I_ = I_.reshape(bsize, args.beam_size_1, args.beam_size_2)

            # aggregate path scores
            # path_scores = np.expand_dims(D, axis=2) + np.expand_dims(D_, axis=1)
            path_scores = np.expand_dims(D, axis=2) + D_

            for idx in range(bsize):
                search_scores = path_scores[idx]
                ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], (args.beam_size_1, args.beam_size_2))).transpose()
                retrieved_titles = []
                hop1_titles = []
                paths, path_titles = [], []
                for _ in range(args.topk):
                    path_ids = ranked_pairs[_]
                    hop_1_id = I[idx, path_ids[0]]
                    # hop_2_id = I_[idx, path_ids[1]]
                    hop_2_id = I_[idx, path_ids[0], path_ids[1]]
                    retrieved_titles.append(id2doc[str(hop_1_id)][0])
                    retrieved_titles.append(id2doc[str(hop_2_id)][0])
                    paths.append([str(hop_1_id), str(hop_2_id)])
                    path_titles.append([id2doc[str(hop_1_id)][0], id2doc[str(hop_2_id)][0]])
                    hop1_titles.append(id2doc[str(hop_1_id)][0])
                
                sp = batch_ann[idx]
                assert len(set(sp)) == 2
                path_covered = [int(set(p) == set(sp)) for p in path_titles]
                path_covered = np.sum(path_covered) > 0
                
                all_retrieved_titles = set()
                for p in path_titles:
                    all_retrieved_titles.add(p[0])
                    all_retrieved_titles.add(p[1])
                sp_covered = int(sp[0] in all_retrieved_titles and sp[1] in all_retrieved_titles)

                metrics.append({
                'path_covered': int(path_covered),
                'sp_covered': sp_covered
                })

                if args.save_path != "":
                    candidate_chains = []
                    for path in path_titles:
                        candidate_chains.append([(path[0], title2doc[path[0]]), (path[1], title2doc[path[1]])])
                    retrieval_outputs.append({
                        "_id": batch_ids[idx],
                        "question": batch_origq[idx],
                        "q_pairs": batch_q[idx],
                        "sp": [(sp[0], title2doc[sp[0]]), (sp[1], title2doc[sp[1]])],
                        # "answer": batch_ann[idx]["answer"],
                        "candidate_chains": candidate_chains
                    })

    if args.save_path != "":
        with open(f"/private/home/xwhan/data/QDMR/{args.save_path}", "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")
                
    logger.info(f"Evaluating {len(metrics)} samples...")
    logger.info(f'Path Recall: {np.mean([m["path_covered"] for m in metrics])}')
    logger.info(f'SP Recall: {np.mean([m["sp_covered"] for m in metrics])}')

