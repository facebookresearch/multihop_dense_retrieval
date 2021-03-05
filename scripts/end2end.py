#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Efficient end2end QA with HNSW index

taskset --cpu-list 0-15 python end2end.py ../data/hotpot/hotpot_qas_val.json
"""
import argparse
import json
import logging
from functools import partial
import time

import argparse
import collections
import json
import logging
from torch.utils.data import DataLoader

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from retrieval.models.mhop_retriever import RobertaRetriever
from retrieval.utils.utils import load_saved

from qa.qa_model import QAModel
from mdr.qa.qa_dataset import qa_collate, QAEvalDataset
from .train_qa import eval_final
from qa.hotpot_evaluate_v1 import f1_score, exact_match_score
from qa.utils import set_global_logging_level

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

def convert_hnsw_query(query_vectors):
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('--indexpath', type=str, default="retrieval/index/wiki_index_hnsw_roberta")
    parser.add_argument('--corpus_dict', type=str, default='retrieval/index/hotpotQA_corpus_dict.json')
    parser.add_argument('--retriever_path', type=str, default="retrieval/logs/08-16-2020/roberta_momentum_freeze_k-seed16-bsz150-fp16True-lr1e-05-decay0.0-warm0-valbsz3000-m0.999-k76800/checkpoint_q_best.pt")
    parser.add_argument('--reader_path', type=str, default="qa/logs/08-10-2020/electra_val_top30-epoch7-lr5e-05-seed42-rdrop0-qadrop0-decay0-qpergpu2-aggstep8-clip2-evalper250-evalbsize1024-negnum5-warmup0.1-adamTrue-spweight0.025/checkpoint_best.pt")
    parser.add_argument('--topk', type=int, default=1, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-len', type=int, default=70)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument("--max-ans-len", default=35, type=int)
    parser.add_argument("--save-prediction", default="", type=str)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--sp-pred", action="store_true", help="whether to predict sentence sp")
    parser.add_argument("--sp-weight", default=0, type=float, help="weight of the sp loss")
    # parser.add_argument('--hnsw', action="store_true")
    args = parser.parse_args()

    logger.info("Loading trained models...")
    retrieval_config = AutoConfig.from_pretrained('roberta-base')
    retrieval_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    args.model_name = "roberta-base"
    retriever = RobertaRetriever(retrieval_config, args)
    retriever = load_saved(retriever, args.retriever_path)
    retriever.eval()
    
    qa_config = AutoConfig.from_pretrained('google/electra-large-discriminator')
    qa_tokenizer = AutoTokenizer.from_pretrained('google/electra-large-discriminator')
    args.model_name = "google/electra-large-discriminator"
    reader = QAModel(qa_config, args)
    reader = load_saved(reader, args.reader_path, False)
    reader.eval()

    logger.info("Loading index...")
    index = faiss.read_index(args.indexpath)

    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    logger.info(f"Corpus size {len(id2doc)}")

    logger.info("Loading queries...")
    qas_items = [json.loads(_) for _ in open(args.raw_data).readlines()[:5]]
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in qas_items]
    id2gold_ans = {_["_id"]: _["answer"][0] for _ in qas_items}


    start = time.time()
    logger.info("Retrieving...")
    retrieval_results = []

    encode_times = []
    search_times = []


    with torch.no_grad():
        for b_start in tqdm(range(0, len(questions), args.batch_size)):
            # 1-hop retrieval
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_qas = qas_items[b_start:b_start + args.batch_size]
            batch_q_encodes = retrieval_tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            q_embeds = retriever.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.numpy()
            q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)
            scores_1, docid_1 = index.search(q_embeds_numpy, args.topk)

            # construct 2hop queries
            bsize = len(batch_q)
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(docid_1[b_idx]):
                    doc = id2doc[str(doc_id)]["text"]
                    if doc.strip() == "":
                        # roberta tokenizer does not accept empty string as segment B
                        doc = id2doc[str(doc_id)]["title"]
                        scores_1[b_idx][_] = float("-inf")
                    query_pairs.append((batch_q[b_idx], doc))
            
            # 2-hop retrieval
            s1 = time.time()
            batch_q_sp_encodes = retrieval_tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            q_sp_embeds = retriever.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], batch_q_sp_encodes.get("token_type_ids", None))
            encode_times.append(time.time() - s1)

            s2 = time.time()
            q_sp_embeds = q_sp_embeds.numpy()
            q_sp_embeds = convert_hnsw_query(q_sp_embeds)
            scores_2, docid_2 = index.search(q_sp_embeds, args.topk)
            search_times.append(time.time() - s2)

            # aggregate chain scores
            scores_2 = scores_2.reshape(bsize, args.topk, args.topk)
            docid_2 = docid_2.reshape(bsize, args.topk, args.topk)
            path_scores = - (np.expand_dims(scores_1, axis=2) + scores_2)

            for idx in range(bsize):
                search_scores = path_scores[idx]
                ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], (args.topk, args.topk))).transpose()

                chains = []
                for _ in range(args.topk):
                    path_ids = ranked_pairs[_]
                    doc1_id = str(docid_1[idx, path_ids[0]])
                    doc2_id = str(docid_2[idx, path_ids[0], path_ids[1]])
                    chains.append([id2doc[doc1_id], id2doc[doc2_id]])

                retrieval_results.append({
                    "_id": batch_qas[idx]["_id"],
                    "question": batch_qas[idx]["question"],
                    "candidate_chains": chains
                })


    logger.info("Reading...")
    collate_fc = partial(qa_collate, pad_id=qa_tokenizer.pad_token_id)
    qa_eval_dataset = QAEvalDataset(qa_tokenizer, retrieval_results, max_seq_len=512, max_q_len=64)
    qa_eval_dataloader = DataLoader(qa_eval_dataset, batch_size=args.topk, collate_fn=collate_fc, pin_memory=True, num_workers=0)
    qa_results = eval_final(args, reader, qa_eval_dataloader, gpu=False)
    print(f"Finishing evaluation in {time.time() - start}s")


    ems = [exact_match_score(qa_results["answer"][k], id2gold_ans[k]) for k in qa_results["answer"].keys()]
    f1s = [f1_score(qa_results["answer"][k], id2gold_ans[k]) for k in qa_results["answer"].keys()]

    logger.info(f"Answer EM {np.mean(ems)}, F1 {np.mean(f1s)}")
