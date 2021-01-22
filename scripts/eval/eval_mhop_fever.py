# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
python eval_mhop_fever.py /private/home/xwhan/data/fever/retrieval/dev_multi_evidence.txt index/fever.npy index/fever_corpus_id2doc.json logs/08-27-2020/fever-seed16-bsz96-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue/checkpoint_best.pt --batch-size 100 --beam-size-1 2 --beam-size-2 10 --topk 20 --shared-encoder --model-name roberta-base --gpu --save-path dense_fever_b2_10_k20.json

# unified retrieval
python eval_mhop_fever.py /private/home/xwhan/data/fever/retrieval/dev.txt index/fever_unified.npy index/fever_corpus_id2doc.json logs/08-30-2020/fever_unified_roberta-seed16-bsz96-fp16True-lr2e-05-decay0.0-adamTrue/checkpoint_best.pt --batch-size 100 --beam-size-1 1 --beam-size-2 20 --topk 20 --shared-encoder --model-name roberta-base --gpu --save-path dense_all_b1_k10_unified.json


python eval_mhop_fever.py /private/home/xwhan/data/fever/retrieval/dev_multi_evidence.txt index/fever_unified.npy index/fever_corpus_id2doc.json logs/08-30-2020/fever_unified_roberta-seed16-bsz96-fp16True-lr2e-05-decay0.0-adamTrue/checkpoint_best.pt --batch-size 100 --beam-size-1 1 --beam-size-2 20 --topk 20 --shared-encoder --model-name roberta-base --gpu --save-path dense_multi_b1_k10_unified.json

# fix parenthesis
python eval_mhop_fever.py /private/home/xwhan/data/fever/retrieval/multi_dev.txt index/fever_.npy index/fever_corpus_id2doc.json logs/08-27-2020/fever_-seed16-bsz96-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue/checkpoint_best.pt --batch-size 100 --beam-size-1 1 --beam-size-2 20 --topk 20 --shared-encoder --model-name roberta-base --gpu --save-path dense_fever_b1_k20_fix_brc.json

"""
import argparse
import json
import logging
from os import path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.mhop_retriever import RobertaRetriever
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
    parser.add_argument('--max-q-len', type=int, default=45)
    parser.add_argument('--max-c-len', type=int, default=350)
    parser.add_argument('--max-q-sp-len', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size-1', type=int, default=5)
    parser.add_argument('--beam-size-2', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
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
    title2doc = {item[0]:item[1] for item in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")
    
    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RobertaRetriever(bert_config, args)
    # model = UnifiedRetriever(bert_config, args)
    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info("Encoding claims and searching")
    questions = [_["claim"] for _ in ds_items]
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

            D, I = index.search(q_embeds_numpy, args.beam_size_1)

            # 2hop search
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):
                    doc = id2doc[str(doc_id)][1]
                    if "roberta" in  args.model_name and doc.strip() == "":
                        # doc = "fadeaxsaa" * 100
                        doc = id2doc[str(doc_id)][0]
                        D[b_idx][_] = float("-inf")
                    query_pairs.append((batch_q[b_idx], doc))

            batch_q_sp_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")

            batch_q_sp_encodes = move_to_cuda(dict(batch_q_sp_encodes))
            q_sp_embeds = model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], batch_q_sp_encodes.get("token_type_ids", None))
            q_sp_embeds = q_sp_embeds.contiguous().cpu().numpy()
            # search_start = time.time()
            D_, I_ = index.search(q_sp_embeds, args.beam_size_2)
            # logger.info(f"MIPS searching: {time.time() - search_start}")
            D_ = D_.reshape(bsize, args.beam_size_1, args.beam_size_2)
            I_ = I_.reshape(bsize, args.beam_size_1, args.beam_size_2)

            # aggregate path scores
            path_scores = np.expand_dims(D, axis=2) + D_

            # path_scores = D_
            # eval
            for idx in range(bsize):
                search_scores = path_scores[idx]
                ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], (args.beam_size_1, args.beam_size_2))).transpose()
                retrieved_titles = []
                hop1_titles = []
                paths, path_titles = [], []
                paths_both_are_intro = []
                for _ in range(args.topk):
                    path_ids = ranked_pairs[_]
                    hop_1_id = I[idx, path_ids[0]]
                    hop_2_id = I_[idx, path_ids[0], path_ids[1]]
                    retrieved_titles.append(id2doc[str(hop_1_id)][0])
                    retrieved_titles.append(id2doc[str(hop_2_id)][0])

                    paths.append([str(hop_1_id), str(hop_2_id)])
                    path_titles.append([id2doc[str(hop_1_id)][0], id2doc[str(hop_2_id)][0]])
                    paths_both_are_intro.append(id2doc[str(hop_1_id)][2] and id2doc[str(hop_2_id)][2])
                    hop1_titles.append(id2doc[str(hop_1_id)][0])

                # saving when there's no annotations
                if args.save_path != "":
                    candidaite_chains = []
                    for path in path_titles:
                        candidaite_chains.append([(path[0], title2doc[path[0]]), (path[1], title2doc[path[1]])])
                    retrieval_outputs.append({
                        "id": batch_ann[idx]["id"],
                        "claim": batch_ann[idx]["claim"],
                        "candidate_chains": candidaite_chains,

                    })

    if args.save_path != "":
        with open(f"/private/home/xwhan/data/fever/retrieval/{args.save_path}", "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")

