# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
python eval_single_fever.py /private/home/xwhan/data/fever/retrieval/dev_single_evidence.txt index/fever_single.npy index/fever_corpus_id2doc.json logs/08-30-2020/fever_single-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0-bert-base-uncased/checkpoint_best.pt --batch-size 1000 --shared-encoder --model-name bert-base-uncased --gpu --save-path dense_dev_single_k10.json --topk 10


python eval_single_fever.py /private/home/xwhan/data/fever/retrieval/dev_single_evidence.txt index/fever_unified.npy index/fever_corpus_id2doc.json logs/08-30-2020/fever_unified_roberta-seed16-bsz96-fp16True-lr2e-05-decay0.0-adamTrue/checkpoint_best.pt --batch-size 1000 --shared-encoder --model-name roberta-base --gpu --save-path dense_unified_dev_single_k10.json --topk 10

"""
import argparse
import json
import logging

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.retriever import BertRetrieverSingle
from models.unified_retriever import UnifiedRetriever
from utils.basic_tokenizer import SimpleTokenizer
from utils.utils import (load_saved, move_to_cuda)

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
    parser.add_argument('--batch-size', type=int, default=100)
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
    # model = BertRetrieverSingle(bert_config, args)
    model = UnifiedRetriever(bert_config, args)

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

            D, I = index.search(q_embeds_numpy, args.topk)

            for b_idx in range(bsize):
                topk_docs = []
                for _, doc_id in enumerate(I[b_idx]):
                    doc = id2doc[str(doc_id)]
                    topk_docs.append({"title": doc[0], "text": doc[1]})

                # saving when there's no annotations
                if args.save_path != "":
                    candidaite_chains = []
                    retrieval_outputs.append({
                        "id": batch_ann[b_idx]["id"],
                        "claim": batch_ann[b_idx]["claim"],
                        "topk": topk_docs,
                    })

    if args.save_path != "":
        with open(f"/private/home/xwhan/data/fever/retrieval/{args.save_path}", "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")

