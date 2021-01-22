# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from models.mhop_retriever import MhopRetriever


import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
import json
import logging
import argparse
from .utils.utils import (load_saved, move_to_cuda)

parser = argparse.ArgumentParser()
parser.add_argument('--topk', type=int, default=2, help="topk paths")
parser.add_argument('--num-workers', type=int, default=10)
parser.add_argument('--max-q-len', type=int, default=70)
parser.add_argument('--max-c-len', type=int, default=300)
parser.add_argument('--max-q-sp-len', type=int, default=350)
parser.add_argument('--model-name', type=str, default='bert-base-uncased')
parser.add_argument('--gpu', action="store_true")
parser.add_argument('--shared-encoder', action="store_true")
parser.add_argument("--stop-drop", default=0, type=float)
args = parser.parse_args()

index_path = "index/abstracts_v0_fixed.npy"
corpus_path = "index/abstracts_id2doc.json"
model_path = "logs/08-05-2020/baseline_v0_fixed-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt"

print(f"Loading corpus and index...")
id2doc = json.load(open(corpus_path))
index_vectors = np.load(index_path).astype('float32')

index = faiss.IndexFlatIP(768)
index.add(index_vectors)
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 1, index)

print(f"Loading retrieval model...")
bert_config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MhopRetriever(bert_config, args)
model = load_saved(model, args.model_path, exact=False)

cuda = torch.device('cuda')
model.to(cuda)
from apex import amp
model = amp.initialize(model, opt_level='O1')
model.eval()

while True:
    question = input("Type Question:")
    question = "the Danish musicians who died in 1931"
    batch_q_encodes = tokenizer.batch_encode_plus(["question"], max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
    batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
    q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
    q_embeds_numpy = q_embeds.cpu().contiguous().numpy() 
    D, I = index.search(q_embeds_numpy, 1)

    print(I)

