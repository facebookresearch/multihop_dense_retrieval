# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import numpy as np

def explore(path):
    train = json.load(open(path))

    neg_counts = []
    for item in train:
        tfidf_neg = item["tfidf_neg"]
        linked_neg = item["linked_neg"]
        neg_counts.append(len(tfidf_neg + linked_neg))
        
    import pdb; pdb.set_trace()
    return 

def load_corpus(corpus_path="/private/home/xwhan/data/hotpot/tfidf/abstracts.txt"):
    content = [json.loads(l) for l in open(corpus_path).readlines()]
    title2doc = {item["title"]:item["text"] for item in content}

if __name__ == "__main__":
    explore("/private/home/xwhan/data/hotpot/hotpot_rerank_train_2_neg_types.json")