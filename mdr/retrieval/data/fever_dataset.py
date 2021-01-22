# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from torch import normal
from torch.utils.data import Dataset
import torch
import json
import random
import unicodedata
import re

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string

class FeverDataset(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_q_len,
        max_q_sp_len,
        max_c_len,
        train=False,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.max_q_sp_len = max_q_sp_len
        self.train = train
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]
        print(f"Total sample count {len(self.data)}")

    def encode_para(self, para, max_len):
        para["title"] = normalize(para["title"])
        # para["text"] = convert_brc(para["text"])

        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para["text"].strip(), max_length=max_len, return_tensors="pt")
    
    def __getitem__(self, index):
        sample = self.data[index]
        question = sample["claim"]

        evidence_multi = [e for e in sample["evidence"] if len(set([p["title"] for p in e])) > 1]
        neg_paras = sample["tfidf_neg"] + sample["linked_neg"]

        if self.train:
            random.shuffle(evidence_multi)
            random.shuffle(neg_paras)
        start_para, bridge_para = evidence_multi[0][0], evidence_multi[0][1]

        start_para_codes = self.encode_para(start_para, self.max_c_len)
        bridge_para_codes = self.encode_para(bridge_para, self.max_c_len)
        neg_codes_1 = self.encode_para(neg_paras[0], self.max_c_len)
        neg_codes_2 = self.encode_para(neg_paras[1], self.max_c_len)

        q_sp_codes = self.tokenizer.encode_plus(question, text_pair=start_para["text"].strip(), max_length=self.max_q_sp_len, return_tensors="pt")
        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")

        return {
                "q_codes": q_codes,
                "q_sp_codes": q_sp_codes,
                "start_para_codes": start_para_codes,
                "bridge_para_codes": bridge_para_codes,
                "neg_codes_1": neg_codes_1,
                "neg_codes_2": neg_codes_2,
                }

    def __len__(self):
        return len(self.data)

