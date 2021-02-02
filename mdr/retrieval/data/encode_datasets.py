# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import csv
import json
import pdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import codecs
from .data_utils import collate_tokens
import unicodedata
import re
import os

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

class EmDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_q_len,
                 max_c_len,
                 is_query_embed,
                 save_path
                 ):
        super().__init__()
        self.is_query_embed = is_query_embed
        self.tokenizer = tokenizer
        self.max_c_len = max_c_len

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, "id2doc.json") # ID to doc mapping

        print(f"Loading data from {data_path}")
        if self.is_query_embed:
            self.data = [json.loads(_.strip())
                        for _ in tqdm(open(data_path).readlines())]
        else:
            if data_path.endswith("tsv"):
                self.data = []
                with open(data_path) as tsvfile:
                    reader = csv.reader(tsvfile, delimiter='\t', )
                    for row in reader:
                        if row[0] != 'id':
                            id_, text, title = row[0], row[1], row[2]
                            self.data.append({"id": id_, "text": text, "title": title})
            elif "fever" in data_path:
                raw_data = [json.loads(l) for l in tqdm(open(data_path).readlines())]
                self.data = []
                for _ in raw_data:
                #     _["title"] = normalize(_["title"])
                    # _["title"] = convert_brc(_["title"])
                    # _["text"] = convert_brc(_["text"])

                    self.data.append(_)
            else:
                self.data = [json.loads(l) for l in open(data_path).readlines()]
            print(f"load {len(self.data)} documents...")
            id2doc = {}
            for idx, doc in enumerate(self.data):
                id2doc[idx] = (doc["title"], doc["text"], doc.get("intro", False))
            with open(save_path, "w") as g:
                json.dump(id2doc, g)

        self.max_len = max_q_len if is_query_embed else max_c_len
        print(f"Max sequence length: {self.max_len}")


    def __getitem__(self, index):
        sample = self.data[index]

        if "Roberta" in self.tokenizer.__class__.__name__ and sample["text"].strip() == "":
            print(f"empty doc title: {sample['title']}")
            sample["text"] = sample["title"]
        # if sample["text"].endswith("."):
        #     sample["text"] = sample["text"][:-1]

        sent_codes = self.tokenizer.encode_plus(normalize(sample["title"].strip()), text_pair=sample['text'].strip(), max_length=self.max_len, return_tensors="pt")

        return sent_codes

    def __len__(self):
        return len(self.data)

def em_collate(samples):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([s['input_ids'].view(-1) for s in samples], 0),
        'input_mask': collate_tokens([s['attention_mask'].view(-1) for s in samples], 0),
    }

    if "token_type_ids" in samples[0]:
        batch["input_type_ids"] = collate_tokens([s['token_type_ids'].view(-1) for s in samples], 0)

    return batch
