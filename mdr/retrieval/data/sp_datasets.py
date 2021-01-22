# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
Dataset classes for NQ expeirments
"""

from torch.utils.data import Dataset
import json
import random
from .data_utils import collate_tokens

class SPDataset(Dataset):

    """
    strongerly supervised data, following DPR
    """

    def __init__(self,
        tokenizer,
        data_path,
        max_q_len,
        max_c_len,
        train=False,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.train = train
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]

        if isinstance(sample["pos_paras"], list):
            if self.train:
                pos_para = random.choice(sample["pos_paras"])
            else:
                pos_para = sample["pos_paras"][0]
            sample["pos_para"] = pos_para

        pos_title = sample['pos_para']['title'].strip()
        paragraph = sample['pos_para']['text'].strip()

        if self.train:
            random.shuffle(sample["neg_paras"])
        if len(sample["neg_paras"]) == 0:
            if self.train:
                neg_item = random.choice(self.data)

                if "pos_paras" in neg_item:
                    neg_item["pos_para"] = neg_item["pos_paras"][0]
            
                neg_title = neg_item["pos_para"]["title"].strip()
                neg_paragraph = neg_item["pos_para"]["text"].strip()
            else:
                neg_title = "dummy"
                neg_paragraph = "dummy"
        else:
            neg_title = sample['neg_paras'][0]['title'].strip()
            neg_paragraph = sample['neg_paras'][0]['text'].strip()
        neg_codes = self.tokenizer.encode_plus(neg_title, text_pair=neg_paragraph, max_length=self.max_c_len, return_tensors="pt")
        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")

        pos_codes = self.tokenizer.encode_plus(pos_title, text_pair=paragraph, max_length=self.max_c_len, return_tensors="pt")

        return {
                "q_codes": q_codes,
                "pos_codes": pos_codes,
                "neg_codes": neg_codes,
                }

    def __len__(self):
        return len(self.data)

import unicodedata
def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

class FeverSingleDataset(Dataset):

    """
    strongerly supervised data, following DPR
    """

    def __init__(self,
        tokenizer,
        data_path,
        max_q_len,
        max_c_len,
        train=False,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.train = train
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]

    def encode_para(self, para, max_len):
        para["title"] = normalize(para["title"])

        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para["text"].strip(), max_length=max_len, return_tensors="pt")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['claim']
        neg_paras = sample["tfidf_neg"] + sample["linked_neg"]
        evidence_titles = set()
        pos_paras = []
        for e in sample["evidence"]:
            for p in e:
                if p["title"] not in evidence_titles:
                    pos_paras.append(p)
                    evidence_titles.add(p["title"])
        if self.train:
            random.shuffle(neg_paras)
            random.shuffle(pos_paras)
        
        pos_para = pos_paras[0]
        if len(neg_paras) == 0:
            neg_para = {"title": "dummy", "text": "dummy"}
        else:
            neg_para = neg_paras[0]

        neg_codes = self.encode_para(neg_para, self.max_c_len)
        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")
        pos_codes = self.encode_para(pos_para, self.max_c_len)

        return {
                "q_codes": q_codes,
                "pos_codes": pos_codes,
                "neg_codes": neg_codes,
                }

    def __len__(self):
        return len(self.data)
        

class NQMhopDataset(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_q_sp_len,
        max_c_len,
        train=False,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_c_len = max_c_len
        self.max_q_sp_len = max_q_sp_len
        self.train = train
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]
        # if train:
        self.data = [_ for _ in self.data if len(_["top_neg"]) >= 2]
        print(f"Total sample count {len(self.data)}")

    def encode_para(self, para, max_len):
        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para["text"].strip(), max_length=max_len, return_tensors="pt")
    
    def encode_q(self, q, max_len, augment=True):
        q_toks = self.tokenizer.tokenize(q.strip())
        q_toks = q_toks[:max_len-2] # 2 special tokens
        if len(q_toks) < max_len - 2 and augment:
            # query augmentation
            q_toks = q_toks + [self.tokenizer.mask_token] * (max_len - 2 - len(q_toks))
        return self.tokenizer.encode_plus(q_toks, max_length=max_len, return_tensors="pt", is_pretokenized=True)


    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']

        if self.train:
            random.shuffle(sample["top_neg"])
        
        error_para = sample["top_neg"][0]
        pos_para = sample["pos_paras"][0]
        neg_para = sample["top_neg"][1]

        if error_para["text"].strip() == "":
            error_para["text"] = error_para["title"]
        q_codes = self.tokenizer.encode_plus(question, text_pair=error_para["text"].strip(), max_length=self.max_q_sp_len, return_tensors="pt")
       
        if pos_para["text"].strip() == "":
            pos_para["text"] = error_para["title"]
        pos_codes = self.tokenizer.encode_plus(pos_para["title"].strip(), text_pair=pos_para["text"].strip(), max_length=self.max_c_len, return_tensors="pt")

        if neg_para["text"].strip() == "":
            neg_para["text"] = neg_para["title"]
        neg_codes = self.tokenizer.encode_plus(neg_para["title"].strip(), text_pair=neg_para["text"].strip(), max_length=self.max_c_len, return_tensors="pt")

        return {
                "q_codes": q_codes,
                "pos_codes": pos_codes,
                "neg_codes": neg_codes,
                }

    def __len__(self):
        return len(self.data)


def sp_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
            'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'q_mask':collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], 0),
            'c_input_ids': collate_tokens([s["pos_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'c_mask': collate_tokens([s["pos_codes"]["attention_mask"].view(-1) for s in samples], 0),
            'neg_input_ids': collate_tokens([s["neg_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'neg_mask': collate_tokens([s["neg_codes"]["attention_mask"].view(-1) for s in samples], 0),
            
        }
    
    if "token_type_ids" in samples[0]["q_codes"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'c_type_ids': collate_tokens([s["pos_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'neg_type_ids': collate_tokens([s["neg_codes"]["token_type_ids"].view(-1) for s in samples], 0),
        })

    return batch


class MHopDataset(Dataset):

    """
    strongerly supervised data, following DPR
    """

    def __init__(self,
        tokenizer,
        data_path,
        max_q_len,
        max_c_len,
        train=False,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.train = train
        print(f"Loading data from {data_path}")

        self.data = [json.loads(line) for line in open(data_path).readlines()]
        if train:
            self.data = [_ for _ in self.data if len(_["neg_paras"]) >= 2]
        print(f"Total sample count {len(self.data)}")

        # q_lens = [len(self.tokenizer.encode(_["question"])) for _ in self.data]
        # 
        # print(f"Max q len {np.max(q_lens)}, mean {np.mean(q_lens)}")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        
        if sample["type"] == "bridge":
            # make sure bridge is in the second hop
            if sample['pos_paras'][0]["title"].strip() == sample["bridge"].strip():
                sample["pos_paras"] = sample["pos_paras"][::-1]

        if sample["type"] == "comparison":
            # if comparison, then the retrieval order does not matter
            random.shuffle(sample["pos_paras"])

        pos_title_1 = sample['pos_paras'][0]['title'].strip()
        paragraph_1 = sample['pos_paras'][0]['text'].strip()

        pos_title_2 = sample['pos_paras'][1]['title'].strip()
        paragraph_2 = sample['pos_paras'][1]['text'].strip()

        if self.train:
            random.shuffle(sample["neg_paras"])
        # if len(sample["neg_paras"]) == 0:
        #     if self.train:
        #         neg_item = random.choice(self.data)
        #         neg_title = neg_item["pos_paras"][0]["title"].strip()
        #         neg_paragraph = neg_item["pos_paras"][0]["text"].strip()
        #     else:
        #         neg_title = "dummy"
        #         neg_paragraph = "dummy"
        # else:

        neg_title_1 = sample['neg_paras'][0]['title'].strip()
        neg_paragraph_1 = sample['neg_paras'][0]['text'].strip()

        neg_title_2 = sample['neg_paras'][1]['title'].strip()
        neg_paragraph_2 = sample['neg_paras'][1]['text'].strip()

        # # assert neg_title != pos_title_1 and neg_title != pos_title_2
        neg_codes_1 = self.tokenizer.encode_plus(neg_title_1, text_pair=neg_paragraph_1, max_length=self.max_c_len, return_tensors="pt")

        neg_codes_2 = self.tokenizer.encode_plus(neg_title_2, text_pair=neg_paragraph_2, max_length=self.max_c_len, return_tensors="pt")
        
        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")

        pos_codes_1 = self.tokenizer.encode_plus(pos_title_1, text_pair=paragraph_1, max_length=self.max_c_len, return_tensors="pt")

        pos_codes_2 = self.tokenizer.encode_plus(pos_title_2, text_pair=paragraph_2, max_length=self.max_c_len, return_tensors="pt")
        
        return {
                "q_codes": q_codes,
                "pos_codes_1": pos_codes_1,
                "pos_codes_2": pos_codes_2,
                "neg_codes_1": neg_codes_1,
                "neg_codes_2": neg_codes_2,
                }

    def __len__(self):
        return len(self.data)

def mhop_collate(samples, pad_id=0):
    batch = {
            'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'q_mask':collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], 0),
            'c_input_ids_1': collate_tokens([s["pos_codes_1"]["input_ids"].view(-1) for s in samples], pad_id),
            'c_mask_1': collate_tokens([s["pos_codes_1"]["attention_mask"].view(-1) for s in samples], 0),
            'c_input_ids_2': collate_tokens([s["pos_codes_2"]["input_ids"].view(-1) for s in samples], pad_id),
            'c_mask_2': collate_tokens([s["pos_codes_2"]["attention_mask"].view(-1) for s in samples], 0),
            'neg_input_ids_1': collate_tokens([s["neg_codes_1"]["input_ids"].view(-1) for s in samples], pad_id),
            'neg_mask_1': collate_tokens([s["neg_codes_1"]["attention_mask"].view(-1) for s in samples], 0),
            'neg_input_ids_2': collate_tokens([s["neg_codes_2"]["input_ids"].view(-1) for s in samples], pad_id),
            'neg_mask_2': collate_tokens([s["neg_codes_2"]["attention_mask"].view(-1) for s in samples], 0),
            }

    if "token_type_ids" in samples[0]["q_codes"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'c_type_ids_1': collate_tokens([s["pos_codes_1"]["token_type_ids"].view(-1) for s in samples], 0),
            'c_type_ids_2': collate_tokens([s["pos_codes_2"]["token_type_ids"].view(-1) for s in samples], 0),
            'neg_type_ids_1': collate_tokens([s["neg_codes_1"]["token_type_ids"].view(-1) for s in samples], 0),
            'neg_type_ids_2': collate_tokens([s["neg_codes_2"]["token_type_ids"].view(-1) for s in samples], 0),
        })
    
    return batch
    