# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from torch.utils.data import Dataset, Sampler
import torch
import json
import random

from .data_utils import collate_tokens

class UnifiedDataset(Dataset):

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
        if train:
            self.data = [_ for _ in self.data if len(_["neg_paras"]) >= 2]
        print(f"Total sample count {len(self.data)}")

    def encode_para(self, para, max_len):
        para_text = para["text"].strip()
        # NQ passages do not end with periods
        if para_text.endswith("."):
            para_text = para_text[:-1]
        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para_text, max_length=max_len, return_tensors="pt")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        mhop = True
        if sample["type"] == "comparison":
            random.shuffle(sample["pos_paras"])
            start_para, bridge_para = sample["pos_paras"]
        elif sample["type"] == "bridge":
            for para in sample["pos_paras"]:
                if para["title"] != sample["bridge"]:
                    start_para = para
                else:
                    bridge_para = para
        elif sample["type"] == "single":
            mhop = False
            assert len(sample["pos_paras"]) == 1
            start_para = sample["pos_paras"][0]
            if len(sample["neg_paras"]) > 0:
                bridge_para = random.choice(sample["neg_paras"]) # not used as positive
            else:
                bridge_para = {"title": "dummy", "text": "dummy"}
        else:
            assert False

        if self.train:
            random.shuffle(sample["neg_paras"])

        start_para_codes = self.encode_para(start_para, self.max_c_len)
        bridge_para_codes = self.encode_para(bridge_para, self.max_c_len)

        if len(sample["neg_paras"]) >= 2:
            neg_codes_1 = self.encode_para(sample["neg_paras"][0], self.max_c_len)
            neg_codes_2 = self.encode_para(sample["neg_paras"][1], self.max_c_len)
        else:
            if not sample["neg_paras"]:
                neg_codes_1 = self.encode_para({"title": "dummy", "text": "dummy"}, self.max_c_len)
            else:
                neg_codes_1 = self.encode_para(sample["neg_paras"][0], self.max_c_len)
            neg_codes_2 = self.encode_para({"title": "dummy", "text": "dummy"}, self.max_c_len)
        q_sp_codes = self.tokenizer.encode_plus(question, text_pair=start_para["text"].strip(), max_length=self.max_q_sp_len, return_tensors="pt")
        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")

        return {
                "q_codes": q_codes,
                "q_sp_codes": q_sp_codes,
                "start_para_codes": start_para_codes,
                "bridge_para_codes": bridge_para_codes,
                "neg_codes_1": neg_codes_1,
                "neg_codes_2": neg_codes_2,
                "stop": torch.LongTensor([int(mhop)]) # 0 to stop
                }

    def __len__(self):
        return len(self.data)

import unicodedata

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

class FeverUnifiedDataset(Dataset):

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
        
        self.single_ids = [idx for idx, _ in enumerate(self.data) if len(_["correct_normalized"]) == 1]
        self.multi_ids = [idx for idx, _ in enumerate(self.data) if len(_["correct_normalized"]) > 1]
        print(f"Total sample count {len(self.data)}")
        print(f"Total single-evidence count {len(self.single_ids)}")
        print(f"Total multi-evidence count {len(self.multi_ids)}")

    def encode_para(self, para, max_len):
        para["title"] = normalize(para["title"])

        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para["text"].strip(), max_length=max_len, return_tensors="pt")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['claim']

        mhop = len(sample["correct_normalized"]) > 1
        if mhop:
            neg_paras = sample["tfidf_neg"] + sample["linked_neg"]
            evidence_multi = [e for e in sample["evidence"] if len(set([p["title"] for p in e])) > 1]
            if self.train:
                random.shuffle(neg_paras)
                random.shuffle(evidence_multi)
            start_para, bridge_para = evidence_multi[0][0], evidence_multi[0][1]
        else:
            neg_paras = sample["tfidf_neg"] + sample["linked_neg"]
            evidence = sample["evidence"]
            if self.train:
                random.shuffle(neg_paras)
                random.shuffle(evidence)
            start_para = evidence[0][0]
            if len(neg_paras) == 0:
                neg_paras.append({"title": "dummy", "text": "dummy"})
            bridge_para = random.choice(neg_paras) # not used for training

        start_para_codes = self.encode_para(start_para, self.max_c_len)
        bridge_para_codes = self.encode_para(bridge_para, self.max_c_len)

        if len(neg_paras) >= 2:
            neg_codes_1 = self.encode_para(neg_paras[0], self.max_c_len)
            neg_codes_2 = self.encode_para(neg_paras[1], self.max_c_len)
        else:
            if not neg_paras:
                neg_codes_1 = self.encode_para({"title": "dummy", "text": "dummy"}, self.max_c_len)
            else:
                neg_codes_1 = self.encode_para(neg_paras[0], self.max_c_len)
            neg_codes_2 = self.encode_para({"title": "dummy", "text": "dummy"}, self.max_c_len)
        q_sp_codes = self.tokenizer.encode_plus(question, text_pair=start_para["text"].strip(), max_length=self.max_q_sp_len, return_tensors="pt")
        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")

        return {
                "q_codes": q_codes,
                "q_sp_codes": q_sp_codes,
                "start_para_codes": start_para_codes,
                "bridge_para_codes": bridge_para_codes,
                "neg_codes_1": neg_codes_1,
                "neg_codes_2": neg_codes_2,
                "stop": torch.LongTensor([int(mhop)]) # 0 to stop
                }

    def __len__(self):
        return len(self.data)

class FeverSampler(Sampler):
    """
    avoid the retrieval model to bias towards single-evidence claims
    the ratio for single/multi evidence 
    """

    def __init__(self, data_source, ratio=1):
        # for each QA pair, sample negative paragraphs
        self.single_ids = data_source.single_ids
        self.multi_ids = data_source.multi_ids
        self.ratio = ratio
        self._num_samples = len(self.multi_ids) * (ratio + 1)

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        random.shuffle(self.single_ids)
        sample_indice = self.multi_ids + self.single_ids[:len(self.multi_ids) * self.ratio]
        random.shuffle(sample_indice)
        return iter(sample_indice)

def unified_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    
    batch = {
            'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'q_mask':collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], 0),

            'q_sp_input_ids': collate_tokens([s["q_sp_codes"]["input_ids"].view(-1) for s in samples], 0),
            'q_sp_mask':collate_tokens([s["q_sp_codes"]["attention_mask"].view(-1) for s in samples], 0),

            'c1_input_ids': collate_tokens([s["start_para_codes"]["input_ids"] for s in samples], 0),
            'c1_mask': collate_tokens([s["start_para_codes"]["attention_mask"] for s in samples], 0),
                
            'c2_input_ids': collate_tokens([s["bridge_para_codes"]["input_ids"] for s in samples], 0),
            'c2_mask': collate_tokens([s["bridge_para_codes"]["attention_mask"] for s in samples], 0),

            'neg1_input_ids': collate_tokens([s["neg_codes_1"]["input_ids"] for s in samples], 0),
            'neg1_mask': collate_tokens([s["neg_codes_1"]["attention_mask"] for s in samples], 0),
            
            'neg2_input_ids': collate_tokens([s["neg_codes_2"]["input_ids"] for s in samples], 0),
            'neg2_mask': collate_tokens([s["neg_codes_2"]["attention_mask"] for s in samples], 0),
            
            'stop_targets': collate_tokens([s["stop"] for s in samples], 0)

        }

    if "token_type_ids" in samples[0]["q_codes"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'c1_type_ids': collate_tokens([s["start_para_codes"]["token_type_ids"] for s in samples], 0),
            'c2_type_ids': collate_tokens([s["bridge_para_codes"]["token_type_ids"] for s in samples], 0),
            "q_sp_type_ids": collate_tokens([s["q_sp_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'neg1_type_ids': collate_tokens([s["neg_codes_1"]["token_type_ids"] for s in samples], 0),
            'neg2_type_ids': collate_tokens([s["neg_codes_2"]["token_type_ids"] for s in samples], 0),
        })

    return batch


class NQUnifiedDataset(Dataset):
    """
    For each question, define two training targets 
    1. Q -> P_pos
    2. (Q, P_neg1) -> P_pos
    """
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
        # if train:
        self.data = [_ for _ in self.data if len(_["dpr_neg"]) > 0 and len(_["top_neg"]) > 1]

        print(f"Total sample count {len(self.data)}")

    def encode_para(self, para, max_len):
        para_text = para["text"].strip()
        if para_text == "":
            para_text = para["title"].strip()
        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para_text, max_length=max_len, return_tensors="pt")

    def encode_q(self, q):
        q_toks = self.tokenizer.tokenize(q)
        q_toks = ['[unused0]'] + q_toks
        return self.tokenizer.encode_plus(q_toks, max_length=self.max_q_len, return_tensors="pt", is_pretokenized=True)

    def encode_q_neg(self, q, neg):
        neg_para_toks = self.tokenizer.tokenize(neg["title"].strip() + " [SEP] " + neg["text"].strip())
        q_toks = ['[unused1]'] + self.tokenizer.tokenize(q)
        return self.tokenizer.encode_plus(q_toks, text_pair=neg_para_toks, max_length=self.max_q_sp_len, return_tensors="pt", is_pretokenized=True)

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        # assert len(sample["pos_paras"]) == 1

        if self.train:
            random.shuffle(sample["top_neg"])
            random.shuffle(sample["dpr_neg"])

            p_neg = sample["dpr_neg"][0]
            dense_neg1, dense_neg2 = sample["top_neg"][:2]
            # p_neg1, p_neg2 = sample["dpr_neg"][:2]

        else:
            p_neg = sample["dpr_neg"][0] if len(sample["dpr_neg"]) > 0 else {"title": "dummy", "text": "dummy"}

            dense_neg1, dense_neg2 = sample["top_neg"][:2]
            # p_neg2 = sample["dpr_neg"][1] if len(sample["dpr_neg"]) > 1 else {"title": "dummy", "text": "dummy"}

        # pos_para = sample["pos_paras"][0]

        if self.train:
            pos_para = random.choice(sample["pos_paras"])
        else:
            pos_para = sample["pos_paras"][0]

        # q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt")
        q_codes = self.encode_q(question)
        q_neg1_codes = self.encode_q_neg(question, dense_neg1)

        # q_neg1_codes = self.tokenizer.encode_plus(question, text_pair=dense_neg1["title"] + " [SEP] " + dense_neg1["text"].strip(), max_length=self.max_q_sp_len, return_tensors="pt")

        neg_codes = self.encode_para(p_neg, self.max_c_len)
        pos_codes = self.encode_para(pos_para, self.max_c_len)
        
        dense_neg1_codes = self.encode_para(dense_neg1, self.max_c_len)
        dense_neg2_codes = self.encode_para(dense_neg2, self.max_c_len)

        return {
                "q_codes": q_codes,
                "q_neg1_codes": q_neg1_codes,
                "neg_codes": neg_codes,
                "dense_neg1_codes": dense_neg1_codes,
                "dense_neg2_codes": dense_neg2_codes,
                "pos_codes": pos_codes
                }

    def __len__(self):
        return len(self.data)

def nq_unified_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    
    batch = {
            'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'q_mask':collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], 0),

            'q_neg1_input_ids': collate_tokens([s["q_neg1_codes"]["input_ids"].view(-1) for s in samples], 0),
            'q_neg1_mask':collate_tokens([s["q_neg1_codes"]["attention_mask"].view(-1) for s in samples], 0),

            'c_input_ids': collate_tokens([s["pos_codes"]["input_ids"] for s in samples], 0),
            'c_mask': collate_tokens([s["pos_codes"]["attention_mask"] for s in samples], 0),

            'neg_input_ids': collate_tokens([s["neg_codes"]["input_ids"] for s in samples], 0),
            'neg_mask': collate_tokens([s["neg_codes"]["attention_mask"] for s in samples], 0),

            'dense_neg1_input_ids': collate_tokens([s["dense_neg1_codes"]["input_ids"] for s in samples], 0),
            'dense_neg1_mask': collate_tokens([s["dense_neg1_codes"]["attention_mask"] for s in samples], 0),

            'dense_neg2_input_ids': collate_tokens([s["dense_neg2_codes"]["input_ids"] for s in samples], 0),
            'dense_neg2_mask': collate_tokens([s["dense_neg2_codes"]["attention_mask"] for s in samples], 0),
        
        }

    if "token_type_ids" in samples[0]["q_codes"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'c_type_ids': collate_tokens([s["pos_codes"]["token_type_ids"] for s in samples], 0),
            "q_neg1_type_ids": collate_tokens([s["q_neg1_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'neg_type_ids': collate_tokens([s["neg_codes"]["token_type_ids"] for s in samples], 0),
            'dense_neg1_type_ids': collate_tokens([s["dense_neg1_codes"]["token_type_ids"] for s in samples], 0),
            'dense_neg2_type_ids': collate_tokens([s["dense_neg2_codes"]["token_type_ids"] for s in samples], 0),
        })

    return batch

