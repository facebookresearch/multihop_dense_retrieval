# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from torch.utils.data import Dataset
import json
import random

from .data_utils import collate_tokens

class MhopDataset(Dataset):

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

            import pdb; pdb.set_trace()

            # debug TODO: remove for final release
            for idx in range(len(self.data)):
                self.data[idx]["neg_paras"] = self.data[idx]["tfidf_neg"]


            self.data = [_ for _ in self.data if len(_["neg_paras"]) >= 2]
        print(f"Total sample count {len(self.data)}")

    def encode_para(self, para, max_len):
        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para["text"].strip(), max_length=max_len, return_tensors="pt")
    
    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        if sample["type"] == "comparison":
            random.shuffle(sample["pos_paras"])
            start_para, bridge_para = sample["pos_paras"]
        else:
            for para in sample["pos_paras"]:
                if para["title"] != sample["bridge"]:
                    start_para = para
                else:
                    bridge_para = para
        if self.train:
            random.shuffle(sample["neg_paras"])

        start_para_codes = self.encode_para(start_para, self.max_c_len)
        bridge_para_codes = self.encode_para(bridge_para, self.max_c_len)
        neg_codes_1 = self.encode_para(sample["neg_paras"][0], self.max_c_len)
        neg_codes_2 = self.encode_para(sample["neg_paras"][1], self.max_c_len)

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

def mhop_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    
    batch = {
            'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], 0),
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

    if "sent_ids" in samples[0]["start_para_codes"]:
        batch["c1_sent_target"] = collate_tokens([s["start_para_codes"]["sent_ids"] for s in samples], -1)
        batch["c1_sent_offsets"] = collate_tokens([s["start_para_codes"]["sent_offsets"] for s in samples], 0),

    return batch
