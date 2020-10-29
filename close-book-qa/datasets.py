from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import json
import numpy as np
import random
from tqdm import tqdm

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class QAPairs(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_q_length=300,
                 max_ans_length=50,
                 lower=True,
                 decode_bridge=False
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_ans_length = max_ans_length
        self.max_q_length = max_q_length
        self.lower = lower
        print(f"Loading QA data from {data_path}...")
        self.all_data = [json.loads(l) for l in open(data_path).readlines()]

        if decode_bridge:
            for idx in range(len(self.all_data)):
                self.all_data[idx]["answer"] = [self.all_data[idx]["bridge"]]
        
        # q_lens = [len(self.tokenizer.encode(_["question"])) for _ in self.all_data]
        # print(f"Max q len {np.max(q_lens)}, mean {np.mean(q_lens)}")
    
    def __getitem__(self, index):
        question = self.all_data[index]["question"]
        if len(self.all_data[index]["answer"]) > 1:
            answer = random.choice(self.all_data[index]["answer"])
        else:
            answer = self.all_data[index]["answer"][0]

        if "Bart" in self.tokenizer.__class__.__name__:
            question = "<s> " + question.lower()
            answer = "<s> " + answer.lower()
        else:
            question = question.lower() + " </s>"
            answer = answer.lower() + " </s>"

        q_encodings = self.tokenizer.encode_plus(question, return_tensors="pt", max_length=self.max_q_length)
        ans_encodings = self.tokenizer.encode_plus(answer, return_tensors="pt", max_length=self.max_ans_length)

        encoded = {
            "input_ids": q_encodings["input_ids"],
            "attention_mask": q_encodings["attention_mask"],
            "lm_labels": ans_encodings["input_ids"],
            "target_mask": ans_encodings["attention_mask"],
            "answer": self.all_data[index]["answer"],
            "type": self.all_data[index].get("type", "single"),
            "question": self.all_data[index]["question"]
        }

        if "level" in self.all_data[index]:
            encoded["level"] = self.all_data[index]["level"]
        return encoded

    def __len__(self):
        return len(self.all_data)

def collate(samples, pad_id=0, target_pad_id=-100):
    if len(samples) == 0:
        return {}

    batch = {
            "answers": [s["answer"] for s in samples],
            "types": [s["type"] for s in samples],
            "questions": [s["question"] for s in samples],
            "net_inputs": {
            'input_ids': collate_tokens([s['input_ids'].view(-1) for s in samples], pad_id),
            'attention_mask': collate_tokens([s['attention_mask'].view(-1) for s in samples], 0),
            'lm_labels': collate_tokens([s['lm_labels'].view(-1) for s in samples], target_pad_id),
            "target_mask": collate_tokens([s['target_mask'].view(-1) for s in samples], 0),
            }
        }

    if "level" in samples[0]:
        batch["levels"] = [s["level"] for s in samples]

    return batch