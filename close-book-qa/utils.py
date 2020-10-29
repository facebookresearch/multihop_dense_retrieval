import torch
import sqlite3
import unicodedata

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def convert_to_half(sample):
    if len(sample) == 0:
        return {}

    def _convert_to_half(maybe_floatTensor):
        if torch.is_tensor(maybe_floatTensor) and maybe_floatTensor.type() == "torch.FloatTensor":
            return maybe_floatTensor.half()
        elif isinstance(maybe_floatTensor, dict):
            return {
                key: _convert_to_half(value)
                for key, value in maybe_floatTensor.items()
            }
        elif isinstance(maybe_floatTensor, list):
            return [_convert_to_half(x) for x in maybe_floatTensor]
        else:
            return maybe_floatTensor

    return _convert_to_half(sample)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

import json
from tqdm import tqdm

def complex_qas(path):
    for split in ["train", "dev"]:
        qas = []
        data = json.load(open(path + f"/ComplexWebQuestions_{split}.json"))
        for item in tqdm(data):
            answer = []
            for ans in item["answers"]:
                answer.append(ans["answer"])
                answer += ans["aliases"]
            if None in answer and split == "train":
                continue
            qas.append({
                "question": item["question"],
                "id": item["ID"],
                "type": item["compositionality_type"],
                "answer": answer
            })
        with open(path + f"/complexwebq_{split}_qas.txt", "w") as g:
            for _ in tqdm(qas):
                g.write(json.dumps(_) + "\n")

if __name__ == "__main__":
    complex_qas("/private/home/xwhan/data/ComplexWebQ")