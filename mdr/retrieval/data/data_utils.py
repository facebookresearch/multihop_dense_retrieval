# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import collections
import pdb
import re
import string

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
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

import json

def simplify_dpr_data():
    train = json.load(open("/private/home/xwhan/code/DPR/data/retriever/nq-train.json"))
    dev = json.load(open("/private/home/xwhan/code/DPR/data/retriever/nq-dev.json"))

    for s, d in {"train": train, "val": dev}.items():
        data = []
        for item in d:
            data.append({
                "question": item["question"],
                "answer": item["answers"],
                "pos_paras": [item["positive_ctxs"][0]],
                "neg_paras": item["hard_negative_ctxs"]
            })
        print(len(data))
        with open(f"/private/home/xwhan/data/nq-dpr/nq-{s}-simplified.txt", "w") as g:
            for _ in data:
                g.write(json.dumps(_) + "\n")

def combine():
    """
    combine HotpotQA and NQ for a unified model
    """
    nq_train = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/nq-train-simplified.txt").readlines()]
    nq_dev = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/nq-val-simplified.txt").readlines()]

    combined_train = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/hotpot_train_with_neg_v0.json").readlines()]
    combined_dev = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/hotpot_val_with_neg_v0.json").readlines()]

    for item in nq_train:
        combined_train.append({
            "question":item["question"],
            "pos_paras": item["pos_paras"],
            "neg_paras": item["neg_paras"],
            "type": "single",
            "answer": item["answer"]
        })

    for item in nq_dev:
        combined_dev.append({
            "question":item["question"],
            "pos_paras": item["pos_paras"],
            "neg_paras": item["neg_paras"],
            "type": "single",
            "answer": item["answer"]
        })

    with open("/private/home/xwhan/data/combined/combined_train.json", "w") as out:
        for l in combined_train:
            out.write(json.dumps(l) + "\n")

    with open("/private/home/xwhan/data/combined/combined_val.json", "w") as out1:
        for l in combined_dev:
            out1.write(json.dumps(l) + "\n")

import collections
from tqdm import tqdm

def combine_corpus():
    hotpot_abstracts = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/tfidf/abstracts.txt").readlines()]
    hotpot_title2doc = {doc["title"]: doc["text"] for doc in hotpot_abstracts}
    nq_title2docs = collections.defaultdict(list)
    import csv
    dpr_count = 0
    with open("/private/home/xwhan/code/DPR/data/wikipedia_split/psgs_w100.tsv") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        for row in reader:
            if row[0] != 'id':
                id_, text, title = row[0], row[1], row[2]
                dpr_count += 1
                nq_title2docs[title].append(text)

    merged = []
    for title, passages in tqdm(nq_title2docs.items()):
        if title in hotpot_title2doc:
            # compare the length of 1st split of dpr and abstracts
            abstract = hotpot_title2doc[title].strip()
            merged.append({
                "title": title,
                "text": abstract[:-1] if abstract.endswith(".") else abstract,
                "intro": True
            })

        for idx, p in enumerate(passages):
            p = p.strip()
            merged.append({
                "title": title,
                "text": p[:-1] if p.endswith(".") else p,
                "intro": idx == 0
            })

    for title, doc in hotpot_title2doc.items():
        if title not in nq_title2docs:
            if doc.endswith("."):
                doc = doc[:-1]
            merged.append({
                "title": title,
                "text": doc,
                "intro": True
            })

    print(f"Merged corpus size {len(merged)}")
    with open("/private/home/xwhan/data/combined/corpus/merged_no_period.txt", "w") as g:
        for item in merged:
            g.write(json.dumps(item) + "\n")

def combine_questions():
    hotpot_val = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/hotpot_qas_val.json").readlines()]
    nq_val = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/nq-dev-qas.txt").readlines()]
    for idx, item in enumerate(nq_val):
        item["type"] = "single"
        item["_id"] = f"nq_{idx}"

    import pdb; pdb.set_trace()
    with open("/private/home/xwhan/data/combined/combined_qas_val.txt", "w") as g:
        for item in hotpot_val + nq_val:
            g.write(json.dumps(item) + "\n")

def nq_multihop():
    """
    experiments with nq multihop:
    try to recover from error cases with recursive dense retrieval
    """
    train_data = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/results/nq-train-shared-dpr-top100.txt").readlines()]
    val_data = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/results/nq-val-shared-dpr-top100.txt").readlines()]
    
    train_ = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/nq-train-simplified.txt")]
    val_ = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/nq-val-simplified.txt")]

    for split, data, orig in zip(["train", "val"], [train_data, val_data], [train_, val_]):
        data_recursive = []
        for idx, item in enumerate(data):
            assert item["question"] == orig[idx]["question"]
            pos_paras = orig[idx]["pos_paras"]
            top_neg = []
            for para, label in item["topk"]:
                if label == 0:
                    top_neg.append(para)
            data_recursive.append({
                "question": item["question"],
                "ans": item["ans"],
                "dpr_neg": orig[idx]["neg_paras"],
                "top_neg": top_neg,
                "pos_paras": pos_paras,
            })

        print(len(data_recursive))
        with open(f"/private/home/xwhan/data/nq-dpr/nq-mhop/nq-mhop-{split}-dpr-shared-top100.txt", "w") as g:
            for _ in data_recursive:
                g.write(json.dumps(_) + "\n")

def webQdata_simplify():
    train_data = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/results/wq-train-shared-dpr-top100.txt").readlines()]
    val_data = [json.loads(l) for l in open("/private/home/xwhan/data/nq-dpr/results/wq-dev-shared-dpr-top100.txt").readlines()]
    
    train_ = [json.loads(l) for l in open("/private/home/xwhan/data/WebQ/wq-train-simplified.txt")]
    val_ = [json.loads(l) for l in open("/private/home/xwhan/data/WebQ/wq-dev-simplified.txt")]

    for split, data, orig in zip(["train", "val"], [train_data, val_data], [train_, val_]):
        data_recursive = []
        for idx, item in enumerate(data):
            assert item["question"] == orig[idx]["question"][:-1]
            pos_paras = orig[idx]["pos_paras"]
            top_neg = []
            for para, label in item["topk"]:
                if label == 0:
                    top_neg.append(para)
            data_recursive.append({
                "question": item["question"],
                "ans": item["ans"],
                "dpr_neg": orig[idx]["neg_paras"],
                "top_neg": top_neg,
                "pos_paras": pos_paras,
            })

        print(len(data_recursive))
        with open(f"/private/home/xwhan/data/WebQ/wq-mhop/wq-mhop-{split}-dpr-shared-top100.txt", "w") as g:
            for _ in data_recursive:
                g.write(json.dumps(_) + "\n")

if __name__ == "__main__":
    # combine()
    # simplify_dpr_data()

    # combine_corpus()
    # combine_questions()

    # nq_multihop()

    webQdata_simplify()
