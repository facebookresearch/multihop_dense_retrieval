# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import sys
import json
from tqdm import tqdm
import collections
import re
from collections import Counter
import string
from basic_tokenizer import SimpleTokenizer, filter_ngram
import csv

def pick_bridge_v0(title2linked, title2doc, titles, q, ans):
    """
    1. mainly based on if the passage includes the answer (assuming that only the 2nd hop passage has the answer)
    2. if 1 fails, then resort the linking structure, if A links to B, then B is the 
    """
    # check answer
    if (ans in titles[0] + " " + title2doc[titles[0]]) and ans not in titles[1] + " " + title2doc[titles[1]]:
        return titles[0]
    elif (ans in titles[1] + " " + title2doc[titles[1]]) and (ans not in titles[0] + " " + title2doc[titles[0]]):
        return titles[1]
    elif titles[0] in title2linked[titles[1]] and titles[1] not in title2linked[titles[0]]:
        return titles[0]
    else:
        return titles[1]

def load_annotated(path="/private/home/xwhan/data/hotpot/tfidf/abstracts.txt"):
    content = [json.loads(l) for l in open(path).readlines()]
    title2doc = {item["title"]:item["text"] for item in content}
    title2linked = {item["title"]:item["linked"] for item in content}
    return title2doc, title2linked

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


def hotpot_sp_data(raw_path):
    train = json.load(open(raw_path + '/hotpot_train_v1.1.json'))
    dev = json.load(open(raw_path + '/hotpot_dev_distractor_v1.json'))
    title2doc, title2linked = load_annotated()

    for split_name, split in {"train": train, "val": dev}.items():
        data_to_save = []
        for item in tqdm(split):
            title2passage = {_[0]: _[1] for _ in item["context"]}
            sp_titles = list(set([_[0] for _ in item["supporting_facts"]]))
            question = item["question"]
            if item["type"] == "comparison":
                pos_paras = []
                for title in sp_titles:
                    pos_paras.append({
                        "title": title,
                        "text": "".join(title2passage[title])
                    })
                data_to_save.append({
                    "question": question,
                    "pos_paras": pos_paras,
                    "neg_paras": [],
                    "type": item["type"],
                    "answers": item["answer"]
                })
            
            else:
                bridge = pick_bridge(title2linked, title2doc, sp_titles, question, item["answer"])
                if sp_titles[0] == bridge:
                    sp_titles = sp_titles[::-1]
                start, bridge = sp_titles[0], sp_titles[1]
                pos_paras = []
                for title in sp_titles:
                    pos_paras.append({
                        "title": title,
                        "text": "".join(title2passage[title])
                    })
                data_to_save.append({
                    "question": question,
                    "pos_paras": pos_paras,
                    "neg_paras": [],
                    "type": item["type"],
                    "answers": item["answer"],
                    "bridge": bridge
                })

        with open(raw_path + f"/hotpot_retrieval_{split_name}.json", "w") as g:
            for line in data_to_save:
                g.write(json.dumps(line) + "\n")


def add_qid(raw_path):
    title2doc, title2linked = load_annotated()
    raw_train = json.load(open(raw_path + '/hotpot_train_v1.1.json'))
    raw_val = json.load(open(raw_path + '/hotpot_dev_distractor_v1.json'))
    for s, raw_data in zip(["train", "val"], [raw_train, raw_val]):
        qas_data = []
        for item in raw_data:
            question = item["question"]
            _id = item["_id"]
            _type = item["type"]
            answer = [item["answer"]]
            sp = list(set([f[0] for f in item["supporting_facts"]]))
            if _type == "bridge":
                # make sure the sp order follows reasoning process
                bridge_title = pick_bridge_v0(title2linked, title2doc, sp, question, answer[0])
                if sp[0] == bridge_title:
                    sp = sp[::-1]
            qas_data.append({
                "question": question,
                "_id": _id,
                "answer": answer,
                "sp": sp,
                "type": _type
            })

        with open(raw_path + f"/hotpot_qas_{s}.json", "w") as g1:
            for _ in qas_data:
                g1.write(json.dumps(_) + "\n")

def add_bridge_ann(raw_path):
    title2doc, title2linked = load_annotated()

    raw_train = json.load(open(raw_path + '/hotpot_train_v1.1.json'))
    raw_val = json.load(open(raw_path + '/hotpot_dev_distractor_v1.json'))
    for split, raw in zip(["train", "val"], [raw_train, raw_val]):
        
        data = json.load(open(raw_path + f"/hotpot_{split}_with_neg.txt"))
        for idx, item in enumerate(data):
            assert item["question"] == raw[idx]["question"]
            item["_id"] = raw[idx]["_id"]
            if item["type"] == "bridge":
                ans = item["answers"][0]
                sp_titles = [p["title"] for p in item["pos_paras"]]
                bridge_title = pick_bridge_v0(title2linked, title2doc, sp_titles, item["question"], ans)
                item["bridge"] = bridge_title
        with open(raw_path + f"/hotpot_{split}_with_neg_v0.json", "w") as g:
            for _ in data:
                g.write(json.dumps(_) + "\n")
        # data = [json.loads(l) for l in open(raw_path + f"/hotpot_qas_{split}.json").readlines()]
        # for item in data:
        #     if item["type"] == "bridge":
        #         ans = item["answer"][0]
        #         sp_titles = item["sp"]
        #         bridge_title = pick_bridge(title2linked, title2doc, sp_titles, item["question"], ans)
        #         item["bridge"] = bridge_title
        # with open(raw_path + f"/hotpot_qas_{split}_bridge_label.json", "w") as g:
        #     for _ in data:
        #         g.write(json.dumps(_) + "\n")
import numpy as np

def check_2hop(raw_path):
    data = [json.loads(l) for l in open(raw_path + "/bridge_val.json").readlines()]

    target_in_query = [int(item["pos_para"]["title"].strip() in item["sp1"] or item["pos_para"]["title"].lower().strip() in item["sp1"]) for item in data]
    print(np.mean(target_in_query))


def add_sp_labels(raw_path, input_file, save_path,
                  title2sent_map="data/hotpot_index/title2sents.txt"):
    """
    Add sp sentence supervision for QA model training
    
    Inputs:
    raw_path: original HotpotQA data
    input_file: from MDR retrieval
    save_path: retrieved results with sentence level annotation
    """
    # raw_train = json.load(open(raw_path + '/hotpot_train_v1.1.json'))
    # train = [json.loads(l) for l in open(raw_path + "/dense_train_b100_k100.json").readlines()]
    raw_data = json.load(open(raw_path))
    retrieved = [json.loads(l) for l in open(input_file).readlines()]

    # title2sents
    title_and_sents = [json.loads(l) for l in open(title2sent_map).readlines()]
    title2sents = {_['title']:_['sents'] for _ in title_and_sents}

    for instance, raw in zip(retrieved, raw_data):
        assert instance["question"] == raw["question"]

        if "supporting_facts" in raw:
            orig_sp = raw["supporting_facts"]
            sptitle2sentids = collections.defaultdict(list)
            for _ in orig_sp:
                sptitle2sentids[_[0]].append(_[1])

            instance["sp"] = []

            for title in sptitle2sentids.keys():
                instance["sp"].append({"title": title, "sents": title2sents[title], "sp_sent_ids": sptitle2sentids[title]})
        
            instance["answer"] = [raw["answer"]]

    with open(save_path, "w") as out:
        for l in retrieved:
            out.write(json.dumps(l) + "\n")

def explore_QDMR(path="/private/home/xwhan/data/Break-dataset/QDMR-high-level"):
    """
    question decomposition from the BREAK dataset
    try to get the single-hop questions without introduce different question styles (hopefully the questions with have similar style as the original hotpotQA data), such the model is not only learning to distinguish the styles of the questions
    """

    hotpot_retrieval_train = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/hotpot_train_with_neg_v0.json").readlines()]
    hotpot_retrieval_val = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/hotpot_val_with_neg_v0.json").readlines()]
    qid2data = {}
    for item in hotpot_retrieval_train + hotpot_retrieval_val:
        qid2data[item["_id"]] = item

    for split in ["train", "dev"]:
        break_data = []
        with open(f"{path}/{split}.csv") as csvfile:
            reader = csv.reader(csvfile, quotechar='"', delimiter=',')
            for row in reader:
                if row[0] != "question_id":
                    q_id, orig_q, decom = row[0], row[1], row[2]
                    if q_id.startswith("HOTPOT"):
                        orig_split, orig_id = q_id.split("_")[1], q_id.split("_")[2]
                        sp_paras = qid2data[orig_id]["pos_paras"]
                        break_data.append({
                            "id": orig_id,
                            "split": orig_split,
                            "q": orig_q,
                            "q_decom": decom,
                            "sp": sp_paras,
                            "type": qid2data[orig_id]["type"]
                        })
            
        with open(f"/private/home/xwhan/data/QDMR/{split}.json", "w") as out:
            for _ in break_data:
                out.write(json.dumps(_) + "\n")



def add_sents_to_corpus_dict():
    id2doc = json.load(open("/private/home/xwhan/Mhop-Pretrain/retrieval/index/abstracts_id2doc.json"))
    title_and_sents = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/tfidf/title_sents.txt").readlines()]
    title2sents = {_['title']:_['sents'] for _ in title_and_sents}

    for k in id2doc.keys():
        title, text = id2doc[k][0], id2doc[k][1]
        sents = title2sents[title]
        id2doc[k] = {
            "title": title,
            "text": text,
            "sents": sents
        }
    
    json.dump(id2doc, open("/private/home/xwhan/Mhop-Pretrain/retrieval/index/hotpotQA_corpus_dict.json", "w"))

if __name__ == "__main__":
    original_hotpot_data, retrieved, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    add_sp_labels(original_hotpot_data, retrieved, output_path)
