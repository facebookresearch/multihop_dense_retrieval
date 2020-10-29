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

def pick_bridge_v1(title2linked, title2doc, titles, q, ans):
    """
    1.  mainly use the unigram/bigram overlap with the question to decide
    2.  back to the original rule v0
    """
    tok = SimpleTokenizer()
    q_grams = tok.tokenize(q).ngrams(n=2, uncased=True, filter_fn=filter_ngram)
    doc_0, doc_1 = title2doc[titles[0]], title2doc[titles[1]]
    doc_0_grams = tok.tokenize(doc_0).ngrams(n=2, uncased=True, filter_fn=filter_ngram)
    doc_1_grams = tok.tokenize(doc_1).ngrams(n=2, uncased=True, filter_fn=filter_ngram)
    doc_0_overlaps = len(set(q_grams) & set(doc_0_grams))
    doc_1_overlaps = len(set(q_grams) & set(doc_1_grams))
    if doc_0_overlaps > doc_1_overlaps:
        return titles[1]
    elif doc_0_overlaps < doc_1_overlaps:
        return titles[0]
    else:
        return pick_bridge_v0(title2linked, title2doc, titles, q, ans)

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

def num_same(a, b):
    a_tokens = normalize_answer(a).split()
    b_tokens = normalize_answer(b).split()
    common = Counter(a_tokens) & Counter(b_tokens)
    return sum(common.values())

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

def separate_type(raw_path):
    for split in ["train", "val"]:
        comparison = []
        bridge = []
        data = [json.loads(l) for l in open(raw_path + f"/hotpot_retrieval_{split}.json").readlines()]
        for item in data:
            if item["type"] == "comparison":
                comparison.append(item)
            else:
                bridge.append(item)
        with open(raw_path + f"/hotpot_retrieval_{split}_bridge.json", "w") as g1:
            json.dump(bridge, g1, indent=1)

        with open(raw_path + f"/hotpot_retrieval_{split}_comparison.json", "w") as g2:
            json.dump(comparison, g2, indent=1)

def separate_qas(raw_path):
    title2doc, title2linked = load_annotated()
    for split in ["train", "val"]:
        comparison = []
        bridge = []
        data = [json.loads(l) for l in open(raw_path + f"/hotpot_qas_{split}.json").readlines()]
        for item in data:
            if item["type"] == "comparison":
                comparison.append(item)
            else:
                # infer the bridge
                bridge_title = pick_bridge(title2linked, title2doc, item["sp"], item["question"], item["answer"][0])
                item["bridge"] = bridge_title 
                bridge.append(item)
        with open(raw_path + f"/hotpot_qas_{split}_bridge.json", "w") as g1:
            for _ in bridge:
                g1.write(json.dumps(_) + "\n")

        with open(raw_path + f"/hotpot_qas_{split}_comparison.json", "w") as g2:
            for _ in comparison:
                g2.write(json.dumps(_) + "\n")

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
                bridge_title = pick_bridge(title2linked, title2doc, sp, question, answer[0])
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

def add_sent_label(raw_path):
    """
    in order to use sentence info to train the retriever
    """
    # title2sents
    title_and_sents = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/tfidf/title_sents.txt").readlines()]
    title2para_sents = {_['title']:_['sents'] for _ in title_and_sents}

    raw_train = json.load(open(raw_path + '/hotpot_train_v1.1.json'))
    train_with_neg = [json.loads(l) for l in open(raw_path + "/hotpot_train_with_neg.json").readlines()]

    for idx in tqdm(range(len(train_with_neg))):
        assert raw_train[idx]["question"] == train_with_neg[idx]["question"]
        sp2sentids = collections.defaultdict(list)
        title2sents = {_[0]: _[1] for _ in raw_train[idx]["context"]}

        for sp in raw_train[idx]["supporting_facts"]:
            if sp[1] < len(title2sents[sp[0]]):
                sp2sentids[sp[0]].append(sp[1])
        for pos_para in train_with_neg[idx]["pos_paras"]:
            pos_para["sents"] = title2sents[pos_para["title"]]
            pos_para["sent_ids"] = sorted(sp2sentids[pos_para["title"]])
        for neg_para in train_with_neg[idx]["neg_paras"]:
            neg_para["sents"] = title2para_sents[neg_para["title"]]

    with open(raw_path + '/hotpot_train_with_neg_sent_ann.json', "w") as g:
        for item in train_with_neg:
            g.write(json.dumps(item) + "\n")

    raw_dev = json.load(open(raw_path + '/hotpot_dev_distractor_v1.json'))
    dev_with_neg = [json.loads(l) for l in open(raw_path + "/hotpot_val_with_neg.json").readlines()]
    for idx in tqdm(range(len(dev_with_neg))):
        assert raw_dev[idx]["question"] == dev_with_neg[idx]["question"]
        sp2sentids = collections.defaultdict(list)
        title2sents = {_[0]: _[1] for _ in raw_dev[idx]["context"]}

        for sp in raw_dev[idx]["supporting_facts"]:
            if sp[1] < len(title2sents[sp[0]]):
                sp2sentids[sp[0]].append(sp[1])
        for pos_para in dev_with_neg[idx]["pos_paras"]:
            pos_para["sents"] = title2sents[pos_para["title"]]
            pos_para["sent_ids"] = sorted(sp2sentids[pos_para["title"]])
        for neg_para in dev_with_neg[idx]["neg_paras"]:
            neg_para["sents"] = title2para_sents[neg_para["title"]]

    with open(raw_path + '/hotpot_val_with_neg_sent_ann.json', "w") as g:
        for item in dev_with_neg:
            g.write(json.dumps(item) + "\n")


def add_sp_labels(raw_path, use_existing_segmentation=True):
    """
    add sp sentence supervision for QA model training
    """
    # raw_train = json.load(open(raw_path + '/hotpot_train_v1.1.json'))
    # train = [json.loads(l) for l in open(raw_path + "/dense_train_b100_k100.json").readlines()]
    raw_dev = json.load(open(raw_path + '/hotpot_dev_distractor_v1.json'))
    dev = [json.loads(l) for l in open(raw_path + "/dense_val_b5_k5_best.json").readlines()]

    if not use_existing_segmentation:
        import stanza
        segmentor = stanza.Pipeline(lang='en', processors='tokenize')
    else:
        # title2sents
        title_and_sents = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/tfidf/title_sents.txt").readlines()]
        title2sents = {_['title']:_['sents'] for _ in title_and_sents}

    for instance, raw in zip(dev, raw_dev):
        assert instance["question"] == raw["question"]

        if "supporting_facts" in raw and use_existing_segmentation:
            orig_sp = raw["supporting_facts"]
            sptitle2sentids = collections.defaultdict(list)
            for _ in orig_sp:
                sptitle2sentids[_[0]].append(_[1])

            instance["sp"] = []

            for title in sptitle2sentids.keys():
                instance["sp"].append({"title": title, "sents": title2sents[title], "sp_sent_ids": sptitle2sentids[title]})
        
            instance["answer"] = [raw["answer"]]

        # for chain in instance["candidate_chains"]:
        #     for idx, _ in enumerate(chain):
        #         if use_existing_segmentation:
        #             sents = title2sents[_[0]]
        #         else:
        #             doc = segmentor(_[1])
        #             sents = [s.text for s in doc.sentences]
        #         chain[idx] = {"title": _[0], "sents": sents}

    with open(raw_path + "/dense_val_b5_k5_best_sents.json", "w") as out:
        for l in dev:
            out.write(json.dumps(l) + "\n")
            

def inspect_easy(path="/private/home/xwhan/data/hotpot"):
    train_raw = json.load(open(path + "/hotpot_train_v1.1.json"))
    train = [json.loads(l) for l in open(path + "/hotpot_train_with_neg.json").readlines()]
    tok = SimpleTokenizer()
    easy_count = 0
    non_easy, easy = [], []
    for raw, ann in zip(train_raw, train):
        assert raw["question"] == ann["question"]
        question = raw["question"][:-1] if raw["question"].endswith("?") else raw["question"]
        title2p = {_[0]:"".join(_[1]) for _ in raw["context"]}
        answer = ann["answers"][0]
        if raw["level"] == "easy":
            sp_titles = list(set([_[0] for _ in raw["supporting_facts"]]))
            sp1 = title2p[sp_titles[0]]
            sp2 = title2p[sp_titles[1]]
            if answer in sp1 and answer not in sp2:
                gold_sp = sp_titles[0]
            elif answer in sp2 and answer not in sp1:
                gold_sp = sp_titles[1]
            else:
                q_grams = tok.tokenize(question).ngrams(n=2, uncased=True)
                sp1_grams = tok.tokenize(sp1).ngrams(n=2, uncased=True)
                sp2_grams = tok.tokenize(sp2).ngrams(n=2, uncased=True)
                sp1_overlaps = len(set(q_grams) & set(sp1_grams))
                sp2_overlaps = len(set(q_grams) & set(sp2_grams))
                if sp1_overlaps > sp2_overlaps:
                    gold_sp = sp_titles[0]
                else:
                    gold_sp = sp_titles[1]

            ann["type"] = "single"
            ann["pos_paras"] = [{"title": gold_sp, "text": title2p[gold_sp]}]
            import pdb; pdb.set_trace()
            
        else:
            non_easy.append(ann)

        easy_count += 1

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

# def hotpot_test():
#     test_data = json.load(open("/private/home/xwhan/data/hotpot/hotpot_test_fullwiki_v1.json"))
#     test_qas = []
#     for item in test_data:
#         test_qas.append({
#             "_id": item["_id"],
#             "question": item["question"]
#         })

#     print(len(test_qas), len(test_data))
#     with open("/private/home/xwhan/data/hotpot/hotpot_qas_test.json", "w") as g:
#         for l in test_qas:
#             g.write(json.dumps(l) + "\n")
    # import pdb; pdb.set_trace()

def split_sents(raw_path, use_existing_segmentation=True):

    dev = [json.loads(l) for l in open(raw_path + "/dense_hotpot_test_b250_k250_roberta_best.json").readlines()]

    if not use_existing_segmentation:
        import stanza
        segmentor = stanza.Pipeline(lang='en', processors='tokenize')
    else:
        # title2sents
        title_and_sents = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/tfidf/title_sents.txt").readlines()]
        title2sents = {_['title']:_['sents'] for _ in title_and_sents}

        for instance in dev:
            for chain in instance["candidate_chains"]:
                for idx, _ in enumerate(chain):
                    if use_existing_segmentation:
                        sents = title2sents[_[0]]
                    else:
                        doc = segmentor(_[1])
                        sents = [s.text for s in doc.sentences]
                    chain[idx] = {"title": _[0], "sents": sents}
    with open(raw_path + "/dense_hotpot_test_b250_k250_roberta_best_sents.json", "w") as out:
        for l in dev:
            out.write(json.dumps(l) + "\n")


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
    # hotpot_sp_data("/private/home/xwhan/data/hotpot")
    # separate_qas("/private/home/xwhan/data/hotpot")
    # add_bridge_ann("/private/home/xwhan/data/hotpot")

    # explore_QDMR()

    # check_2hop("/private/home/xwhan/data/2-hop-retrieval")
    # add_sent_label("/private/home/xwhan/data/hotpot")
    # add_qid("/private/home/xwhan/data/hotpot")
    add_sp_labels("/private/home/xwhan/data/hotpot")
    
    # inspect_easy()

    # split_sents("/private/home/xwhan/data/hotpot")

    # hotpot_test()

    # add_sents_to_corpus_dict()
