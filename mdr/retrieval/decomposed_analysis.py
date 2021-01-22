# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import json


def decomposed_errors():
    top1_pred = [json.loads(l) for l in open("/private/home/xwhan/data/hotpot/dense_val_b1_top1.json").readlines()]
    analysis_folder = "/private/home/xwhan/data/hotpot/analysis"

    start_errors, bridge_errors, failed = [], [], []
    correct = []
    for item in top1_pred:
        pred_titles = [_[0] for _ in item["candidate_chains"][0]]
        gold_titles = [_[0] for _ in item["sp"]]
        if set(pred_titles) == set(gold_titles):
            if item["type"] == "bridge":
                correct.append(item)
            continue
        if item["type"] == "bridge":
            start_title = None
            for t in gold_titles:
                if t != item["bridge"]:
                    start_title = t
            assert start_title is not None
            if item["bridge"] in pred_titles and start_title not in pred_titles:
                start_errors.append(item)
            elif item["bridge"] not in pred_titles and start_title in pred_titles:
                bridge_errors.append(item)
            else:
                failed.append(item)

    with open(analysis_folder + "/correct.json", "w") as g:
        for _ in correct:
            _["predicted"] = _.pop("candidate_chains")[0]
            g.write(json.dumps(_) + "\n")

    with open(analysis_folder + "/start_errors.json", "w") as g:
        for _ in start_errors:
            _["predicted"] = _.pop("candidate_chains")[0]
            g.write(json.dumps(_) + "\n")

    with open(analysis_folder + "/bridge_errors.json", "w") as g:
        for _ in bridge_errors:
            _["predicted"] = _.pop("candidate_chains")[0]
            g.write(json.dumps(_) + "\n")

    with open(analysis_folder + "/total_errors.json", "w") as g:
        for _ in failed:
            _["predicted"] = _.pop("candidate_chains")[0]
            g.write(json.dumps(_) + "\n")


    print(len(correct))
    print(len(start_errors))
    print(len(bridge_errors))
    print(len(failed))

import random
def collect_gold_decomposition():
    """
    interactively collect
    """
    dev_qdmr = [json.loads(l) for l in open("/private/home/xwhan/data/QDMR/dev.json").readlines()]
    bridge_dev = [_ for _ in dev_qdmr if _["type"] == "bridge"]

    random.shuffle(bridge_dev)
    idx = 0
    samples_to_inspect = []
    while True:
        print(f"\n-----{len(samples_to_inspect)} samples collected so far-----")
        sample = bridge_dev[idx]
        idx += 1
        print(f"Original Q: {sample['q']}")
        print(f"Decomposed Q: {sample['q_decom']}")
        print(f"Supporting Passages: {sample['sp']}")
        subq1 = input("Type SUB Q1:")
        if subq1 == "bad":
            continue
        elif subq1 == "stop":
            break
        subq2 = input("Type SUB Q2:")
        samples_to_inspect.append({
            "id": sample["id"],
            "sp": sample["sp"],
            "orig_q": sample['q'],
            "subQ_1": subq1,
            "subQ_2": subq2
        })

    print(f"{len(samples_to_inspect)} samples collected in total..")

    with open("/private/home/xwhan/data/QDMR/inspect.json", "w") as g:
        for _ in samples_to_inspect:
            g.write(json.dumps(_) + "\n")

def qdmr_utils():
    """
    change file format for decomposed and end-to-end retrieval
    """
    qdmr_data = [json.loads(l) for l in open("/private/home/xwhan/data/QDMR/inspect.json").readlines()]

    mhop_data, decomposed_data = [], []
    for idx, item in enumerate(qdmr_data):
        if idx in [65,66,67]:
            continue
        sp = [_["title"] for _ in item["sp"]]
        question = item["orig_q"]
        mhop_data.append({
            "question": question,
            "sp": sp,
            "type": "bridge",
            "_id": item["id"]
        })
        decomposed_data.append(item)

    # with open("/private/home/xwhan/data/QDMR/qdmr_decomposed.json", "w") as g:
    #     for item in decomposed_data:
    #         g.write(json.dumps(item) + "\n")

    with open("/private/home/xwhan/data/QDMR/qdmr_e2e.json", "w") as g:
        for item in mhop_data:
            g.write(json.dumps(item) + "\n")


def analyze_results():
    decomposed_results = [json.loads(l) for l in open("/private/home/xwhan/data/QDMR/qdmr_decomposed_results.json")]
    e2e_results = [json.loads(l) for l in open("/private/home/xwhan/data/QDMR/qdmr_e2e_results.json")]
    better = 0
    worse = 0
    both = 0
    for res1, res2 in zip(decomposed_results, e2e_results):
        sp_titles = set([_[0] for _ in res1["sp"]])

        res1_top1 = [_[0] for _ in res1["candidate_chains"][0]]
        res2_top1 = [_[0] for _ in res2["candidate_chains"][0]]

        assert res1["_id"] == res2["_id"]

        question = res1["question"]
        q_pairs = res1["q_pairs"]

        if set(res2_top1) == sp_titles and set(res1_top1) != sp_titles:
            # print(sp_titles)
            # import pdb; pdb.set_trace()
            better += 1
        elif set(res2_top1) != sp_titles and set(res1_top1) == sp_titles:
            worse += 1
        elif set(res2_top1) == sp_titles and set(res1_top1) == sp_titles:
            both += 1

    print(both)
    print(better)
    print(worse)
    print(len(decomposed_results))

if __name__ == "__main__":
    # collect_gold_decomposition()
    # qdmr_utils()

    analyze_results()
