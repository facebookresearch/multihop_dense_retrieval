# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import json
import numpy as np
from utils.utils import para_has_answer, normalize
from utils.basic_tokenizer import SimpleTokenizer
from tqdm import tqdm


corpus = json.load(open("../data/hotpot_index/wiki_id2doc.json"))
title2text = {v["title"]:v["text"] for v in corpus.values()}

val_inputs = [json.loads(l) for l in open("../data/hotpot/hotpot_qas_val.json").readlines()]
id2goldsp = {_["_id"]:_["sp"] for _ in val_inputs}
id2goldans = {_["_id"]:_["answer"] for _ in val_inputs}
id2type = {_["_id"]:_["type"] for _ in val_inputs}



simple_tokenizer = SimpleTokenizer()

# out best results
results = json.load(open("../data/hotpot/results/hotpot_val_top100.json"))

# # asai results
# results = json.load(open("/private/home/xwhan/code/learning_to_retrieve_reasoning_paths/results/hotpot_dev_reader_titles.json"))
# for k in results.keys():
#     v = results[k]
#     v = [normalize(_[:-2]) for _ in v]
#     # import pdb; pdb.set_trace()
#     results[k] = v
# results = {"titles":results}

sp_ems = []
ans_recalls = []
bridge_ems = []
compare_ems = []
for qid in tqdm(results["titles"].keys()):
    chain = results["titles"][qid]
    sp = id2goldsp[qid]
    answer = id2goldans[qid]
    type_ = id2type[qid]

    # if answer[0].strip() in ["yes", "no"]:
    #     continue

    sp_covered = int(np.sum([int(_ in chain) for _ in sp]) == len(sp))
    concat_p = "yes no " + " ".join([t + " " + title2text.get(t, "") for t in chain])
    ans_covered = para_has_answer(answer, concat_p, simple_tokenizer)
    ans_recalls.append(ans_covered)
    sp_ems.append(sp_covered)

    if type_ == "bridge":
        bridge_ems.append(sp_covered)
    else:
        compare_ems.append(sp_covered)
    

print(len(sp_ems))
print(np.mean(sp_ems))
print(f"Answer Recall: {np.mean(ans_recalls)}, count: {len(ans_recalls)}")

print("Bridge P EM:", np.mean(bridge_ems))
print("Comparison P EM:", np.mean(compare_ems))