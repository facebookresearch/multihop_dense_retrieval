
from hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp
import collections
import numpy as np
import json

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

id2answer = json.load(open("results/id2pred.json"))
id2gold = json.load(open("results/id2gold.json"))

id2agg = {}
for qid, pred_list in id2answer.items():
    ans2scores = collections.defaultdict(list)
    for pred in pred_list:
        ans2scores[pred["pred_str"]].append((pred["rank_score"], pred["span_score"]))
    ans_tuples = []
    for ans in ans2scores.keys():
        rank_score = np.mean([_[0] for _ in ans2scores[ans]])
        span_score = np.mean([_[1] for _ in ans2scores[ans]])
        combined_score = np.mean([sigmoid(_[0]) * _[1] for _ in ans2scores[ans]])
        ans_tuples.append((ans, rank_score, span_score, combined_score))
    id2agg[qid] = ans_tuples

for lambda_ in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    ems, f1s = [], []

    for qid, ans_res in id2agg.items():
        ans_tuples = ans_res
        # ans_tuples.sort(key=lambda x: lambda_ * x[1] + (1 - lambda_) * x[2], reverse=True)
        ans_tuples.sort(key=lambda x: x[3], reverse=True)
        top_pred = ans_tuples[0][0]
        ems.append(exact_match_score(top_pred, id2gold[qid][0]))
        f1, prec, recall = f1_score(top_pred, id2gold[qid][0])
        f1s.append(f1)

    print(f".......Using combination factor {lambda_}......")
    print(f'answer em: {np.mean(ems)}, count: {len(ems)}')
    print(f'answer f1: {np.mean(f1s)}, count: {len(f1s)}')
