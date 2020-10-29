import numpy as np
import json
import faiss
import argparse
import time

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import defaultdict

from utils.utils import DocDB, normalize,load_saved, move_to_cuda, para_has_answer
from utils.basic_tokenizer import SimpleTokenizer
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig
from models.mhop_retriever import RNNRetriever, MhopRetriever

import csv
import logging
import collections
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def get_title(s):
    return s[:s.find("_")]

if __name__ == "__main__":
    final_titles = json.load(open("/private/home/xwhan/code/learning_to_retrieve_reasoning_paths/complexwebq_reader_titles.json"))
    retrieval_outputs = json.load(open("/private/home/xwhan/code/learning_to_retrieve_reasoning_paths/complexwebq_retrieval.json"))
    id2context = {item["q_id"]:item["context"] for item in retrieval_outputs}

    ground = [json.loads(l) for l in open("/private/home/xwhan/data/ComplexWebQ/complexwebq_dev_qas.txt")]
    id2gold = {item["id"]: item["answer"] for item in ground}
    simple_tokenizer = SimpleTokenizer()

    ans_recall = []
    for qid, titles in final_titles.items():
        t2p = id2context[qid]
        concat_p = " ".join([get_title(t) + " " + t2p[t] for t in titles])
        ans_recall.append(para_has_answer(id2gold[qid], concat_p, simple_tokenizer))

    print(len(ans_recall))
    print(np.mean(ans_recall))