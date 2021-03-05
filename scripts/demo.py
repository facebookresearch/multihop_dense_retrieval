# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import streamlit as st
import torch
import os
import numpy as np
from apex import amp
import faiss
import json
import argparse
from functools import partial
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader

from mdr.retrieval.models.mhop_retriever import RobertaRetriever
from mdr.retrieval.utils.basic_tokenizer import SimpleTokenizer
from mdr.retrieval.utils.utils import load_saved, move_to_cuda

from mdr.qa.qa_model import QAModel
from mdr.qa.qa_dataset import qa_collate, QAEvalDataset
from train_qa import eval_final

@st.cache(allow_output_mutation=True)
def init_retrieval(args):
    print("Initializing retrieval module...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    retriever = RobertaRetriever(bert_config, args)
    retriever = load_saved(retriever, args.model_path, exact=False)
    cuda = torch.device('cuda')
    retriever.to(cuda)
    retriever = amp.initialize(retriever, opt_level='O1')
    retriever.eval()

    print("Loading index...")
    index = faiss.IndexFlatIP(768)
    xb = np.load(args.indexpath).astype('float32')
    index.add(xb)
    if args.index_gpu != -1:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, args.index_gpu, index)

    print("Loading documents...")
    id2doc = json.load(open(args.corpus_dict))

    print("Index ready...")
    return retriever, index, id2doc, tokenizer

@st.cache(allow_output_mutation=True)
def init_reader(args):
    qa_config = AutoConfig.from_pretrained(
        'google/electra-large-discriminator')
    qa_tokenizer = AutoTokenizer.from_pretrained(
        'google/electra-large-discriminator')
    retriever_name = args.model_name
    args.model_name = "google/electra-large-discriminator"
    reader = QAModel(qa_config, args)
    reader = load_saved(reader, args.reader_path, False)    
    cuda = torch.device('cuda')
    reader.to(cuda)
    reader = amp.initialize(reader, opt_level='O1')
    reader.eval()
    args.model_name = retriever_name
    return reader, qa_tokenizer


st.markdown(
    "# Multi-hop Open-domain QA with [MDR](https://github.com/facebookresearch/multihop_dense_retrieval)")

parser = argparse.ArgumentParser()
parser.add_argument('--indexpath', type=str,
                    default='data/hotpot_index/wiki_index.npy')
parser.add_argument('--corpus_dict', type=str,
                    default='data/hotpot_index/wiki_id2doc.json')
parser.add_argument('--model_path', type=str, default='models/q_encoder.pt')
parser.add_argument('--topk', type=int, default=20, help="topk paths")
parser.add_argument('--max-q-len', type=int, default=70)
parser.add_argument('--max-c-len', type=int, default=300)
parser.add_argument('--max-q-sp-len', type=int, default=350)
parser.add_argument('--model-name', type=str, default='roberta-base')
parser.add_argument('--reader_path', type=str, default="models/qa_electra.pt")

parser.add_argument("--sp-pred", action="store_true",
                    help="whether to predict sentence sp")
parser.add_argument("--sp-weight", default=0, type=float,
                    help="weight of the sp loss")
parser.add_argument("--max-ans-len", default=30, type=int)
parser.add_argument("--save-prediction", default="", type=str)
parser.add_argument("--index-gpu", default=-1, type=int)

args = parser.parse_args()

reader, qa_tokenizer = init_reader(args)

retriever, index, id2doc, retriever_tokenizer = init_retrieval(args)

st.markdown("*Trick: Due to the case sensitive tokenization we used during training, try to use capitalized entity names in your question, e.g., type United States instead of united states.*")

query = st.text_input('Enter your question')
if query:

    query = query[:-1] if query.endswith("?") else query
    with torch.no_grad():

        print("Retrieving")
        q_encodes = retriever_tokenizer.batch_encode_plus(
            [query], max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
        q_encodes = move_to_cuda(dict(q_encodes))
        q_embeds = retriever.encode_q(
            q_encodes["input_ids"], q_encodes["attention_mask"], q_encodes.get("token_type_ids", None)).cpu().numpy()
        scores_1, docid_1 = index.search(q_embeds, args.topk)
        query_pairs = [] # for 2nd hop
        for _, doc_id in enumerate(docid_1[0]):
            doc = id2doc[str(doc_id)]["text"]
            if doc.strip() == "":
                # roberta tokenizer does not accept empty string as segment B
                doc = id2doc[str(doc_id)]["title"]
                scores_1[b_idx][_] = float("-inf")
            query_pairs.append((query, doc))
        
        q_sp_encodes = retriever_tokenizer.batch_encode_plus(
            query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
        q_sp_encodes = move_to_cuda(dict(q_sp_encodes))
        q_sp_embeds = retriever.encode_q(
            q_sp_encodes["input_ids"], q_sp_encodes["attention_mask"],q_sp_encodes.get("token_type_ids", None)).cpu().numpy()
        scores_2, docid_2 = index.search(q_sp_embeds, args.topk)

        scores_2 = scores_2.reshape(1, args.topk, args.topk)
        docid_2 = docid_2.reshape(1, args.topk, args.topk)
        path_scores = np.expand_dims(scores_1, axis=2) + scores_2
        search_scores = path_scores[0]
        ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], (args.topk, args.topk))).transpose()
        chains = []
        topk_docs = {}
        for _ in range(args.topk):
            path_ids = ranked_pairs[_]
            doc1_id = str(docid_1[0, path_ids[0]])
            doc2_id = str(docid_2[0, path_ids[0], path_ids[1]])
            chains.append([id2doc[doc1_id], id2doc[doc2_id]])
            topk_docs[id2doc[doc1_id]['title']] = id2doc[doc1_id]['text']
            topk_docs[id2doc[doc2_id]['title']] = id2doc[doc2_id]['text']


        reader_input = [{
            "_id": 0,
            "question": query,
            "candidate_chains": chains
        }]

        print(f"Reading {len(chains)} chains...")
        collate_fc = partial(qa_collate, pad_id=qa_tokenizer.pad_token_id)
        qa_eval_dataset = QAEvalDataset(
            qa_tokenizer, reader_input, max_seq_len=512, max_q_len=64)
        qa_eval_dataloader = DataLoader(
            qa_eval_dataset, batch_size=args.topk, collate_fn=collate_fc, pin_memory=True, num_workers=0)
        qa_results = eval_final(args, reader, qa_eval_dataloader, gpu=True)

        answer_pred = qa_results['answer'][0]
        sp_pred = qa_results['sp'][0]
        titles_pred = qa_results['titles'][0]


        st.markdown(f'**Answer**: {answer_pred}')
        st.markdown(f'**Supporting passages**:')
        st.markdown(f'> **{titles_pred[0]}**: {topk_docs[titles_pred[0]].replace(answer_pred, "**" + answer_pred + "**")}')
        st.markdown(
            f'> **{titles_pred[1]}**: {topk_docs[titles_pred[1]].replace(answer_pred, "**" + answer_pred + "**")}')

        # st.write(qa_results)
