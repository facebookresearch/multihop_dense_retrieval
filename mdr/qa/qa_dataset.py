# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import collections
import json
import random

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from .basic_tokenizer import SimpleTokenizer
from .utils import (find_ans_span_with_char_offsets, match_answer_span, para_has_answer, _is_whitespace)

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


def prepare(item, tokenizer, special_toks=["[SEP]", "[unused1]", "[unused2]"]):
    """
    tokenize the passages chains, add sentence start markers for SP sentence identification
    """
    def _process_p(para):
        """
        handle each para
        """
        title, sents = para["title"].strip(), para["sents"]
        # return "[unused1] " + title + " [unused1] " + text # mark title
        # return title + " " + text
        pre_sents = []
        for idx, sent in enumerate(sents):
            pre_sents.append("[unused1] " + sent.strip())
        return title + " " + " ".join(pre_sents)
        # return " ".join(pre_sents)
    # mark passage boundary
    contexts = []
    for para in item["passages"]:
        contexts.append(_process_p(para))
    context = " [SEP] ".join(contexts)

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    context = "yes no [SEP] " + context

    for c in context:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    sent_starts = []
    orig_to_tok_index = []
    tok_to_orig_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))

        if token in special_toks:
            if token == "[unused1]":
                sent_starts.append(len(all_doc_tokens))

            sub_tokens = [token]
        else:
            sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    item["context_processed"] = {
        "doc_tokens": doc_tokens,
        "char_to_word_offset": char_to_word_offset,
        "orig_to_tok_index": orig_to_tok_index,
        "tok_to_orig_index": tok_to_orig_index,
        "all_doc_tokens": all_doc_tokens,
        "context": context,
        "sent_starts": sent_starts
    }

    return item

class QAEvalDataset(Dataset):

    def __init__(self,
        tokenizer,
        retrievel_results,
        max_seq_len,
        max_q_len,
        ):

        retriever_outputs = retrievel_results
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.data = []

        for item in retriever_outputs:
            if item["question"].endswith("?"):
                item["question"] = item["question"][:-1]

            # for validation, add target predictions
            sp_titles = None
            gold_answer = item.get("answer", [])
            sp_gold = []

            for chain in item["candidate_chains"]:
                chain_titles = [_["title"] for _ in chain]

                if sp_titles:
                    label = int(set(chain_titles) == sp_titles)
                else:
                    label = -1
                self.data.append({
                    "question": item["question"],
                    "passages": chain,
                    "label": label,
                    "qid": item["_id"],
                    "gold_answer": gold_answer,
                    "sp_gold": sp_gold
                })

        print(f"Total instances size {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = prepare(self.data[index], self.tokenizer) 
        context_ann = item["context_processed"]
        q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
        para_offset = len(q_toks) + 2 # cls and seq
        item["wp_tokens"] = context_ann["all_doc_tokens"]
        assert item["wp_tokens"][0] == "yes" and item["wp_tokens"][1] == "no"
        item["para_offset"] = para_offset
        max_toks_for_doc = self.max_seq_len - para_offset - 1
        if len(item["wp_tokens"]) > max_toks_for_doc:
            item["wp_tokens"] = item["wp_tokens"][:max_toks_for_doc]
        item["encodings"] = self.tokenizer.encode_plus(q_toks, text_pair=item["wp_tokens"], max_length=self.max_seq_len, return_tensors="pt", is_pretokenized=True)

        item["paragraph_mask"] = torch.zeros(item["encodings"]["input_ids"].size()).view(-1)
        item["paragraph_mask"][para_offset:-1] = 1
        
        item["doc_tokens"] = context_ann["doc_tokens"]
        item["tok_to_orig_index"] = context_ann["tok_to_orig_index"]

        # filter sentence offsets exceeding max sequence length
        sent_labels, sent_offsets = [], []
        for idx, s in enumerate(item["context_processed"]["sent_starts"]):
            if s >= len(item["wp_tokens"]):
                break
            if "sp_sent_labels" in item:
                sent_labels.append(item["sp_sent_labels"][idx])
            sent_offsets.append(s + para_offset)
            assert item["encodings"]["input_ids"].view(-1)[s+para_offset] == self.tokenizer.convert_tokens_to_ids("[unused1]")

        # supporting fact label
        item["sent_offsets"] = sent_offsets
        item["sent_offsets"] = torch.LongTensor(item["sent_offsets"])
        item["label"] = torch.LongTensor([item["label"]])
        return item

class QADataset(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_seq_len,
        max_q_len,
        train=False,
        no_sent_label=False
        ):

        retriever_outputs = [json.loads(l) for l in tqdm(open(data_path).readlines())]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.train = train
        self.no_sent_label = no_sent_label
        self.simple_tok = SimpleTokenizer()
        self.data = []

        if train:
            self.qid2gold = collections.defaultdict(list) # idx 
            self.qid2neg = collections.defaultdict(list)
            for item in retriever_outputs:
                if item["question"].endswith("?"):
                    item["question"] = item["question"][:-1]

                sp_sent_labels = []
                sp_gold = []
                if not self.no_sent_label:
                    for sp in item["sp"]:
                        for _ in sp["sp_sent_ids"]:
                            sp_gold.append([sp["title"], _])
                        for idx in range(len(sp["sents"])):
                            sp_sent_labels.append(int(idx in sp["sp_sent_ids"]))

                question_type = item["type"]
                self.data.append({
                    "question": item["question"],
                    "passages": item["sp"], 
                    "label": 1,
                    "qid": item["_id"],
                    "gold_answer": item["answer"],
                    "sp_sent_labels": sp_sent_labels,
                    "ans_covered": 1, # includes partial chains.
                    "sp_gold": sp_gold
                })
                self.qid2gold[item["_id"]].append(len(self.data) - 1)

                sp_titles = set([_["title"] for _ in item["sp"]])
                if question_type == "bridge":
                    ans_titles = set([p["title"] for p in item["sp"] if para_has_answer(item["answer"], "".join(p["sents"]), self.simple_tok)])
                else:
                    ans_titles = set()
                # top ranked negative chains
                ds_count = 0 # track how many distant supervised chain to use
                ds_limit = 5
                for chain in item["candidate_chains"]:
                    chain_titles = [_["title"] for _ in chain]
                    if set(chain_titles) == sp_titles:
                        continue
                    if question_type == "bridge":
                        answer_covered = int(len(set(chain_titles) & ans_titles) > 0)
                        ds_count += answer_covered
                    else:
                        answer_covered = 0
                    self.data.append({
                        "question": item["question"],
                        "passages": chain,
                        "label": 0,
                        "qid": item["_id"],
                        "gold_answer": item["answer"],
                        "ans_covered": answer_covered,
                        "sp_gold": sp_gold
                    })
                    self.qid2neg[item["_id"]].append(len(self.data) - 1)
        else:
            for item in retriever_outputs:
                if item["question"].endswith("?"):
                    item["question"] = item["question"][:-1]

                # for validation, add target predictions
                sp_titles = set([_["title"] for _ in item["sp"]]) if "sp" in item else None
                gold_answer = item.get("answer", [])
                sp_gold = []
                if "sp" in item:
                    for sp in item["sp"]:
                        for _ in sp["sp_sent_ids"]:
                            sp_gold.append([sp["title"], _])

                chain_seen = set()
                for chain in item["candidate_chains"]:
                    chain_titles = [_["title"] for _ in chain]

                    # title_set = frozenset(chain_titles)
                    # if len(title_set) == 0 or title_set in chain_seen:
                    #     continue
                    # chain_seen.add(title_set)

                    if sp_titles:
                        label = int(set(chain_titles) == sp_titles)
                    else:
                        label = -1
                    self.data.append({
                        "question": item["question"],
                        "passages": chain,
                        "label": label,
                        "qid": item["_id"],
                        "gold_answer": gold_answer,
                        "sp_gold": sp_gold
                    })

        print(f"Data size {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = prepare(self.data[index], self.tokenizer) 
        context_ann = item["context_processed"]
        q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
        para_offset = len(q_toks) + 2 # cls and seq
        item["wp_tokens"] = context_ann["all_doc_tokens"]
        assert item["wp_tokens"][0] == "yes" and item["wp_tokens"][1] == "no"
        item["para_offset"] = para_offset
        max_toks_for_doc = self.max_seq_len - para_offset - 1
        if len(item["wp_tokens"]) > max_toks_for_doc:
            item["wp_tokens"] = item["wp_tokens"][:max_toks_for_doc]
        item["encodings"] = self.tokenizer.encode_plus(q_toks, text_pair=item["wp_tokens"], max_length=self.max_seq_len, return_tensors="pt", is_pretokenized=True)

        item["paragraph_mask"] = torch.zeros(item["encodings"]["input_ids"].size()).view(-1)
        item["paragraph_mask"][para_offset:-1] = 1
        
        if self.train:
            # if item["label"] == 1:
            if item["ans_covered"]:
                if item["gold_answer"][0] == "yes":
                    # ans_type = 0
                    starts, ends= [para_offset], [para_offset]
                elif item["gold_answer"][0] == "no":
                    # ans_type = 1
                    starts, ends= [para_offset + 1], [para_offset + 1]
                else:
                    # ans_type = 2
                    matched_spans = match_answer_span(context_ann["context"], item["gold_answer"], self.simple_tok)
                    ans_starts, ans_ends= [], []
                    for span in matched_spans:
                        char_starts = [i for i in range(len(context_ann["context"])) if context_ann["context"].startswith(span, i)]
                        if len(char_starts) > 0:
                            char_ends = [start + len(span) - 1 for start in char_starts]
                            answer = {"text": span, "char_spans": list(zip(char_starts, char_ends))}
                            ans_spans = find_ans_span_with_char_offsets(
                            answer, context_ann["char_to_word_offset"], context_ann["doc_tokens"], context_ann["all_doc_tokens"], context_ann["orig_to_tok_index"], self.tokenizer)
                            for s, e in ans_spans:
                                ans_starts.append(s)
                                ans_ends.append(e)
                    starts, ends = [], []
                    for s, e in zip(ans_starts, ans_ends):
                        if s >= len(item["wp_tokens"]):
                            continue
                        else:
                            s = min(s, len(item["wp_tokens"]) - 1) + para_offset
                            e = min(e, len(item["wp_tokens"]) - 1) + para_offset
                            starts.append(s)
                            ends.append(e)
                    if len(starts) == 0:
                        starts, ends = [-1], [-1]         
            else:
                starts, ends= [-1], [-1]
                # ans_type = -1
                        
            item["starts"] = torch.LongTensor(starts)
            item["ends"] = torch.LongTensor(ends)
            # item["ans_type"] = torch.LongTensor([ans_type])

            if item["label"]:
                assert len(item["sp_sent_labels"]) == len(item["context_processed"]["sent_starts"])
        else:
            #     # for answer extraction
            item["doc_tokens"] = context_ann["doc_tokens"]
            item["tok_to_orig_index"] = context_ann["tok_to_orig_index"]

        # filter sentence offsets exceeding max sequence length
        sent_labels, sent_offsets = [], []
        for idx, s in enumerate(item["context_processed"]["sent_starts"]):
            if s >= len(item["wp_tokens"]):
                break
            if "sp_sent_labels" in item:
                sent_labels.append(item["sp_sent_labels"][idx])
            sent_offsets.append(s + para_offset)
            assert item["encodings"]["input_ids"].view(-1)[s+para_offset] == self.tokenizer.convert_tokens_to_ids("[unused1]")

        # supporting fact label
        item["sent_offsets"] = sent_offsets
        item["sent_offsets"] = torch.LongTensor(item["sent_offsets"])
        if self.train:
            item["sent_labels"] = sent_labels if len(sent_labels) != 0 else [0] * len(sent_offsets)
            item["sent_labels"] = torch.LongTensor(item["sent_labels"])
            item["ans_covered"] = torch.LongTensor([item["ans_covered"]])

        item["label"] = torch.LongTensor([item["label"]])
        return item

class MhopSampler(Sampler):
    """
    Shuffle QA pairs not context, make sure data within the batch are from the same QA pair
    """

    def __init__(self, data_source, num_neg=9, n_gpu=8):
        # for each QA pair, sample negative paragraphs
        self.qid2gold = data_source.qid2gold
        self.qid2neg = data_source.qid2neg
        self.neg_num = num_neg
        self.n_gpu = n_gpu
        self.all_qids = list(self.qid2gold.keys())
        assert len(self.qid2gold) == len(self.qid2neg)

        self.q_num_per_epoch = len(self.qid2gold) - len(self.qid2gold) % self.n_gpu
        self._num_samples = self.q_num_per_epoch * (self.neg_num + 1)

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        sample_indice = []
        random.shuffle(self.all_qids)
        
        # when use shared-normalization, passages for each question should be on the same GPU
        qids_to_use = self.all_qids[:self.q_num_per_epoch]
        for qid in qids_to_use:
            neg_samples = self.qid2neg[qid]
            random.shuffle(neg_samples)
            sample_indice += self.qid2gold[qid]
            sample_indice += neg_samples[:self.neg_num]
        return iter(sample_indice)

def qa_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([s["encodings"]['input_ids'] for s in samples], pad_id),
        'attention_mask': collate_tokens([s["encodings"]['attention_mask'] for s in samples], 0),
        'paragraph_mask': collate_tokens([s['paragraph_mask'] for s in samples], 0),
        'label': collate_tokens([s["label"] for s in samples], -1),
        "sent_offsets": collate_tokens([s["sent_offsets"] for s in samples], 0),
        }

    # training labels
    if "starts" in samples[0]:
        batch["starts"] = collate_tokens([s['starts'] for s in samples], -1)
        batch["ends"] = collate_tokens([s['ends'] for s in samples], -1)
        # batch["ans_types"] = collate_tokens([s['ans_type'] for s in samples], -1)
        batch["sent_labels"] = collate_tokens([s['sent_labels'] for s in samples], 0)
        batch["ans_covered"] = collate_tokens([s['ans_covered'] for s in samples], 0)

    # roberta does not use token_type_ids
    if "token_type_ids" in samples[0]["encodings"]:
        batch["token_type_ids"] = collate_tokens([s["encodings"]['token_type_ids']for s in samples], 0)

    batched = {
        "qids": [s["qid"] for s in samples],
        "passages": [s["passages"] for s in samples],
        "gold_answer": [s["gold_answer"] for s in samples],
        "sp_gold": [s["sp_gold"] for s in samples],
        "para_offsets": [s["para_offset"] for s in samples],
        "net_inputs": batch,
    }

    # for answer extraction
    if "doc_tokens" in samples[0]:
        batched["doc_tokens"] = [s["doc_tokens"] for s in samples]
        batched["tok_to_orig_index"] = [s["tok_to_orig_index"] for s in samples]
        batched["wp_tokens"] = [s["wp_tokens"] for s in samples]

    return batched
