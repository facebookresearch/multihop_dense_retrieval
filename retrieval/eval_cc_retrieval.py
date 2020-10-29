"""
python eval_cc_retrieval.py /private/home/xwhan/data/hotpot/hotpot_qas_val.json /checkpoint/xwhan/cc_head_lm250_psg100/index logs/08-08-2020/roberta_v0_fixed-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt --batch-size 200 --beam-size-1 50 --beam-size-2 50 --topk 50 --shared-encoder --model-name roberta-base

top10 answer coverage 28.0%
"""
import argparse
import collections
import json
import logging

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.mhop_retriever import MhopRetriever, RobertaRetriever
from utils.basic_tokenizer import SimpleTokenizer
from utils.utils import (load_saved, move_to_cuda, para_has_answer)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None)
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=2, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-len', type=int, default=70)
    parser.add_argument('--max-c-len', type=int, default=300)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size-1', type=int, default=5)
    parser.add_argument('--beam-size-2', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument('--saved-index', type=str, default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument('--save-index', action="store_true")
    parser.add_argument('--nprob', type=int, default=20)
    args = parser.parse_args()
    
    logger.info("Loading data...")
    ds_items = [json.loads(_) for _ in open(args.raw_data).readlines()]

    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "roberta" in args.model_name:
        model = RobertaRetriever(bert_config, args)
    else:
        model = MhopRetriever(bert_config, args)
    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    logger.info("Building index...")
    if args.saved_index != "":
        index = faiss.read_index(args.saved_index)
    else:    
        # quantizer = faiss.IndexFlatIP(768)
        # nlist = 500
        # m = 64 # number of bytes per vector
        # bit_per_subvector = 8
        # index = faiss.IndexIVFPQ(quantizer, 768, nlist, m, bit_per_subvector)
        # for seg_id in tqdm(range(10)):
        #     logger.info(f"Reading vectors cc_seg{seg_id}.npy...")
        #     xb = np.load(args.indexpath + f"/cc_seg{seg_id}.npy").astype('float32')
        #     if seg_id == 0:
        #         index.train(xb)
        #     index.add(xb)
        xb = np.load(args.indexpath + "/cc_head_lm250_psg100.npy").astype('float32')
        index = faiss.IndexFlatIP(768)
        index.add(xb)

        # if args.save_index:
        #     faiss.write_index(index, f"{args.indexpath}/faiss_cc_centroid{nlist}_PQ{m}_bit{bit_per_subvector}_index")
    
    # index.nprobe = args.nprob

    logger.info("Loading corpus...")
    id2doc = json.load(open(args.indexpath + f"/cc_head_lm250_psg100_id2doc.json"))
    # id2doc = {}
    # offset = 0
    # for seg_id in tqdm(range(10)):
    #     seg_id2doc = json.load(open(args.indexpath + f"/cc_seg{seg_id}_id2doc.json"))
    #     for k, v in seg_id2doc.items():
    #         id2doc[str(int(k) + offset)] = v
    #     offset += len(seg_id2doc)  
    title2doc = {item[0]:item[1] for item in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")
    

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info("Encoding questions and searching")
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    metrics = []
    retrieval_outputs = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ann = ds_items[b_start:b_start + args.batch_size]
            bsize = len(batch_q)
            batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy() 
            D, I = index.search(q_embeds_numpy, args.beam_size_1)

            # 2hop search
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):
                    doc = id2doc[str(doc_id)][1]
                    if "roberta" in  args.model_name and doc.strip() == "":
                        doc = id2doc[str(doc_id)][0]
                        D[b_idx][_] = float("-inf")
                    query_pairs.append((batch_q[b_idx], doc))

            batch_q_sp_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")

            batch_q_sp_encodes = move_to_cuda(dict(batch_q_sp_encodes))
            q_sp_embeds = model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], batch_q_sp_encodes.get("token_type_ids", None))
            q_sp_embeds = q_sp_embeds.contiguous().cpu().numpy()
            
            # search_start = time.time()
            D_, I_ = index.search(q_sp_embeds, args.beam_size_2)
            # logger.info(f"MIPS searching: {time.time() - search_start}")
            D_ = D_.reshape(bsize, args.beam_size_1, args.beam_size_2)
            I_ = I_.reshape(bsize, args.beam_size_1, args.beam_size_2)

            # aggregate path scores
            path_scores = np.expand_dims(D, axis=2) + D_
            # path_scores = D_
            # eval
            for idx in range(bsize):
                search_scores = path_scores[idx]
                ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], (args.beam_size_1, args.beam_size_2))).transpose()
                retrieved_titles = []
                hop1_titles = []
                paths, path_titles = [], []
                paths_both_are_intro = []
                for _ in range(args.topk):
                    path_ids = ranked_pairs[_]
                    hop_1_id = I[idx, path_ids[0]]
                    hop_2_id = I_[idx, path_ids[0], path_ids[1]]
                    retrieved_titles.append(id2doc[str(hop_1_id)][0])
                    retrieved_titles.append(id2doc[str(hop_2_id)][0])

                    paths.append([str(hop_1_id), str(hop_2_id)])
                    path_titles.append([id2doc[str(hop_1_id)][0], id2doc[str(hop_2_id)][0]])
                    paths_both_are_intro.append(id2doc[str(hop_1_id)][2] and id2doc[str(hop_2_id)][2])
                    hop1_titles.append(id2doc[str(hop_1_id)][0])
                
                gold_answers = batch_ann[idx]["answer"]
                concat_p = ""
                for p in paths:
                    concat_p += " ".join([id2doc[doc_id][0] + " " + id2doc[doc_id][1] for doc_id in p])
                metrics.append({
                    "question": batch_ann[idx]["question"],
                    "ans_recall": int(para_has_answer(gold_answers, concat_p, simple_tokenizer)),
                    "type": batch_ann[idx].get("type", "single")
                })
                
                sp = batch_ann[idx]["sp"]
                if args.save_path != "":
                    candidaite_chains = []
                    for path in path_titles:
                        candidaite_chains.append([(path[0], title2doc[path[0]]), (path[1], title2doc[path[1]])])
                    retrieval_outputs.append({
                        "_id": batch_ann[idx]["_id"],
                        "question": batch_ann[idx]["question"],
                        "type": batch_ann[idx]["type"],
                        "sp": [(sp[0], title2doc[sp[0]]), (sp[1], title2doc[sp[1]])],
                        "answer": batch_ann[idx]["answer"],
                        "candidate_chains": candidaite_chains
                    })


    if args.save_path != "":
        with open(f"/private/home/xwhan/data/hotpot/{args.save_path}", "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")

    logger.info(f"Evaluating {len(metrics)} samples...")
    type2items = collections.defaultdict(list)
    for item in metrics:
        type2items[item["type"]].append(item)
    logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
    for t in type2items.keys():
        logger.info(f"{t} Questions num: {len(type2items[t])}")
        logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
