"""
python nq_analysis.py index/psg100_nqmhop.npy index/psgs_w100_id2doc.json logs/08-24-2020/nq_mhop_2_marker_shared_dpr_neg-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt/checkpoint_best.pt --batch-size 100 --shared-encoder --model-name bert-base-uncased --topk 10 --beam-size 10

python nq_analysis.py index/psg100_mhop_add_marker.npy index/psgs_w100_id2doc.json logs/08-22-2020/nq_mhop_add_marker_type2-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 20 --shared-encoder --model-name bert-base-uncased --topk 50 --beam-size 50


python nq_mhop_retrieval.py index/psg100_nqmhop.npy index/psgs_w100_id2doc.json logs/08-24-2020/nq_mhop_2_marker_shared_dpr_neg-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt --batch-size 200 --shared-encoder --model-name bert-base-uncased --topk 10 --beam-size 20
"""

import json
import logging
import argparse

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.mhop_retriever import MhopRetriever, RobertaRetriever
from models.retriever import RobertaRetrieverSingle
from models.unified_retriever import UnifiedRetriever, BertNQRetriever
from utils.basic_tokenizer import SimpleTokenizer
from utils.utils import (load_saved, para_has_answer, move_to_cuda)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def add_marker_q(tokenizer, q, marker):
    q_toks = tokenizer.tokenize(q)
    return [marker] + q_toks

def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument('indexpath', type=str, default=None)
    parser.add_argument('corpus_dict', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)
    parser.add_argument('--topk', type=int, default=20, help="topk paths")
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--max-c-len', type=int, default=300)
    parser.add_argument('--max-q-len', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
    args = parser.parse_args()

    logger.info("Loading data...")
    # all_data = open(f"/private/home/xwhan/data/nq-dpr/results/nq-test-add-marker-type2-top{args.topk}-errors.txt").readlines()

    all_data = open(f"/private/home/xwhan/data/nq-dpr/nq-test-qas.txt").readlines()
    ds_items = [json.loads(_.strip()) for _ in all_data]

    logger.info("Building index...")
    d = 768
    xb = np.load(args.indexpath).astype('float32')
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    title2doc = {item[0]:item[1] for item in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")

    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # model = UnifiedRetriever(bert_config, args)
    # model = RobertaRetrieverSingle(bert_config, args)
    model = BertNQRetriever(bert_config, args)

    model = load_saved(model, args.model_path, exact=False)
    simple_tokenizer = SimpleTokenizer()

    cuda = torch.device('cuda')
    model.to(cuda)
    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    questions = [_["question"] for _ in ds_items]
    answers = [_["answer"] for _ in ds_items]
    # retrieved_results = [_["topk"][:args.topk] for _ in ds_items]
    # retrieved_docids = [_["topk_ids"][:args.topk] for _ in ds_items]

    corrected_cases = []
    ans_recalls = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ans = answers[b_start:b_start + args.batch_size]
            bsize = len(batch_q)

            # retrieve the 1st hop
            batch_q_toks = [add_marker_q(tokenizer, q, '[unused0]') for q in batch_q]
            batch_q_encodes_1st = tokenizer.batch_encode_plus(batch_q_toks, max_length=args.max_q_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_encodes_1st = move_to_cuda(dict(batch_q_encodes_1st))
            q_embeds_1st = model.encode_q(batch_q_encodes_1st["input_ids"], batch_q_encodes_1st["attention_mask"], batch_q_encodes_1st.get("token_type_ids", None))
            q_embeds_numpy_1st = q_embeds_1st.cpu().contiguous().numpy()
            D_1, I_1 = index.search(q_embeds_numpy_1st, args.topk)

            docids_1hop, scores_1hop, topk_1hop, docids_scores_1hop = [], [], [], []
            for idx in range(bsize):
                docids_1hop.append([str(_) for _ in I_1[idx]])
                scores_1hop.append([_ for _ in D_1[idx]])
                docids_scores_1hop.append(list(zip(docids_1hop[-1], scores_1hop[-1])))
                topk_1hop.append([id2doc[_] for _ in docids_1hop[-1]])
                # import pdb; pdb.set_trace()

        
            # topk_1hop = retrieved_results[b_start:b_start + args.batch_size]
            # docids_1hop = retrieved_docids[b_start:b_start + args.batch_size]

            query_pairs = []
            for q, topk in zip(batch_q, topk_1hop):
                for doc in topk:
                    query_pairs.append((q, doc[0] + " [SEP] " + doc[1].strip()))

            # batch_q_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            
            # add marker to q
            tokenized_query_pairs = []
            for q, p in query_pairs:
                tokenized_query_pairs.append((add_marker_q(tokenizer, q, '[unused1]'), tokenizer.tokenize(p)))
    
            batch_q_encodes = tokenizer.batch_encode_plus(tokenized_query_pairs, max_length=args.max_q_sp_len, pad_to_max_length=True, return_tensors="pt")
            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            D, I = index.search(q_embeds_numpy, args.beam_size)

            # add scores from 1st hop
            D = D.reshape(bsize, args.topk, args.beam_size) + np.expand_dims(D_1, axis=2)
            
            D = D.reshape(bsize, args.topk*args.beam_size)
            I = I.reshape(bsize, args.topk*args.beam_size)

            for idx in range(bsize):
                existing_doc_ids = set(docids_1hop[idx])
                assert len(existing_doc_ids) == args.topk

                # trace_back = []
                # for doc_id in docids_1hop[idx]:
                #     trace_back += [doc_id] * args.beam_size 
                # docs_1hop = [{"title": id2doc[_][0], "text": id2doc[_][1]} for _ in existing_doc_ids]

                ids_and_scores_agg = docids_scores_1hop[idx]
                for doc_id, score in list(zip([str(_) for _ in I[idx]], D[idx])):
                    if doc_id not in existing_doc_ids:
                        existing_doc_ids.add(doc_id)
                        ids_and_scores_agg.append((doc_id, score))

                ids_and_scores_agg.sort(key=lambda x:x[1], reverse=True)

                final_topk, ans_recall = [], []
                for doc_id, score in ids_and_scores_agg[:args.topk]:
                    final_topk.append({"title": id2doc[str(doc_id)][0], "text": id2doc[str(doc_id)][1]})
                    ans_recall.append(int(para_has_answer(batch_ans[idx], final_topk[-1]["title"] + " " + final_topk[-1]["text"], simple_tokenizer)))

                    # if str(doc_id) not in existing_doc_ids:
                    #     existing_doc_ids.add(str(doc_id))
                    #     docs_2hop.append({"title": id2doc[str(doc_id)][0], "text": id2doc[str(doc_id)][1]})

                    #     ans_recall.append(int(para_has_answer(batch_ans[idx], docs_2hop[-1]["title"] + " " + docs_2hop[-1]["text"], simple_tokenizer)))
                    #     count += 1
                    #     if count == args.topk:
                    #         break

                # scores = D[idx]
                # ranked = np.argsort(scores)[::-1]

                # for _ in ranked[:args.topk]:
                #     hop2_id = I[idx, _]
                #     docs_2hop.append({"title": id2doc[str(hop2_id)][0], "text": id2doc[str(hop2_id)][1]})
                #     ans_recall.append(int(para_has_answer(batch_ans[idx], docs_2hop[-1]["title"] + " " + docs_2hop[-1]["text"], simple_tokenizer)))

                if np.sum(ans_recall) > 0:
                    corrected_cases.append(
                        {
                            "question": batch_q[idx],
                            "answer": batch_ans[idx],
                        }
                    )

                ans_recalls.append(int(np.sum(ans_recall) > 0))

    with open("/private/home/xwhan/data/nq-dpr/analysis/nq_recovered_cases_top50_errors.json", "w") as g:
        for _ in corrected_cases:
            g.write(json.dumps(_) + "\n")

    assert len(ans_recalls) == len(questions)
    print(f"Top{args.topk} 2hop, answer recall {np.mean(ans_recalls)}, count {len(ans_recalls)}")

if __name__ == '__main__':
    analyze()
    # get_errors()