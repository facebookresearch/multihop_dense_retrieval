# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import collections
import json
import logging
import os
import random
from datetime import date
from functools import partial

import numpy as np
from numpy.core.defchararray import encode
import torch
from torch import sparse_coo_tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from mdr.qa.config import train_args
from mdr.qa.qa_dataset import QADataset, qa_collate, MhopSampler
from mdr.qa.qa_model import QAModel
from mdr.qa.utils import AverageMeter, move_to_cuda, get_final_text

from mdr.qa.hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp

def load_saved(model, path):
    state_dict = torch.load(path)
    def filter(x): return x[7:] if x.startswith('module.') else x
    state_dict = {filter(k): v for (k, v) in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def main():
    args = train_args()
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-neg{args.neg_num}-sn{args.shared_norm}-adam{args.use_adam}-warm{args.warmup_ratio}-sp{args.sp_weight}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)
    tb_logger = SummaryWriter(os.path.join(args.output_dir.replace("logs","tflogs")))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    if args.shared_norm:
        # chains of each question are on the same gpu
        assert (args.train_batch_size // n_gpu) == args.neg_num + 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # define model
    if args.model_name == "spanbert":
        bert_config = AutoConfig.from_pretrained("/private/home/span-bert")
        tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
    else:
        bert_config = AutoConfig.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QAModel(bert_config, args)

    collate_fc = partial(qa_collate, pad_id=tokenizer.pad_token_id)
    eval_dataset = QADataset(tokenizer, args.predict_file, args.max_seq_len, args.max_q_len)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        logger.info(f"Loading model from {args.init_checkpoint}")
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.use_adam:
            optimizer = Adam(optimizer_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)

        if args.fp16:
            from apex import amp
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            from apex import amp
            model = amp.initialize(model, opt_level=args.fp16_opt_level)


    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_em = 0
        train_loss_meter = AverageMeter()
        model.train()
        train_dataset = QADataset(tokenizer, args.train_file, args.max_seq_len, args.max_q_len, train=True)
        train_sampler = MhopSampler(train_dataset, num_neg=args.neg_num)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        logger.info('Start training....')
        for epoch in range(int(args.num_train_epochs)):
            for batch in tqdm(train_dataloader):
                batch_step += 1
                batch_inputs = move_to_cuda(batch["net_inputs"])
                loss = model(batch_inputs)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss_meter.update(loss.item())
                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # logger.info(f"current batch loss: {loss.item()}")
                    tb_logger.add_scalar('batch_train_loss',
                                        loss.item(), global_step)
                    tb_logger.add_scalar('smoothed_train_loss',
                                        train_loss_meter.avg, global_step)

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        metrics = predict(args, model, eval_dataloader, logger)
                        em = metrics["em"]
                        logger.info("Step %d Train loss %.2f em %.2f on epoch=%d" % (global_step, train_loss_meter.avg, em*100, epoch))
                        if best_em < em:
                            logger.info("Saving model with best em %.2f -> em %.2f on step=%d" %
                                        (best_em*100, em*100, global_step))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_em = em

            metrics = predict(args, model, eval_dataloader, logger)
            em = metrics["em"]
            logger.info("Step %d Train loss %.2f em %.2f" % (
                global_step, train_loss_meter.avg, em*100))
            tb_logger.add_scalar('dev_em', em*100, global_step)
            if best_em < em:
                logger.info("Saving model with best em %.2f -> em %.2f on epoch=%d" % (best_em*100, em*100, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                best_em = em

        logger.info("Training finished!")

    elif args.do_predict:
        metrics = predict(args, model, eval_dataloader, logger, fixed_thresh=0.8)
        logger.info(f"test performance {metrics}")

    elif args.do_test:
        eval_final(args, model, eval_dataloader, weight=0.8)

def predict(args, model, eval_dataloader, logger, fixed_thresh=None):
    model.eval()
    id2result = collections.defaultdict(list)
    id2answer = collections.defaultdict(list)
    id2gold = {}
    id2goldsp = {}
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        batch_qids = batch["qids"]
        batch_labels = batch["net_inputs"]["label"].view(-1).tolist()
        with torch.no_grad():
            outputs = model(batch_to_feed)
            scores = outputs["rank_score"]
            scores = scores.view(-1).tolist()
            if args.sp_pred:
                sp_scores = outputs["sp_score"]
                sp_scores = sp_scores.float().masked_fill(batch_to_feed["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores)
                batch_sp_scores = sp_scores.sigmoid()

            # ans_type_predicted = torch.argmax(outputs["ans_type_logits"], dim=1).view(-1).tolist()
            outs = [outputs["start_logits"], outputs["end_logits"]]
        for qid, label, score in zip(batch_qids, batch_labels, scores):
            id2result[qid].append((label, score))

        # answer prediction
        span_scores = outs[0][:, :, None] + outs[1][:, None]
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), args.max_ans_len)
        span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
        span_scores_masked = span_scores.float().masked_fill((1 - span_mask[None].expand_as(span_scores)).bool(), -1e10).type_as(span_scores)
        start_position = span_scores_masked.max(dim=2)[0].max(dim=1)[1]
        end_position = span_scores_masked.max(dim=2)[1].gather(
            1, start_position.unsqueeze(1)).squeeze(1)
        answer_scores = span_scores_masked.max(dim=2)[0].max(dim=1)[0].tolist()
        para_offset = batch['para_offsets']
        start_position_ = list(
            np.array(start_position.tolist()) - np.array(para_offset))
        end_position_ = list(
            np.array(end_position.tolist()) - np.array(para_offset)) 

        for idx, qid in enumerate(batch_qids):
            id2gold[qid] = batch["gold_answer"][idx]
            id2goldsp[qid] = batch["sp_gold"][idx]

            rank_score = scores[idx]
            start = start_position_[idx]
            end = end_position_[idx]
            span_score = answer_scores[idx]
            
            tok_to_orig_index = batch['tok_to_orig_index'][idx]
            doc_tokens = batch['doc_tokens'][idx]
            wp_tokens = batch['wp_tokens'][idx]
            orig_doc_start = tok_to_orig_index[start]
            orig_doc_end = tok_to_orig_index[end]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_tokens = wp_tokens[start:end+1]
            tok_text = " ".join(tok_tokens)
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            pred_str = get_final_text(tok_text, orig_text, do_lower_case=True, verbose_logging=False)

            # get the sp sentences
            pred_sp = []
            if args.sp_pred:
                sp_score = batch_sp_scores[idx].tolist()
                passages = batch["passages"][idx]
                for passage, sent_offset in zip(passages, [0, len(passages[0]["sents"])]):
                    for idx, _ in enumerate(passage["sents"]):
                        try:
                            if sp_score[idx + sent_offset] >= 0.5:
                                pred_sp.append([passage["title"], idx])
                        except:
                            # logger.info(f"sentence exceeds max lengths")
                            continue
            id2answer[qid].append({
                "pred_str": pred_str.strip(),
                "rank_score": rank_score,
                "span_score": span_score,
                "pred_sp": pred_sp
            })
    acc = []
    for qid, res in id2result.items():
        res.sort(key=lambda x: x[1], reverse=True)
        acc.append(res[0][0] == 1)
    logger.info(f"evaluated {len(id2result)} questions...")
    logger.info(f'chain ranking em: {np.mean(acc)}')

    best_em, best_f1, best_joint_em, best_joint_f1, best_sp_em, best_sp_f1 = 0, 0, 0, 0, 0, 0
    best_res = None
    if fixed_thresh:
        lambdas = [fixed_thresh]
    else:
        # selecting threshhold on the dev data
        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for lambda_ in lambdas:
        ems, f1s, sp_ems, sp_f1s, joint_ems, joint_f1s = [], [], [], [], [], []
        results = collections.defaultdict(dict)
        for qid, res in id2result.items():
            ans_res = id2answer[qid]
            ans_res.sort(key=lambda x: lambda_ * x["rank_score"] + (1 - lambda_) * x["span_score"], reverse=True)
            top_pred = ans_res[0]["pred_str"]
            top_pred_sp = ans_res[0]["pred_sp"]

            results["answer"][qid] = top_pred
            results["sp"][qid] = top_pred_sp

            ems.append(exact_match_score(top_pred, id2gold[qid][0]))
            f1, prec, recall = f1_score(top_pred, id2gold[qid][0])
            f1s.append(f1)

            if args.sp_pred:
                metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
                update_sp(metrics, top_pred_sp, id2goldsp[qid])
                sp_ems.append(metrics['sp_em'])
                sp_f1s.append(metrics['sp_f1'])
                # joint metrics
                joint_prec = prec * metrics["sp_prec"]
                joint_recall = recall * metrics["sp_recall"]
                if joint_prec + joint_recall > 0:
                    joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                else:
                    joint_f1 = 0.
                joint_em = ems[-1] * sp_ems[-1]
                joint_ems.append(joint_em)
                joint_f1s.append(joint_f1)

        if args.sp_pred:
            if best_joint_f1 < np.mean(joint_f1s):
                best_joint_f1 = np.mean(joint_f1s)
                best_joint_em = np.mean(joint_ems)
                best_sp_f1 = np.mean(sp_f1s)
                best_sp_em = np.mean(sp_ems)
                best_f1 = np.mean(f1s)
                best_em = np.mean(ems)
                best_res = results
        else:
            if best_f1 < np.mean(f1s):
                best_f1 = np.mean(f1s)
                best_em = np.mean(ems)

        logger.info(f".......Using combination factor {lambda_}......")
        logger.info(f'answer em: {np.mean(ems)}, count: {len(ems)}')
        logger.info(f'answer f1: {np.mean(f1s)}, count: {len(f1s)}')
        logger.info(f'sp em: {np.mean(sp_ems)}, count: {len(sp_ems)}')
        logger.info(f'sp f1: {np.mean(sp_f1s)}, count: {len(sp_f1s)}')
        logger.info(f'joint em: {np.mean(joint_ems)}, count: {len(joint_ems)}')
        logger.info(f'joint f1: {np.mean(joint_f1s)}, count: {len(joint_f1s)}')
    logger.info(f"Best joint F1 from combination {best_f1}")
    if args.save_prediction != "":
        json.dump(best_res, open(f"{args.save_prediction}", "w"))

    model.train()
    return {"em": best_em, "f1": best_f1, "joint_em": best_joint_em, "joint_f1": best_joint_f1, "sp_em": best_sp_em, "sp_f1": best_sp_f1}

import time

def eval_final(args, model, eval_dataloader, weight=0.8, gpu=True):
    """
    for final submission
    """
    model.eval()
    id2answer = collections.defaultdict(list)
    encode_times = []
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch["net_inputs"]) if gpu else batch["net_inputs"]
        batch_qids = batch["qids"]
        with torch.no_grad():
            start = time.time()
            outputs = model(batch_to_feed)
            encode_times.append(time.time() - start)

            scores = outputs["rank_score"]
            scores = scores.view(-1).tolist()

            if args.sp_pred:
                sp_scores = outputs["sp_score"]
                sp_scores = sp_scores.float().masked_fill(batch_to_feed["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores)
                batch_sp_scores = sp_scores.sigmoid()

            # ans_type_predicted = torch.argmax(outputs["ans_type_logits"], dim=1).view(-1).tolist()
            outs = [outputs["start_logits"], outputs["end_logits"]]


        # answer prediction
        span_scores = outs[0][:, :, None] + outs[1][:, None]
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), args.max_ans_len)
        span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
        span_scores_masked = span_scores.float().masked_fill((1 - span_mask[None].expand_as(span_scores)).bool(), -1e10).type_as(span_scores)
        start_position = span_scores_masked.max(dim=2)[0].max(dim=1)[1]
        end_position = span_scores_masked.max(dim=2)[1].gather(
            1, start_position.unsqueeze(1)).squeeze(1)
        answer_scores = span_scores_masked.max(dim=2)[0].max(dim=1)[0].tolist()
        para_offset = batch['para_offsets']
        start_position_ = list(
            np.array(start_position.tolist()) - np.array(para_offset))
        end_position_ = list(
            np.array(end_position.tolist()) - np.array(para_offset)) 

        for idx, qid in enumerate(batch_qids):
            rank_score = scores[idx]
            start = start_position_[idx]
            end = end_position_[idx]
            span_score = answer_scores[idx]
            tok_to_orig_index = batch['tok_to_orig_index'][idx]
            doc_tokens = batch['doc_tokens'][idx]
            wp_tokens = batch['wp_tokens'][idx]
            orig_doc_start = tok_to_orig_index[start]
            orig_doc_end = tok_to_orig_index[end]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_tokens = wp_tokens[start:end+1]
            tok_text = " ".join(tok_tokens)
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            pred_str = get_final_text(tok_text, orig_text, do_lower_case=True, verbose_logging=False)

            chain_titles = [_["title"] for _ in batch["passages"][idx]]

            # get the sp sentences
            pred_sp = []
            if args.sp_pred:
                sp_score = batch_sp_scores[idx].tolist()
                passages = batch["passages"][idx]
                for passage, sent_offset in zip(passages, [0, len(passages[0]["sents"])]):
                    for idx, _ in enumerate(passage["sents"]):
                        try:
                            if sp_score[idx + sent_offset] > 0.5:
                                pred_sp.append([passage["title"], idx])
                        except:
                            # logger.info(f"sentence exceeds max lengths")
                            continue
            id2answer[qid].append({
                "pred_str": pred_str.strip(),
                "rank_score": rank_score,
                "span_score": span_score,
                "pred_sp": pred_sp,
                "chain_titles": chain_titles
            })
    lambda_ = weight
    results = collections.defaultdict(dict)
    for qid in id2answer.keys():
        ans_res = id2answer[qid]
        ans_res.sort(key=lambda x: lambda_ * x["rank_score"] + (1 - lambda_) * x["span_score"], reverse=True)
        top_pred = ans_res[0]["pred_str"]
        top_pred_sp = ans_res[0]["pred_sp"]

        results["answer"][qid] = top_pred
        results["sp"][qid] = top_pred_sp
        results["titles"][qid] = ans_res[0]["chain_titles"]


    if args.save_prediction != "":
        json.dump(results, open(f"{args.save_prediction}", "w"))

    return results

if __name__ == "__main__":
    main()
