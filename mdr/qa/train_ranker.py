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
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from config import train_args
from reranking_datasets import RankingDataset, rank_collate
from reranking_model import RankModel
from utils import AverageMeter, convert_to_half, move_to_cuda

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
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)
    tb_logger = SummaryWriter(os.path.join(args.output_dir.replace("logs","tflogs")))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
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

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = AutoConfig.from_pretrained(args.model_name)
    model = RankModel(bert_config, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(rank_collate, pad_id=tokenizer.pad_token_id)
    if args.do_train and args.max_seq_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_len, bert_config.max_position_embeddings))

    eval_dataset = RankingDataset(
        tokenizer, args.predict_file, args.max_seq_len, args.max_q_len)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
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
        best_acc = 0
        train_loss_meter = AverageMeter()
        model.train()
        train_dataset = RankingDataset(tokenizer, args.train_file, args.max_seq_len, args.max_q_len, train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)

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
                tb_logger.add_scalar('batch_train_loss',
                                     loss.item(), global_step)
                tb_logger.add_scalar('smoothed_train_loss',
                                     train_loss_meter.avg, global_step)
            
                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        acc = predict(args, model, eval_dataloader,
                                     device, logger)
                        logger.info("Step %d Train loss %.2f acc %.2f on epoch=%d" % (global_step, train_loss_meter.avg, acc*100, epoch))

                        # save most recent model
                        torch.save(model.state_dict(), os.path.join(
                            args.output_dir, f"checkpoint_last.pt"))

                        if best_acc < acc:
                            logger.info("Saving model with best acc %.2f -> acc %.2f on epoch=%d" %
                                        (best_acc*100, acc*100, epoch))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_acc = acc

            acc = predict(args, model, eval_dataloader, device, logger)
            logger.info("Step %d Train loss %.2f acc %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, acc*100, epoch))
            tb_logger.add_scalar('dev_acc', acc*100, epoch)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_last.pt"))

            if best_acc < acc:
                logger.info("Saving model with best acc %.2f -> acc %.2f on epoch=%d" % (best_acc*100, acc*100, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                best_acc = acc

        logger.info("Training finished!")

    elif args.do_predict:
        acc = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {acc}")

def predict(args, model, eval_dataloader, device, logger):
    model.eval()
    id2result = collections.defaultdict(list)
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        batch_qids = batch["qids"]
        batch_labels = batch["net_inputs"]["label"].view(-1).tolist()
        with torch.no_grad():
            scores = model(batch_to_feed)
            scores = scores.view(-1).tolist()
        for qid, label, score in zip(batch_qids, batch_labels, scores):
            id2result[qid].append((label, score))

    acc = []
    top_pred = {}
    for qid, res in id2result.items():
        res.sort(key=lambda x: x[1], reverse=True)
        acc.append(res[0][0] == 1)
    logger.info(f"evaluated {len(id2result)} questions...")
    logger.info(f'acc: {np.mean(acc)}')
    model.train()
    return np.mean(acc)


if __name__ == "__main__":
    main()
