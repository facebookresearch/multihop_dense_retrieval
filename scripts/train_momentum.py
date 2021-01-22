# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import logging
import os
import random
from datetime import date
from functools import partial

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from mdr.retrieval.config import train_args
from mdr.retrieval.criterions import (mhop_eval, mhop_loss)
from mdr.retrieval.data.mhop_dataset import MhopDataset, mhop_collate
from mdr.retrieval.models.mhop_retriever import RobertaMomentumRetriever
from mdr.retrieval.utils.utils import AverageMeter, move_to_cuda
from mdr.retrieval.data.fever_dataset import FeverDataset

def main():
    args = train_args()
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}-m{args.m}-k{args.k}-t{args.temperature}"
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
    model = RobertaMomentumRetriever(bert_config, args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(mhop_collate, pad_id=tokenizer.pad_token_id)
    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))


    if "fever" in args.predict_file:
        eval_dataset = FeverDataset(
        tokenizer, args.predict_file, args.max_q_len, args.max_q_sp_len, args.max_c_len)
    else:
        eval_dataset = MhopDataset(
        tokenizer, args.predict_file, args.max_q_len, args.max_q_sp_len, args.max_c_len)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

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
        optimizer = Adam(optimizer_parameters,
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
        best_mrr = 0
        train_loss_meter = AverageMeter()
        model.train()

        if "fever" in args.train_file:
            train_dataset = FeverDataset(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
        else:
            train_dataset = MhopDataset(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        logger.info('Start training....')
        for epoch in range(int(args.num_train_epochs)):
            for batch in tqdm(train_dataloader):
                batch_step += 1
                batch = move_to_cuda(batch)
                loss = mhop_loss(model, batch, args)
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

                    tb_logger.add_scalar('batch_train_loss',
                                        loss.item(), global_step)
                    tb_logger.add_scalar('smoothed_train_loss',
                                        train_loss_meter.avg, global_step)

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        mrrs = predict(args, model, eval_dataloader,
                                     device, logger)
                        mrr = mrrs["mrr_avg"]
                        logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (global_step, train_loss_meter.avg, mrr*100, epoch))

                        if best_mrr < mrr:
                            logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                                        (best_mrr*100, mrr*100, epoch))
                            torch.save(model.module.encoder_q.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_q_best.pt"))
                            torch.save(model.module.encoder_q.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_k_best.pt"))
                            model = model.to(device)
                            best_mrr = mrr

            mrrs = predict(args, model, eval_dataloader, device, logger)
            mrr = mrrs["mrr_avg"]
            logger.info("Step %d Train loss %.2f MRR-AVG %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, mrr*100, epoch))
            for k, v in mrrs.items():
                tb_logger.add_scalar(k, v*100, epoch)

            if best_mrr < mrr:
                logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" % (best_mrr*100, mrr*100, epoch))
                torch.save(model.module.encoder_q.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_q_best.pt"))
                torch.save(model.module.encoder_q.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_k_best.pt"))
                best_mrr = mrr

        logger.info("Training finished!")

    elif args.do_predict:
        acc = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {acc}")

def predict(args, model, eval_dataloader, device, logger):
    model.eval()
    rrs_1, rrs_2 = [], [] # reciprocal rank
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            outputs = model(batch_to_feed)
            eval_results = mhop_eval(outputs, args)
            _rrs_1, _rrs_2 = eval_results["rrs_1"], eval_results["rrs_2"]
            rrs_1 += _rrs_1
            rrs_2 += _rrs_2
    mrr_1 = np.mean(rrs_1)
    mrr_2 = np.mean(rrs_2)
    logger.info(f"evaluated {len(rrs_1)} examples...")
    logger.info(f'MRR-1: {mrr_1}')
    logger.info(f'MRR-2: {mrr_2}')
    model.train()
    return {"mrr_1": mrr_1, "mrr_2": mrr_2, "mrr_avg": (mrr_1 + mrr_2) / 2}


if __name__ == "__main__":
    main()
