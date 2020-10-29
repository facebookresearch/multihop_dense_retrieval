"""
single and multihop queries together

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_unified.py \
    --do_train \
    --prefix fever_unified_roberta \
    --predict_batch_size 5000 \
    --model_name roberta-base \
    --train_batch_size 80 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/fever/retrieval/train_tfidf_neg.txt \
    --predict_file /private/home/xwhan/data/fever/retrieval/dev_tfidf_neg.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 400 \
    --max_q_len 45 \
    --max_q_sp_len 400 \
    --use-adam \
    --warmup-ratio 0.1 \
    --num_train_epochs 50 \
    --accumulate_gradients 1 \
    --gradient_accumulation_steps 1 \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_unified.py \
    --do_train \
    --prefix fever_unified_roberta_balanced \
    --predict_batch_size 5000 \
    --model_name roberta-base \
    --train_batch_size 80 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/fever/retrieval/train_tfidf_neg.txt \
    --predict_file /private/home/xwhan/data/fever/retrieval/dev_tfidf_neg.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 400 \
    --max_q_len 45 \
    --max_q_sp_len 400 \
    --use-adam \
    --warmup-ratio 0.1 \
    --num_train_epochs 50
"""
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

from config import train_args
from criterions import unified_loss, unified_eval
from data.unified_dataset import UnifiedDataset, unified_collate, FeverUnifiedDataset, FeverSampler
from models.unified_retriever import UnifiedRetriever
from utils.utils import AverageMeter, move_to_cuda


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
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-adam{args.use_adam}"
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
    model = UnifiedRetriever(bert_config, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(unified_collate, pad_id=tokenizer.pad_token_id)
    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    if "fever" in args.predict_file:
        eval_dataset = FeverUnifiedDataset(tokenizer, args.predict_file, args.max_q_len, args.max_q_sp_len, args.max_c_len)
    else:
        eval_dataset = UnifiedDataset(
        tokenizer, args.predict_file, args.max_q_len, args.max_q_sp_len, args.max_c_len)
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

    model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_mrr = 0
        train_loss_meter = AverageMeter()
        model.train()
        if "fever" in args.train_file:
            train_dataset = FeverUnifiedDataset(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
            train_sampler = FeverSampler(train_dataset)
            logger.info(f"Total train samples {len(train_sampler)}")
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)
        else:
            train_dataset = UnifiedDataset(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
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
                loss = unified_loss(model, batch, args)
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
                        mrr = predict(args, model, eval_dataloader,
                                     device, logger)
                        logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (global_step, train_loss_meter.avg, mrr*100, epoch))

                        # save most recent model
                        torch.save(model.state_dict(), os.path.join(
                            args.output_dir, f"checkpoint_last.pt"))

                        if best_mrr < mrr:
                            logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                                        (best_mrr*100, mrr*100, epoch))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_mrr = mrr

            mrr = predict(args, model, eval_dataloader, device, logger)
            logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, mrr*100, epoch))
            tb_logger.add_scalar('dev_mrr', mrr*100, epoch)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_last.pt"))
            if best_mrr < mrr:
                logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" % (best_mrr*100, mrr*100, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                best_mrr = mrr

        logger.info("Training finished!")

    elif args.do_predict:
        acc = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {acc}")

def predict(args, model, eval_dataloader, device, logger):
    model.eval()
    rrs_1_mhop, rrs_2_mhop, rrs_nq = [], [], [] # reciprocal rank
    stop_acc = []
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            outputs = model(batch_to_feed)
            eval_results = unified_eval(outputs, batch_to_feed)
            _stop_acc, _rrs_1_mhop, _rrs_2_mhop, _rrs_nq = eval_results["stop_acc"], eval_results["rrs_1_mhop"], eval_results["rrs_2_mhop"], eval_results["rrs_nq"]
            stop_acc += _stop_acc
            rrs_1_mhop += _rrs_1_mhop
            rrs_2_mhop += _rrs_2_mhop
            rrs_nq += _rrs_nq
            
    logger.info(f'Multihop MRR-1: {np.mean(rrs_1_mhop)}, count: {len(rrs_1_mhop)}')
    logger.info(f'Multihop MRR-2: {np.mean(rrs_2_mhop)}, count: {len(rrs_2_mhop)}')
    logger.info(f'Single-Hop MRR: {np.mean(rrs_nq)}, count: {len(rrs_nq)}')
    logger.info(f'Stop Acc: {np.mean(stop_acc)}, count: {len(stop_acc)}')

    model.train()
    return (np.mean(rrs_1_mhop) + np.mean(rrs_2_mhop) + np.mean(rrs_nq) + np.mean(stop_acc)) / 4


if __name__ == "__main__":
    main()
