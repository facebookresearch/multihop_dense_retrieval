"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_nq_mhop.py \
    --do_train \
    --prefix nq_mhop_add_marker_type2_clip \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --train_batch_size 200 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/nq-dpr/nq-mhop/nq-mhop-train-dpr-shared-top100.txt \
    --predict_file /private/home/xwhan/data/nq-dpr/nq-mhop/nq-mhop-val.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 256 \
    --max_q_sp_len 300 \
    --max_q_len 50 \
    --warmup-ratio 0.1 \
    --num_train_epochs 60


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_nq_mhop.py \
    --do_train \
    --prefix wq_mhop_2_marker_shared_dpr_neg_scratch_only1hoploss \
    --predict_batch_size 500 \
    --model_name bert-base-uncased \
    --train_batch_size 150 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/WebQ/wq-mhop/wq-mhop-train-dpr-shared-top100.txt \
    --predict_file /private/home/xwhan/data/WebQ/wq-mhop/wq-mhop-val-dpr-shared-top100.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_sp_len 350 \
    --max_q_len 50 \
    --warmup-ratio 0.1 \
    --num_train_epochs 60 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_nq_mhop.py \
    --do_train \
    --prefix wq_mhop_1_shared_dpr_neg_from_scratch \
    --predict_batch_size 500 \
    --model_name bert-base-uncased \
    --train_batch_size 150 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/WebQ/wq-mhop/wq-mhop-train-dpr-shared-top100.txt \
    --predict_file /private/home/xwhan/data/WebQ/wq-mhop/wq-mhop-val-dpr-shared-top100.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_sp_len 350 \
    --max_q_len 50 \
    --warmup-ratio 0.1 \
    --num_train_epochs 60 \
    # --init_checkpoint logs/08-23-2020/nq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt
"""

import logging
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from datetime import date
from torch.utils.data import DataLoader
from models.unified_retriever import RobertaNQRetriever, BertNQRetriever, BertNQMomentumRetriever
from transformers import AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from data.unified_dataset import NQUnifiedDataset, nq_unified_collate
from utils.utils import move_to_cuda, AverageMeter, load_saved
from config import train_args
from criterions import loss_nq_mhop, eval_nq_mhop
from torch.optim import Adam
from functools import partial
import apex

def main():
    args = train_args()

    if args.fp16:
        apex.amp.register_half_function(torch, 'einsum')

    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-{args.model_name}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)
    tb_logger = SummaryWriter(os.path.join(args.output_dir.replace("logs","tflogs")))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
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

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    bert_config = AutoConfig.from_pretrained(args.model_name)
    if "roberta" in args.model_name:
        model = RobertaNQRetriever(bert_config, args)
    else:
        if args.momentum:
            model = BertNQMomentumRetriever(bert_config, args)
        else:
            model = BertNQRetriever(bert_config, args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(nq_unified_collate, pad_id=tokenizer.pad_token_id)

    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    eval_dataset = NQUnifiedDataset(tokenizer, args.predict_file, args.max_q_len, args.max_q_sp_len, args.max_c_len)
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
        optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.fp16:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            model = apex.amp.initialize(model, opt_level=args.fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_mrr = 0
        train_loss_meter = AverageMeter()
        model.train()
        train_dataset = NQUnifiedDataset(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
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
                loss = loss_nq_mhop(model, batch, args.momentum)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss_meter.update(loss.item())

                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            apex.amp.master_params(optimizer), args.max_grad_norm)
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

                        if best_mrr < mrr:
                            logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" % (best_mrr*100, mrr*100, epoch))
                            if args.momentum:
                                torch.save(model.module.encoder_q.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_q_best.pt"))
                            else:
                                torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_mrr = mrr

            mrr = predict(args, model, eval_dataloader, device, logger)
            logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, mrr*100, epoch))
            tb_logger.add_scalar('dev_mrr', mrr*100, epoch)
            if best_mrr < mrr:
                logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" % (best_mrr*100, mrr*100, epoch))
                if args.momentum:
                    torch.save(model.module.encoder_q.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_q_best.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                model = model.to(device)
                best_mrr = mrr

        logger.info("Training finished!")

    elif args.do_predict:
        acc = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {acc}")


def predict(args, model, eval_dataloader, device, logger):
    model.eval()

    rrs = [] # reciprocal rank
    rrs_2hop = []
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            batch_rrs, batch_rrs_2hop = eval_nq_mhop(model, batch_to_feed)
            rrs += batch_rrs
            rrs_2hop += batch_rrs_2hop

    mrr = np.mean(rrs)
    logger.info(f"evaluated {len(rrs)} examples...")
    logger.info(f'MRR: {mrr}')
    logger.info(f'MRR-2: {np.mean(rrs_2hop)}')
    model.train()
    return (mrr + np.mean(rrs_2hop)) / 2


if __name__ == "__main__":
    main()