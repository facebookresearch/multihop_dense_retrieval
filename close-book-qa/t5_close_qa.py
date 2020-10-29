import collections
import logging
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from adafactor import Adafactor
from collections import defaultdict, namedtuple
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from utils import move_to_cuda, AverageMeter, exact_match_score, metric_max_over_ground_truths
from config import get_args
from datasets import QAPairs, collate
from functools import partial

from complexwebq_eval import complexQ_em

def load_saved(model, path):
    state_dict = torch.load(path)
    def filter(x): return x[7:] if x.startswith('module.') else x
    state_dict = {filter(k): v for (k, v) in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def run(args):

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-{args.model_name}"

    tb_logger = SummaryWriter(os.path.join(args.output_dir, "tflogs", model_name))
    args.output_dir = os.path.join(args.output_dir, model_name)
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

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError(
                "If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    model_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "t5" in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        model = MyBart.from_pretrained(args.model_name)    

    if args.do_train and args.max_q_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, model_config.max_position_embeddings))
    
    if "t5" in args.model_name:
        collate_fc = partial(collate, pad_id=tokenizer.pad_token_id, target_pad_id=-100)
    else:
        collate_fc = partial(collate, pad_id=tokenizer.pad_token_id, target_pad_id=tokenizer.pad_token_id)

    eval_dataset = QAPairs(tokenizer, args.predict_file, args.max_q_length, args.max_ans_length, decode_bridge=args.decode_bridge)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=False)

    if args.init_checkpoint != "":
        logger.info(f"Loading checkpoint from {args.init_checkpoint}")
        model = load_saved(model, args.init_checkpoint)
    model.to(device)


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

        elif args.use_adafactor:
            optimizer = Adafactor(optimizer_parameters,
                          lr=args.learning_rate, scale_parameter=False, relative_step=False)
        else:
            optimizer = AdamW(optimizer_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        logger.info('Start training....')
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        train_loss_meter = AverageMeter()
        best_em = 0
        model.train()
        train_dataset = QAPairs(tokenizer, args.train_file, args.max_q_length, args.max_ans_length, decode_bridge=args.decode_bridge)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)
        for epoch in range(int(args.num_train_epochs)):
            
            for batch in tqdm(train_dataloader):
                batch_step += 1
                net_inputs = move_to_cuda(batch["net_inputs"])
                if "t5" in args.model_name:
                    outputs = model(input_ids=net_inputs["input_ids"], attention_mask=net_inputs["attention_mask"], lm_labels=net_inputs["lm_labels"])
                    loss = outputs[0]
                else:
                    loss = model(input_ids=net_inputs["input_ids"], attention_mask=net_inputs["attention_mask"], decoder_input_ids=net_inputs["lm_labels"], decoder_attention_mask=net_inputs["target_mask"])
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

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
                        em = predict(args, model, eval_dataloader, tokenizer, logger)
                        logger.info("Step %d Train loss %.2f EM %.2f on epoch=%d" % (
                            global_step, train_loss_meter.avg, em, epoch))
                        tb_logger.add_scalar('dev_em', em, global_step)
                        if best_em < em:
                            logger.info("Saving model with best  EM %.2f -> EM %.2f on epoch=%d" % (best_em, em, epoch))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_em = em

            em = predict(args, model, eval_dataloader, tokenizer, logger)
            logger.info("Step %d Train loss %.2f EM %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, em, epoch))
            tb_logger.add_scalar('dev_em', em, global_step)
            if best_em < em:
                logger.info("Saving model with best  EM %.2f -> EM %.2f on epoch=%d" % (best_em, em, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                model = model.to(device)
                best_em = em

    elif args.do_predict:
        em = predict(args, model, eval_dataloader, tokenizer, logger)
        logger.info(f"test performance {em}")


def predict(args, model, eval_dataloader, tokenizer, logger):
    model.eval()
    ems = []

    type2ems = collections.defaultdict(list)
    level2ems = collections.defaultdict(list)
    for batch in tqdm(eval_dataloader):
        net_inputs = move_to_cuda(batch["net_inputs"])
        with torch.no_grad():
            outputs = model.module.generate(input_ids=net_inputs["input_ids"], attention_mask=net_inputs["attention_mask"],
                                 num_beams=args.num_beams,
                                 max_length=args.max_ans_length)
            for idx, pred_tokens in enumerate(outputs):
                pred_str = tokenizer.decode(pred_tokens, skip_special_tokens=True).lower()
                gold = batch["answers"][idx] 
                
                if args.complex:
                    ems.append(int(complexQ_em(pred_str.lower().strip(), gold, batch["questions"][idx])))
                else:
                    ems.append(metric_max_over_ground_truths(exact_match_score, pred_str, gold))
                if "types" in batch:
                    type2ems[batch["types"][idx]].append(ems[-1])
                if "levels" in batch:
                    level2ems[batch["levels"][idx]].append(ems[-1])

    if "types" in batch:
        for k, v in type2ems.items():
            logger.info(f"Type: {k} Count: {len(v)}")
            logger.info(f"EM: {np.mean(v)}")

    if "levels" in batch:
        for k, v in level2ems.items():
            logger.info(f"level: {k} Count: {len(v)}")
            logger.info(f"EM: {np.mean(v)}")
    

    model.train()
    logger.info(f"Evaluating {len(ems)} examples...")
    return np.mean(ems) * 100


if __name__ == "__main__":
    args = get_args()
    run(args)