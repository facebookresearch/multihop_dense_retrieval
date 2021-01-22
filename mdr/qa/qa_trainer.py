# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import argparse
import json
import os
import os.path as osp
import random
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional
import collections
from torch.optim import lr_scheduler
from tqdm import tqdm

import apex
import attr
import numpy as np
import submitit
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from apex import amp
from torch.utils.tensorboard import SummaryWriter
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from config import ClusterConfig
from hotpot_evaluate_v1 import exact_match_score, f1_score, update_sp
from qa_model import QAModel
from reranking_datasets import RankingDataset, rank_collate, MhopSampler
from utils import AverageMeter, move_to_cuda, get_final_text

apex.amp.register_half_function(torch, 'einsum')

@attr.s(auto_attribs=True)
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    epoch: int
    model: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    global_step: int

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "TrainerState", gpu: int) -> "TrainerState":
        data = torch.load(filename, map_location=lambda storage, loc: storage.cuda(gpu))
        # We need this default to load the state dict
        model = default.model
        model.load_state_dict(data["model"])
        data["model"] = model

        optimizer = default.optimizer
        optimizer.load_state_dict(data["optimizer"])
        data["optimizer"] = optimizer

        lr_scheduler = default.lr_scheduler
        lr_scheduler.load_state_dict(data["lr_scheduler"])
        data["lr_scheduler"] = lr_scheduler

        return cls(**data)

class Trainer:
    def __init__(self, train_cfg: NamedTuple, cluster_cfg: ClusterConfig) -> None:
        self._train_cfg = train_cfg
        self._cluster_cfg = cluster_cfg

    def __call__(self) -> Optional[float]:
        """
        Called by submitit for each task.
        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        self._init_state()
        final_acc = self._train()
        return final_acc

    def log(self, log_data: dict):
        job_env = submitit.JobEnvironment()
        # z = {**vars(self._train_cfg), **log_data}
        save_dir = Path(self._train_cfg.output_dir)
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir / 'log.txt', 'a') as f:
            f.write(json.dumps(log_data) + '\n')

    def checkpoint(self, rm_init=True) -> submitit.helpers.DelayedSubmission:
        # will be called by submitit in case of preemption
        job_env = submitit.JobEnvironment()
        save_dir = osp.join(self._train_cfg.output_dir, str(job_env.job_id))
        os.makedirs(save_dir, exist_ok=True)
        self._state.save(osp.join(save_dir, "checkpoint.pth"))

        # Trick here: when the job will be requeue, we will use the same init file
        # but it must not exist when we initialize the process group
        # so we delete it, but only when this method is called by submitit for requeue
        if rm_init and osp.exists(self._cluster_cfg.dist_url[7:]):
            os.remove(self._cluster_cfg.dist_url[7:])  # remove file:// at the beginning
        # This allow to remove any non-pickable part of the Trainer instance.
        empty_trainer = Trainer(self._train_cfg, self._cluster_cfg)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_process_group(self) -> None:
        job_env = submitit.JobEnvironment()
        torch.cuda.set_device(job_env.local_rank)
        torch.distributed.init_process_group(
            backend=self._cluster_cfg.dist_backend,
            init_method=self._cluster_cfg.dist_url,
            world_size=job_env.num_tasks,
            rank=job_env.global_rank,
        )
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

    def _init_state(self) -> None:
        """
        Initialize the state and load it from an existing checkpoint if any
        """
        job_env = submitit.JobEnvironment()

        if job_env.global_rank == 0:
            # config_path = Path(args.save_folder) / str(job_env.job_id) / 'config.json'
            os.makedirs(self._train_cfg.output_dir, exist_ok=True)
            config_path = Path(self._train_cfg.output_dir)  / 'config.json'
            with open(config_path, "w") as g:
                g.write(json.dumps(self._train_cfg._asdict()))

        print(f"Setting random seed {self._train_cfg.seed}", flush=True)
        random.seed(self._train_cfg.seed)
        np.random.seed(self._train_cfg.seed)
        torch.manual_seed(self._train_cfg.seed)

        print("Create data loaders", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self._train_cfg.model_name)
        collate_fc = partial(rank_collate, pad_id=tokenizer.pad_token_id)
        train_set = RankingDataset(tokenizer, self._train_cfg.train_file, self._train_cfg.max_seq_len, self._train_cfg.max_q_len, train=True)

        train_sampler = MhopSampler(train_set, num_neg=self._train_cfg.neg_num)

        batch_size_per_gpu = (1 + self._train_cfg.neg_num) * self._train_cfg.num_q_per_gpu
        n_gpu = torch.cuda.device_count()
        print(f"Number of GPUs: {n_gpu}", flush=True)
        print(f"Batch size per node: {batch_size_per_gpu * n_gpu}", flush=True)

        self._train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_per_gpu * n_gpu, num_workers=self._train_cfg.num_workers, collate_fn=collate_fc, sampler=train_sampler)
        test_set = RankingDataset(tokenizer, self._train_cfg.predict_file, self._train_cfg.max_seq_len, self._train_cfg.max_q_len)
        self._test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self._train_cfg.predict_batch_size,
            num_workers=self._train_cfg.num_workers, collate_fn=collate_fc
        )

        print("Create model", flush=True)
        print(f"Local rank {job_env.local_rank}", flush=True)
        bert_config = AutoConfig.from_pretrained(self._train_cfg.model_name)
        model = QAModel(bert_config, self._train_cfg)
        model.cuda(job_env.local_rank)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self._train_cfg.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if self._train_cfg.use_adam:
            optimizer = optim.Adam(optimizer_parameters, lr=self._train_cfg.learning_rate)
        else:
            optimizer = AdamW(optimizer_parameters, lr=self._train_cfg.learning_rate)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        if self._train_cfg.fp16:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=self._train_cfg.fp16_opt_level)

        t_total = len(self._train_loader) // self._train_cfg.gradient_accumulation_steps * self._train_cfg.num_train_epochs
        warmup_steps = t_total * self._train_cfg.warmup_ratio
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        model = torch.nn.DataParallel(model)
        self._state = TrainerState(
            epoch=0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, global_step=0
        )
        self.tb_logger = SummaryWriter(self._train_cfg.output_dir.replace("logs", "tflogs"))

        checkpoint_fn = osp.join(self._train_cfg.output_dir, str(job_env.job_id), "checkpoint.pth")
        # checkpoint_fn = osp.join(self._train_cfg.output_dir, "checkpoint.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(
                checkpoint_fn, default=self._state, gpu=job_env.local_rank)

    def _train(self) -> Optional[float]:
        job_env = submitit.JobEnvironment()
        batch_step = 0 # forward batch count
        best_metric = 0
        train_loss_meter = AverageMeter()
        print(f"Start training", flush=True)
        # Start from the loaded epoch
        start_epoch = self._state.epoch
        global_step = self._state.global_step
        for epoch in range(start_epoch, self._train_cfg.num_train_epochs):
            print(f"Start epoch {epoch}", flush=True)
            self._state.model.train()
            self._state.epoch = epoch

            for batch in self._train_loader:
                batch_step += 1
                batch_inputs = move_to_cuda(batch["net_inputs"])
                loss = self._state.model(batch_inputs)
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                if self._train_cfg.gradient_accumulation_steps > 1:
                    loss = loss / self._train_cfg.gradient_accumulation_steps
                if self._train_cfg.fp16:
                    with amp.scale_loss(loss, self._state.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss_meter.update(loss.item())
                if (batch_step + 1) % self._train_cfg.gradient_accumulation_steps == 0:
                    if self._train_cfg.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self._state.optimizer), self._train_cfg.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self._state.model.parameters(), self._train_cfg.max_grad_norm)
                    self._state.optimizer.step()
                    self._state.lr_scheduler.step()
                    self._state.model.zero_grad()
                    global_step += 1
                    self._state.global_step = global_step

                    self.tb_logger.add_scalar('batch_train_loss',
                                        loss.item(), global_step)
                    self.tb_logger.add_scalar('smoothed_train_loss',
                                        train_loss_meter.avg, global_step)
                    if job_env.global_rank == 0:
                        if self._train_cfg.eval_period != -1 and global_step % self._train_cfg.eval_period == 0:
                            metrics = self._eval()
                            for k, v in metrics.items():
                                self.tb_logger.add_scalar(k, v*100, global_step)
                            score = metrics[self._train_cfg.final_metric]
                            if best_metric < score:
                                print("Saving model with best %s %.2f -> em %.2f" % (self._train_cfg.final_metric, best_metric*100, score*100), flush=True)
                                torch.save(self._state.model.state_dict(), os.path.join(self._train_cfg.output_dir, f"checkpoint_best.pt"))
                                best_metric = score
            # Checkpoint only on the master
            if job_env.global_rank == 0:
                self.checkpoint(rm_init=False)
                metrics = self._eval()
                for k, v in metrics.items():
                    self.tb_logger.add_scalar(k, v*100, global_step)
                score = metrics[self._train_cfg.final_metric]
                if best_metric < score:
                    print("Saving model with best %s %.2f -> em %.2f" % (self._train_cfg.final_metric, best_metric*100, score*100), flush=True)
                    torch.save(self._state.model.state_dict(), os.path.join(self._train_cfg.output_dir, f"checkpoint_best.pt"))
                    best_metric = score
                self.log({
                    "best_score": best_metric,
                    "curr_score": score,
                    "smoothed_loss": train_loss_meter.avg,
                    "epoch": epoch
                })
        return best_metric

    def _eval(self) -> dict:
        print("Start evaluation of the model", flush=True)
        job_env = submitit.JobEnvironment()
        args = self._train_cfg
        eval_dataloader = self._test_loader
        model = self._state.model
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
                sp_score = batch_sp_scores[idx].tolist()
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

                pred_sp = []
                passages = batch["passages"][idx]
                for passage, sent_offset in zip(passages, [0, len(passages[0]["sents"])]):
                    for idx, _ in enumerate(passage["sents"]):
                        try:
                            if sp_score[idx + sent_offset] > 0.5:
                                pred_sp.append([passage["title"], idx])
                        except:
                            continue
                id2answer[qid].append((pred_str.strip(), rank_score, span_score, pred_sp))

        acc = []
        for qid, res in id2result.items():
            res.sort(key=lambda x: x[1], reverse=True)
            acc.append(res[0][0] == 1)
        print(f"evaluated {len(id2result)} questions...", flush=True)
        print(f'chain ranking em: {np.mean(acc)}', flush=True)

        best_em, best_f1, best_joint_em, best_joint_f1 = 0, 0, 0, 0
        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for lambda_ in lambdas:
            ems, f1s = [], []
            sp_ems, sp_f1s = [], []
            joint_ems, joint_f1s = [], []
            for qid, res in id2result.items():
                ans_res = id2answer[qid]
                ans_res.sort(key=lambda x: lambda_ * x[1] + (1 - lambda_) * x[2], reverse=True)
                top_pred = ans_res[0][0]
                ems.append(exact_match_score(top_pred, id2gold[qid][0]))
                f1, prec, recall = f1_score(top_pred, id2gold[qid][0])
                f1s.append(f1)

                top_pred_sp = ans_res[0][3]
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
                    joint_f1 = 0
                joint_em = ems[-1] * sp_ems[-1]
                joint_ems.append(joint_em)
                joint_f1s.append(joint_f1)

            if best_joint_f1 < np.mean(joint_f1s):
                best_joint_f1 = np.mean(joint_f1s)
                best_joint_em = np.mean(joint_ems)
                best_f1 = np.mean(f1s)
                best_em = np.mean(ems)

            print(f".......Using combination factor {lambda_}......", flush=True)
            print(f'answer em: {np.mean(ems)}, count: {len(ems)}', flush=True)
            print(f'answer f1: {np.mean(f1s)}, count: {len(f1s)}', flush=True)
            print(f'sp em: {np.mean(sp_ems)}, count: {len(sp_ems)}', flush=True)
            print(f'sp f1: {np.mean(sp_f1s)}, count: {len(sp_f1s)}', flush=True)
            print(f'joint em: {np.mean(joint_ems)}, count: {len(joint_ems)}', flush=True)
            print(f'joint f1: {np.mean(joint_f1s)}, count: {len(joint_f1s)}', flush=True)
        print(f"Best joint EM/F1 from combination {best_em}/{best_f1}", flush=True)

        model.train()
        return {"em": best_em, "f1": best_f1, "joint_em": best_joint_em, "joint_f1": best_joint_f1}

