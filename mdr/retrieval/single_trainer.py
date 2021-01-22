# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.


"""
trainer defined for submitit hyperparameter tuning
"""

import os
import os.path as osp
from typing import Optional, NamedTuple
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import attr
import submitit
import argparse
from functools import partial
from torch.nn import CrossEntropyLoss
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from .utils import move_to_cuda, convert_to_half, AverageMeter
from .config import ClusterConfig
from .data.sp_datasets import SPDataset, sp_collate
from .models.retriever import BertForRetrieverSP
from transformers import AdamW, BertConfig, BertTokenizer
import json

import apex
apex.amp.register_half_function(torch, 'einsum')
from apex import amp

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
        tokenizer = BertTokenizer.from_pretrained(self._train_cfg.bert_model_name)
        collate_fc = sp_collate
        train_set = SPDataset(tokenizer, self._train_cfg.train_file, self._train_cfg.max_q_len, self._train_cfg.max_c_len, train=True)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(
        #     train_set, num_replicas=job_env.num_tasks, rank=job_env.global_rank
        # )
        # self._train_loader = torch.utils.data.DataLoader(
        #     train_set,
        #     batch_size=self._train_cfg.train_batch_size,
        #     num_workers=4,
        #     sampler=train_sampler, collate_fn=collate_fc
        # )
        self._train_loader = torch.utils.data.DataLoader(train_set, batch_size=self._train_cfg.train_batch_size, num_workers=4, collate_fn=collate_fc)
        test_set = SPDataset(tokenizer, self._train_cfg.predict_file, self._train_cfg.max_q_len, self._train_cfg.max_c_len)
        self._test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self._train_cfg.predict_batch_size,
            num_workers=4, collate_fn=collate_fc
        )
        print(f"Per Node batch_size: {self._train_cfg.train_batch_size // job_env.num_tasks}", flush=True)

        print("Create model", flush=True)
        print(f"Local rank {job_env.local_rank}", flush=True)
        bert_config = BertConfig.from_pretrained(self._train_cfg.bert_model_name)
        model = BertForRetrieverSP(bert_config, self._train_cfg)
        model.cuda(job_env.local_rank)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self._train_cfg.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_parameters,
                          lr=self._train_cfg.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5)

        if self._train_cfg.fp16:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=self._train_cfg.fp16_opt_level)
        model = torch.nn.DataParallel(model) # 
        self._state = TrainerState(
            epoch=0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, global_step=0
        )

        self.tb_logger = SummaryWriter(os.path.join(self._train_cfg.output_dir, "tblog"))

        checkpoint_fn = osp.join(self._train_cfg.output_dir, str(job_env.job_id), "checkpoint.pth")
        # checkpoint_fn = osp.join(self._train_cfg.output_dir, "checkpoint.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(
                checkpoint_fn, default=self._state, gpu=job_env.local_rank)

    def _train(self) -> Optional[float]:
        job_env = submitit.JobEnvironment()

        loss_fct = CrossEntropyLoss()
        batch_step = 0 # forward batch count
        best_mrr = 0
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
                batch = move_to_cuda(batch)
                outputs = self._state.model(batch)
                q = outputs['q']
                c = outputs['c']
                neg_c = outputs['neg_c']
                product_in_batch = torch.mm(q, c.t())
                product_neg = (q * neg_c).sum(-1).unsqueeze(1)
                product = torch.cat([product_in_batch, product_neg], dim=-1)
                target = torch.arange(product.size(0)).to(product.device)
                loss = loss_fct(product, target)

                if self._train_cfg.gradient_accumulation_steps > 1:
                    loss = loss / self._train_cfg.gradient_accumulation_steps
                if self._train_cfg.fp16:
                    with amp.scale_loss(loss, self._state.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                train_loss_meter.update(loss.item())
                self.tb_logger.add_scalar('batch_train_loss',
                                     loss.item(), global_step)
                self.tb_logger.add_scalar('smoothed_train_loss',
                                     train_loss_meter.avg, global_step)

                if (batch_step + 1) % self._train_cfg.gradient_accumulation_steps == 0:
                    if self._train_cfg.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self._state.optimizer), self._train_cfg.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self._state.model.parameters(), self._train_cfg.max_grad_norm)
                    self._state.optimizer.step()    # We have accumulated enought gradients
                    self._state.model.zero_grad()
                    global_step += 1
                    self._state.global_step = global_step

            # Checkpoint only on the master
            # if job_env.global_rank == 0:
            self.checkpoint(rm_init=False)
            mrr = self._eval()
            self.tb_logger.add_scalar('dev_mrr', mrr*100, epoch)
            self._state.lr_scheduler.step(mrr)
            if best_mrr < mrr:
                print("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" % (best_mrr*100, mrr*100, epoch))
                torch.save(self._state.model.state_dict(), os.path.join(self._train_cfg.output_dir, f"checkpoint_best.pt"))
                best_mrr = mrr
            self.log({
                "best_mrr": best_mrr,
                "curr_mrr": mrr,
                "smoothed_loss": train_loss_meter.avg,
                "epoch": epoch
            })
        return best_mrr

    def _eval(self) -> float:
        print("Start evaluation of the model", flush=True)
        job_env = submitit.JobEnvironment()
        args = self._train_cfg
        eval_dataloader = self._test_loader
        num_correct = 0
        num_total = 0.0
        rrs = [] # reciprocal rank
        self._state.model.eval()
        for batch in self._test_loader:
            batch_to_feed = move_to_cuda(batch)
            with torch.no_grad():
                outputs = self._state.model(batch_to_feed)
                q = outputs['q']
                c = outputs['c']
                neg_c = outputs['neg_c']

                product_in_batch = torch.mm(q, c.t())            
                product_neg = (q * neg_c).sum(-1).unsqueeze(1)
                product = torch.cat([product_in_batch, product_neg], dim=-1)

                target = torch.arange(product.size(0)).to(product.device)
                ranked = product.argsort(dim=1, descending=True)

                # MRR
                idx2rank = ranked.argsort(dim=1)
                for idx, t in enumerate(target.tolist()):
                    rrs.append(1 / (idx2rank[idx][t].item() +1))

                prediction = product.argmax(-1)
                pred_res = prediction == target

                num_total += pred_res.size(0)
                num_correct += pred_res.sum(0)

        acc = num_correct/num_total
        mrr = np.mean(rrs)
        print(f"evaluated {num_total} examples...", flush=True)
        print(f"avg. Acc: {acc}", flush=True)
        print(f'MRR: {mrr}', flush=True)
        self._state.model.train()
        return mrr
