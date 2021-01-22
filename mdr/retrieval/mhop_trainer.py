# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

"""
submitit trainer for hyperparameter tuning
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
from functools import partial
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from transformers import (
    AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup)

from torch.optim import Adam
from .utils.utils import move_to_cuda, AverageMeter
from .config import ClusterConfig
from .data.mhop_dataset import MhopDataset, mhop_collate
from .models.mhop_retriever import (MhopRetriever, RobertaRetriever)
from .criterions import (mhop_loss, mhop_eval)

from tqdm import tqdm
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
        torch.cuda.manual_seed_all(self._train_cfg.seed)

        print("Create data loaders", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self._train_cfg.model_name)
        collate_fc = partial(mhop_collate, pad_id=tokenizer.pad_token_id)
        train_set = MhopDataset(tokenizer, self._train_cfg.train_file,  self._train_cfg.max_q_len, self._train_cfg.max_q_sp_len, self._train_cfg.max_c_len, train=True)

        self._train_loader = torch.utils.data.DataLoader(train_set, batch_size=self._train_cfg.train_batch_size, num_workers=self._train_cfg.num_workers, collate_fn=collate_fc, shuffle=True)
        test_set = MhopDataset(tokenizer, self._train_cfg.predict_file, self._train_cfg.max_q_len, self._train_cfg.max_q_sp_len, self._train_cfg.max_c_len)

        self._test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self._train_cfg.predict_batch_size,
            num_workers=self._train_cfg.num_workers, collate_fn=collate_fc, pin_memory=True
        )

        print("Create model", flush=True)
        print(f"Local rank {job_env.local_rank}", flush=True)
        bert_config = AutoConfig.from_pretrained(self._train_cfg.model_name)
        if "roberta" in self._train_cfg.model_name:
            model = RobertaRetriever(bert_config, self._train_cfg)
        else:
            model = MhopRetriever(bert_config, self._train_cfg)
        model.cuda(job_env.local_rank)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self._train_cfg.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters, lr=self._train_cfg.learning_rate, eps=self._train_cfg.adam_epsilon)

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
                loss = mhop_loss(self._state.model, batch, self._train_cfg)

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

            # Checkpoint only on the master
            # if job_env.global_rank == 0:
            self.checkpoint(rm_init=False)
            mrrs = self._eval()
            mrr = mrrs["mrr_avg"]
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
        self._state.model.eval()
        rrs_1, rrs_2 = [], [] # reciprocal rank
        for batch in tqdm(eval_dataloader):
            batch_to_feed = move_to_cuda(batch)
            with torch.no_grad():
                outputs = self._state.model(batch_to_feed)
                eval_results = mhop_eval(outputs, args)
                _rrs_1, _rrs_2 = eval_results["rrs_1"], eval_results["rrs_2"]
                rrs_1 += _rrs_1
                rrs_2 += _rrs_2
        mrr_1 = np.mean(rrs_1)
        mrr_2 = np.mean(rrs_2)
        print(f"evaluated {len(rrs_1)} examples...")
        print(f'MRR-1: {mrr_1}')
        print(f'MRR-2: {mrr_2}')
        self._state.model.train()
        return {"mrr_1": mrr_1, "mrr_2": mrr_2, "mrr_avg": (mrr_1 + mrr_2) / 2}
