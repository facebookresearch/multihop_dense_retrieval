# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import uuid
import itertools
from typing import Dict
import submitit
from collections import Iterable, namedtuple
from pathlib import Path
from datetime import date

from qa_trainer import Trainer
from config import ClusterConfig, train_args

def get_shared_folder() -> Path:
    return Path("/checkpoint/xwhan/mhop-qa")

def get_init_file() -> Path:
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def grid_parameters(grid: Dict):
    """
    Yield all combinations of parameters in the grid (as a dict)
    """
    grid_copy = dict(grid)
    # Turn single value in an Iterable
    for k in grid_copy:
        if not isinstance(grid_copy[k], Iterable):
            grid_copy[k] = [grid_copy[k]]
    for p in itertools.product(*grid_copy.values()):
        yield dict(zip(grid.keys(), p))

def grid_search(args):
    cluster_cfg = ClusterConfig(dist_backend="nccl", dist_url="")

    date_curr = date.today().strftime("%m-%d-%Y")
    log_dir = os.path.join(args.output_dir, date_curr)
    
    TrainerConfig = namedtuple("TrainerConfig", sorted(vars(args)))
    train_cfg = TrainerConfig(**vars(args))

    # Create the executor
    print("Create the submitit Executor (can take time on FB cluster)")
    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=get_shared_folder() / "%j")
    num_gpus_per_node = 8
    executor.update_parameters(
        mem_gb=400,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=1,  # one task per GPU
        cpus_per_task=10,
        nodes=1,
        timeout_min=60*72,
        slurm_partition="learnfair",
        slurm_signal_delay_s=120,
        slurm_constraint='volta32gb'
    )

    # Launch one job per grid position
    grid_meta = {
        "num_train_epochs": (7, lambda val: f'epoch{val}'), 
        "learning_rate": ([2e-5, 5e-5, 3e-5], lambda val: f'lr{val}'), 
        "seed": ([42,5], lambda val: f'seed{val}'),
        "rank_drop": (0, lambda val: f'rdrop{val}'),
        "qa_drop": (0, lambda val: f'qadrop{val}'),
        # "max_seq_len": (512, lambda val: f'c_len{val}'),
        # "max_q_len": (100, lambda val: f'q_len{val}'),
        "weight_decay": (0, lambda val: f'decay{val}'),
        "num_q_per_gpu": (2, lambda val: f'qpergpu{val}'), # how many questions per gpu
        "gradient_accumulation_steps": (8, lambda val: f'aggstep{val}'),
        "max_grad_norm": (2, lambda val: f'clip{val}'),
        "eval_period": (250, lambda val: f'evalper{val}'),
        "predict_batch_size": (1024, lambda val: f'evalbsize{val}'),
        "neg_num": (5, lambda val: f'negnum{val}'),
        "warmup_ratio": ([0.1, 0.2], lambda val: f'warmup{val}'),
        "use_adam": (True, lambda val: f'adam{val}'),
        "sp_weight": ([0.05, 0.025], lambda val: f'spweight{val}'),
        "shared_norm": (False, lambda val: f'sn{val}'),
        }
    grid = {k:v[0] for k, v in grid_meta.items()}
    save_key = {k:v[1] for k, v in grid_meta.items()}
    
    hyper_parameters = list(grid_parameters(grid))
    jobs = []
    for i, grid_data in enumerate(hyper_parameters):
        cluster_cfg = cluster_cfg._replace(dist_url=get_init_file().as_uri())
        train_cfg = train_cfg._replace(**grid_data)

        run_name = f"{train_cfg.prefix}"
        for k, v in grid_data.items():
            run_name += "-" + save_key[k](v)
        train_cfg = train_cfg._replace(output_dir=os.path.join(log_dir, run_name))

        # Chronos needs a different job name each time
        executor.update_parameters(name=f"sweep_{i:02d}_{uuid.uuid4().hex}")
        trainer = Trainer(train_cfg, cluster_cfg)
        job = executor.submit(trainer)
        jobs.append(job)
        print(f"Run {i:02d} submitted with train cfg: {train_cfg}, cluster cfg: {cluster_cfg}")
    print(f"Submitted jobs ids: {','.join([str(job.job_id) for job in jobs])}")

    # Wait for the master's results of each job
    results = [job.task(0).result() for job in jobs]
    print(f"Jobs results: {results}")
    best_job = np.argmax(results)
    print(f"Best configuration: {hyper_parameters[best_job]} (val acc = {results[best_job]:.1%})")


if __name__ == "__main__":
    args = train_args()
    grid_search(args)