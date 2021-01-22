#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=hotpot_eval
#SBATCH --output=/private/home/xwhan/Mhop-Pretrain/eval_logs
#SBATCH --partition=dev
#SBATCH --error=/checkpoint/%u/hotpot-jobs/sample-%j.err
#SBATCH --mem=500GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append

--wrap="srun python end2end.py \
    ../data/hotpot/hotpot_qas_val.json"



sbatch --job-name=hotpot_eval \
--error=/checkpoint/hotpot-jobs/hotpot-%j.err \
--output=/checkpoint/hotpot-jobs/hotpot-%j.out \
--partition=dev --nodes=1 --ntasks-per-node=1 \
--cpus-per-task=16 \
--gpus-per-node=1 --open-mode=append \
--time=12:00:00 \
--wrap="srun python end2end.py ../data/hotpot/hotpot_qas_val.json"