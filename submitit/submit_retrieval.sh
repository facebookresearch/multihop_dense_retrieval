#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
#!/bin/bash

MKL_THREADING_LAYER=GNU  python submitit_train.py --prefix mhop_retrieval_roberta \
    --train_file /private/home/xwhan/data/hotpot/hotpot_train_with_neg_v0.json \
    --predict_file /private/home/xwhan/data/hotpot/hotpot_val_with_neg_v0.json  \
    --model_name roberta-base \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --fp16
