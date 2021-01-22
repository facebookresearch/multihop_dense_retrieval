#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
#!/bin/bash

PREFIX=wwm_val_top50
MODEL_BACKEND=bert-large-uncased-whole-word-masking

MKL_THREADING_LAYER=GNU  python submitit_train_qa.py --prefix ${PREFIX} \
    --train_file /private/home/xwhan/data/hotpot/dense_train_b100_k100_sents.json \
    --predict_file /private/home/xwhan/data/hotpot/dense_val_b50_k50_roberta_sents.json \
    --model_name ${MODEL_BACKEND} \
    --fp16 \
    --sp-pred \
    