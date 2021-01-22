# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import argparse
from ast import parse

from typing import NamedTuple

from torch.nn import parallel
class ClusterConfig(NamedTuple):
    dist_backend: str
    dist_url: str

def common_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--train_file", type=str,
                        default="../data/nq-with-neg-train.txt")
    parser.add_argument("--predict_file", type=str,
                        default="../data/nq-with-neg-dev.txt")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action="store_true", help="for final test submission")

    # model
    parser.add_argument("--model_name",
                        default="bert-base-uncased", type=str)
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_q_len", default=64, type=int)
    parser.add_argument("--max_ans_len", default=35, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--predict_batch_size", default=256,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--save-prediction", default="", type=str)

    parser.add_argument("--sp-pred", action="store_true", help="whether to predict sentence sp")
    return parser

def train_args():
    parser = common_args()
    # optimization
    parser.add_argument('--prefix', type=str, default="eval")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--output_dir", default="./logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=128,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--num_q_per_gpu", default=1)        
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=3,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval-period', type=int, default=2500)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--neg-num", type=int, default=9, help="how many neg/distant passage chains to use")
    parser.add_argument("--shared-norm", action="store_true")
    parser.add_argument("--qa-drop", default=0, type=float)
    parser.add_argument("--rank-drop", default=0, type=float)
    parser.add_argument("--sp-drop", default=0, type=float)
    parser.add_argument("--final-metric", default="joint_f1")
    parser.add_argument("--use-adam", action="store_true", help="use adam or adamW")
    parser.add_argument("--warmup-ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--sp-weight", default=0, type=float, help="weight of the sp loss")
    return parser.parse_args()
