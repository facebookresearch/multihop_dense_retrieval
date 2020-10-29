#!/bin/bash



## use model trained on hotpotQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --predict_file /checkpoint/xwhan/cc_head_lm250_psg100/cc_head_lm250_psg100.csv \
    --init_checkpoint logs/08-08-2020/roberta_v0_fixed-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --embed_save_path /checkpoint/xwhan/cc_head_lm250_psg100/index/cc_head_lm250_psg100.npy \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20 



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 4000 \
    --model_name roberta-base \
    --predict_file /private/home/xwhan/data/hotpot/tfidf/abstracts.txt \
    --init_checkpoint logs/09-10-2020/roberta_no_linked_neg-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --embed_save_path index/abstracts_roberta_no_linked_neg.npy \
    --max_c_len 300 \
    --num_workers 30 \
    --fp16 


# encode fever corpus
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 5000 \
    --model_name roberta-base \
    --predict_file /private/home/xwhan/data/fever/tfidf/fever_corpus.txt \
    --init_checkpoint logs/08-27-2020/fever-seed16-bsz96-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue/checkpoint_best.pt \
    --embed_save_path index/fever.npy \
    --max_c_len 400 \
    --num_workers 30 \
    --fp16 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --predict_file /private/home/xwhan/code/DPR/data/wikipedia_split/psgs_w100.tsv \
    --init_checkpoint logs/08-23-2020/nq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt \
    --embed_save_path index/psg100_dpr_shared_baseline.npy \
    --max_c_len 300 \
    --num_workers 20 \
    --fp16 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --predict_file /private/home/xwhan/code/DPR/data/wikipedia_split/psgs_w100.tsv \
    --init_checkpoint logs/08-26-2020/wq_mhop_1_shared_dpr_neg_from_scratch-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt \
    --embed_save_path index/psg100_mhop_wq_1_from_baseline.npy \
    --max_c_len 300 \
    --num_workers 20 \
    --fp16 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --predict_file /private/home/xwhan/code/DPR/data/wikipedia_split/psgs_w100.tsv \
    --init_checkpoint logs/08-24-2020/wq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt \
    --embed_save_path index/psg100_wq_dpr_base_marker.npy \
    --max_c_len 300 \
    --num_workers 20 \
    --fp16 

# indexing with roberta-base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 2000 \
    --model_name roberta-base \
    --predict_file /private/home/xwhan/data/hotpot/tfidf/abstracts.txt \
    --init_checkpoint logs/08-08-2020/roberta_v0_fixed-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --embed_save_path index/abstracts_roberta.npy \
    --max_c_len 300 \
    --num_workers 20 \
    --fp16 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --predict_file /private/home/xwhan/data/combined/corpus/merged_no_period.txt \
    --init_checkpoint logs/08-12-2020/unified_roberta_no_period-seed16-bsz128-fp16True-lr2e-05-decay0.0-adamTrue/checkpoint_90.67.pt \
    --embed_save_path index/merged_no_period.npy \
    --max_c_len 300 \
    --num_workers 20 \
    --fp16 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --predict_file /private/home/xwhan/code/DPR/data/wikipedia_split/psgs_w100.tsv \
    --init_checkpoint logs/08-17-2020/nq_mhop_from_error_continue-seed42-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-roberta-base/checkpoint_best.pt \
    --embed_save_path index/nq_mhop_from_error.npy \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 2000 \
    --model_name bert-base-uncased \
    --predict_file /private/home/xwhan/code/DPR/data/wikipedia_split/psgs_w100.tsv \
    --init_checkpoint $1 \
    --embed_save_path index/${2}.npy \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20

# encode merged corpus
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 3000 \
    --model_name bert-base-uncased \
    --predict_file /private/home/xwhan/data/combined/corpus/merged_all.txt \
    --init_checkpoint logs/08-23-2020/nq_dpr_shared-seed16-bsz256-fp16True-lr2e-05-decay0.0-warm0.1-bert-base-uncased/checkpoint_best.pt \
    --embed_save_path index/merged_all_single_only.npy \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 5000 \
    --model_name bert-base-uncased \
    --predict_file /private/home/xwhan/data/combined/corpus/merged_all.txt \
    --init_checkpoint logs/09-15-2020/unified_bert_no_period-seed16-bsz128-fp16True-lr2e-05-decay0.0-adamTrue/checkpoint_best.pt \
    --embed_save_path index/merged_all_retrained_no_period.npy \
    --fp16 \
    --max_c_len 300 \
    --num_workers 40
    
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --predict_file /private/home/xwhan/data/combined/corpus/merged_all.txt \
    --init_checkpoint logs/08-08-2020/roberta_v0_fixed-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --embed_save_path index/merged_all_roberta.npy \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20


# encode corpus with sentence boundaries
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
#     --do_predict \
#     --predict_batch_size 2000 \
#     --bert_model_name bert-base-uncased \
#     --predict_file /private/home/xwhan/data/hotpot/tfidf/title_sents.txt \
#     --init_checkpoint $1 \
#     --embed_save_path index/${2}.npy \
#     --eval-workers 16 \
#     --fp16 \
#     --max_c_len 300 \
#     --eval-workers 5 \
#     --sent-level

