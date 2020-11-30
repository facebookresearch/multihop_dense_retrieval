

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
    --do_train \
    --prefix qa_wwm_bert_title_mark_eval_debug \
    --predict_batch_size 512 \
    --model_name bert-large-uncased-whole-word-masking \
    --train_batch_size 80 \
    --learning_rate 3e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/hotpot/dense_train_b10_top20_outputs.json \
    --predict_file /private/home/xwhan/data/hotpot/dense_val_outputs.json \
    --seed 3 \
    --eval-period 10 \
    --max_seq_len 512 \
    --max_q_len 100 \
    --gradient_accumulation_steps 8 \
    --neg-num 4


# spanbert debug, fp16 does not work
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
    --do_train \
    --prefix ranked_spanbert_debug \
    --predict_batch_size 1024 \
    --model_name spanbert \
    --train_batch_size 48 \
    --learning_rate 3e-5 \
    --train_file /private/home/xwhan/data/hotpot/dense_train_b10_top20_outputs_sents.json \
    --predict_file /private/home/xwhan/data/hotpot/dense_val_outputs_sents.json \
    --seed 3 \
    --eval-period 500 \
    --max_seq_len 512 \
    --max_q_len 64 \
    --gradient_accumulation_steps 8 \
    --neg-num 5 \
    --use-adam

# test electra
CUDA_VISIBLE_DEVICES=0 python train_qa.py \
    --do_train \
    --prefix electra_large_debug_sn \
    --predict_batch_size 1024 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 12 \
    --learning_rate 5e-5 \
    --train_file /private/home/xwhan/data/hotpot/dense_train_b100_k100_sents.json \
    --predict_file /private/home/xwhan/data/hotpot/dense_val_b30_k30_roberta_sents.json \
    --seed 42 \
    --eval-period 250 \
    --max_seq_len 512 \
    --max_q_len 64 \
    --gradient_accumulation_steps 8 \
    --neg-num 11 \
    --fp16 \
    --use-adam \
    --warmup-ratio 0.1 \
    --sp-weight 0.05 \
    --sp-pred \
    --shared-norm


# QA evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
    --do_predict \
    --predict_batch_size 2000 \
    --model_name google/electra-large-discriminator \
    --fp16 \
    --predict_file /private/home/xwhan/data/hotpot/dense_val_b100_k100_roberta_best_sents.json \
    --max_seq_len 512 \
    --max_q_len 64 \
    --init_checkpoint qa/logs/08-10-2020/electra_val_top30-epoch7-lr5e-05-seed42-rdrop0-qadrop0-decay0-qpergpu2-aggstep8-clip2-evalper250-evalbsize1024-negnum5-warmup0.1-adamTrue-spweight0.025/checkpoint_best.pt \
    --sp-pred \
    --max_ans_len 30 \
    --save-prediction hotpot_val_top100.json

# QA evaluation with wwm

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
    --do_predict \
    --predict_batch_size 1024 \
    --model_name bert-large-uncased-whole-word-masking \
    --fp16 \
    --predict_file /private/home/xwhan/data/hotpot/dense_hotpot_val_b250_k250_roberta_best_sents.json \
    --max_seq_len 512 \
    --max_q_len 64 \
    --init_checkpoint qa/logs/08-17-2020/wwm_val_top50-epoch7-lr5e-05-seed42-rdrop0-qadrop0-decay0-qpergpu2-aggstep8-clip2-evalper250-evalbsize1024-negnum5-warmup0.2-adamTrue-spweight0.025-snFalse/checkpoint_best.pt \
    --sp-pred \
    --max_ans_len 30 \
    --save-prediction hotpot_val_wwm_top250.json


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_qa.py \
    --do_predict \
    --predict_batch_size 1024 \
    --model_name google/electra-large-discriminator \
    --fp16 \
    --predict_file /private/home/xwhan/data/hotpot/dense_val_b50_k50_roberta_best_sents.json \
    --max_seq_len 512 \
    --max_q_len 64 \
    --init_checkpoint qa/logs/08-10-2020/electra_val_top30-epoch7-lr5e-05-seed42-rdrop0-qadrop0-decay0-qpergpu2-aggstep8-clip2-evalper250-evalbsize1024-negnum5-warmup0.1-adamTrue-spweight0.025/checkpoint_best.pt \
    --sp-pred \
    --max_ans_len 30 \
    --save-prediction hotpot_val_b5_k5.json \

srun --gres=gpu:8 --partition learnfair --time=48:00:00 --mem 500G --constraint volta32gb --cpus-per-task 80 --pty /bin/bash -l
