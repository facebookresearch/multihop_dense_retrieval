  
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py\
    --do_train \
    --prefix t5-nq-adafactor-scratch \
    --eval_period -1 \
    --model_name t5-large \
    --train_batch_size 2000 \
    --predict_batch_size 256 \
    --gradient_accumulation_steps 8 \
    --accumulate_gradients 8 \
    --learning_rate 1e-3 \
    --train_file /private/home/xwhan/data/nq-close/nq-train.txt \
    --predict_file /private/home/xwhan/data/nq-close/nq-dev.txt \
    --seed 3 \
    --use-adafactor
    # --use-adam \
    
    # \
    # --init_checkpoint /private/home/xwhan/code/close-book-qa/logs/t5-nq-debug-continue-nodecay-seed16-bsz8000-fp16False-lr0.0005-decay0.0-t5-large/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py\
    --do_train \
    --prefix bart-nq-adamw-scratch \
    --eval_period -1 \
    --fp16 \
    --model_name facebook/bart-large \
    --train_batch_size 1024 \
    --predict_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 3e-5 \
    --train_file /private/home/xwhan/data/nq-close/nq-train.txt \
    --predict_file /private/home/xwhan/data/nq-close/nq-dev.txt \
    --seed 3 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py \
    --do_train \
    --prefix t5-hotpot-adafactor-scratch \
    --model_name t5-large \
    --train_batch_size 1000 \
    --predict_batch_size 256 \
    --gradient_accumulation_steps 10 \
    --accumulate_gradients 10 \
    --learning_rate 1e-3 \
    --train_file /private/home/xwhan/data/hotpot/hotpot_qas_train_bridge.json \
    --predict_file /private/home/xwhan/data/hotpot/hotpot_qas_val_bridge.json \
    --max_q_length 200 \
    --max_ans_length 50 \
    --use-adafactor \
    --seed 3 \
    --eval_period -1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py \
    --do_train \
    --prefix bart-hotpot-adamw-decode-bridge \
    --model_name facebook/bart-large \
    --train_batch_size 1024 \
    --predict_batch_size 200 \
    --gradient_accumulation_steps 2 \
    --accumulate_gradients 2 \
    --learning_rate 3e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/hotpot/hotpot_qas_train_bridge.json \
    --predict_file /private/home/xwhan/data/hotpot/hotpot_qas_val_bridge.json \
    --max_q_length 200 \
    --max_ans_length 50 \
    --seed 3 \
    --eval_period -1 \
    --decode-bridge


srun --gres=gpu:8 --partition dev --time=24:00:00 --constraint volta32gb --cpus-per-task 80 --pty /bin/bash -l

srun --partition learnfair --time=12:00:00 --constraint volta32gb --cpus-per-task 80 --pty /bin/bash -l


# complexwebquestions
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py\
    --do_train \
    --prefix bart-complexwebq-adamw \
    --eval_period -1 \
    --fp16 \
    --model_name facebook/bart-large \
    --train_batch_size 1024 \
    --predict_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --accumulate_gradients 2 \
    --learning_rate 3e-5 \
    --train_file /private/home/xwhan/data/ComplexWebQ/complexwebq_train_qas.txt \
    --predict_file /private/home/xwhan/data/ComplexWebQ/complexwebq_dev_qas.txt \
    --max_q_length 100 \
    --max_ans_length 50 \
    --seed 3 \
    --complex


# eval

python t5_close_qa.py\
    --do_predict \
    --prefix t5-nq-debug-continue \
    --eval_period -1 \
    --model_name t5-large \
    --predict_batch_size 64 \
    --train_file /private/home/xwhan/data/nq-close/nq-train.txt \
    --predict_file /private/home/xwhan/data/nq-close/nq-dev.txt \
    --init_checkpoint /private/home/xwhan/code/close-book-qa/logs/t5-nq-debug-seed3-bsz400-fp16False-lr1e-05-t5-large/checkpoint_best.pt \
    --num_beams 4

