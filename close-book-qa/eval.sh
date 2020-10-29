CUDA_VISIBLE_DEVICES=0,1,2,3 python t5_close_qa.py \
    --do_predict \
    --prefix eval_ \
    --model_name facebook/bart-large \
    --predict_batch_size 256 \
    --learning_rate 1e-3 \
    --predict_file /private/home/xwhan/data/hotpot/hotpot_qas_val_bridge.json \
    --max_q_length 200 \
    --max_ans_length 50 \
    --use-adafactor \
    --seed 3 \
    --eval_period -1 \
    --init_checkpoint logs/bart-hotpot-adamw-decode-bridge-seed3-bsz1024-fp16True-lr3e-05-decay0.0-facebook/bart-large/checkpoint_best.pt \
    --decode-bridge


# complexwebq
CUDA_VISIBLE_DEVICES=0,1,2,3 python t5_close_qa.py \
    --do_predict \
    --prefix eval_ \
    --model_name facebook/bart-large \
    --predict_batch_size 256 \
    --predict_file /private/home/xwhan/data/ComplexWebQ/complexwebq_dev_qas.txt \
    --max_q_length 200 \
    --max_ans_length 50 \
    --use-adafactor \
    --seed 3 \
    --init_checkpoint logs/bart-complexwebq-adamw-seed3-bsz1024-fp16True-lr3e-05-decay0.0-facebook/bart-large/checkpoint_best.pt \
    --complex