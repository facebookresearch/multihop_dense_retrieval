# Mhop-Pretrain

- [Mhop-Pretrain](#mhop-pretrain)
  - [close-book QA experiments](#close-book-qa-experiments)
    - [NQ](#nq)
    - [complexwebq](#complexwebq)
    - [hotpotqa](#hotpotqa)
  - [Single-hop Dense Retrieval](#single-hop-dense-retrieval)
  - [Multi-hop Dense Retrieval](#multi-hop-dense-retrieval)
    - [Hyperparameter search train using submitit](#hyperparameter-search-train-using-submitit)
    - [evaluation](#evaluation)

## close-book QA experiments

### NQ
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py \
    --do_train \
    --prefix t5-nq-adafactor-scratch-continue \
    --model_name t5-large \
    --train_batch_size 2000 \
    --predict_batch_size 256 \
    --gradient_accumulation_steps 10 \
    --accumulate_gradients 10 \
    --learning_rate 1e-3 \
    --train_file /private/home/xwhan/data/nq-close/nq-combined.txt \
    --predict_file /private/home/xwhan/data/nq-close/nq-test.txt \
    --max_q_length 40 \
    --max_ans_length 30 \
    --use-adafactor \
    --seed 16 \
    --eval_period -1 \
    --init_checkpoint logs/t5-nq-adafactor-scratch-seed3-bsz2000-fp16False-lr0.001-decay0.0-t5-large/checkpoint_best.pt
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py \
    --do_predict \
    --model_name t5-large \
    --predict_batch_size 256 \
    --predict_file /private/home/xwhan/data/nq-close/nq-test.txt \
    --max_q_length 200 \
    --max_ans_length 30 \
    --num_beams 1 \
    --init_checkpoint logs/t5-nq-adafactor-scratch-seed3-bsz1000-fp16False-lr0.001-decay0.0-t5-large/checkpoint_best.pt
```


### complexwebq
```
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
    --seed 4 \
    --complex
```

### hotpotqa
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python t5_close_qa.py \
    --do_train \
    --prefix t5-hotpot-adafactor-scratch \
    --model_name t5-large \
    --train_batch_size 1000 \
    --predict_batch_size 256 \
    --gradient_accumulation_steps 10 \
    --accumulate_gradients 10 \
    --learning_rate 1e-3 \
    --train_file /private/home/xwhan/data/hotpot/hotpot_qas_train.json \
    --predict_file /private/home/xwhan/data/hotpot/hotpot_qas_val.json \
    --max_q_length 100 \
    --max_ans_length 40 \
    --use-adafactor \
    --seed 3 \
    --eval_period -1
```

## Single-hop Dense Retrieval
Training (not necessarily the best hyperparameters)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_single.py \
    --do_train \
    --prefix nq_dpr_baseline \
    --predict_batch_size 5000 \
    --model_name {roberta-base|bert-base-uncased} \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --accumulate_gradients 1 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /private/home/xwhan/data/nq-dpr/nq-with-neg-train.txt \
    --predict_file /private/home/xwhan/data/nq-dpr/nq-with-neg-dev.txt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 100 \
    --warmup-ratio 0.1 \
    --shared-encoder
```
Encoding the corpus, it will automatically output an id2doc mapping under index/, i.e., index/psgs_w100_id2doc.json. This will be used for retrieval evaluation
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py \
    --do_predict \
    --predict_batch_size 2000 \
    --model_name roberta-base \
    --predict_file /private/home/xwhan/code/DPR/data/wikipedia_split/psgs_w100.tsv \ # corpus path
    --init_checkpoint {CHECKPOINT PATH} \ # trained checkpoint, check logs/ under the retrieval/ folder
    --embed_save_path index/wiki_dpr.npy \ # encoded vectors
    --fp16 \
    --max_c_len 300 \
    --num_workers 20
```
Evaluating retrieval with FAISS
```
python eval_retrieval.py \
  /private/home/xwhan/data/nq-dpr/nq-test-qas.txt merged_all.npy \
  index/merged_all_id2doc.json \ # the id2doc mapping for last step
  logs/07-24-2020/unified_continue-seed16-bsz150-fp16True-lr1e-05-decay0.0/checkpoint_best.pt \ # retriever checkpoint
  --batch-size 1000 \
  --shared-encoder \
  --model-name \
  bert-base-uncased
```


## Multi-hop Dense Retrieval
### Hyperparameter search train using submitit
```
MKL_THREADING_LAYER=GNU  python submitit_train.py --prefix mhop_retriever_roberta_base \
    --train_file /private/home/xwhan/data/hotpot/hotpot_train_with_neg_sent_ann.json \
    --predict_file /private/home/xwhan/data/hotpot/hotpot_val_with_neg_sent_ann.json  \
    --model_name roberta-base
```

### evaluation
1. Use `get_para_embed.sh` to encode the corpus
```
./get_para_embed.sh logs/07-13-2020/mhop_retriever_no_sent_level_adam-seed16-bsz128-fp16True-lr1e-05-decay0.0/checkpoint_best.pt mhop_reencode
```
2. Retrieval evaluation with faiss
```
python eval_mhop_retrieval.py /private/home/xwhan/data/hotpot/hotpot_qas_val.json index/mhop_reencode.npy index/id2doc.json logs/07-13-2020/mhop_retriever_no_sent_level_adam-seed16-bsz128-fp16True-lr1e-05-decay0.0/checkpoint_best.pt --batch-size 1000 --beam-size-1 1 --topk 1 --beam-size-2 1 --gpu
```