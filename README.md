



# [<p align=center>Multi-Hop Dense Text Retrieval (`MDR`)</p>](#p-aligncentermulti-hop-dense-text-retrieval-mdrp)

`MDR` is a simple and generalized dense retrieval method which recursively retrieves supporting text passages for answering complex open-domain questions. The repo provides code and pretrained retrieval models that produce **state-of-the-art** retrieval performance on two multi-hop QA datasets (the [HotpotQA](https://hotpotqa.github.io) dataset and the multi-hop subset of the [FEVER fact extraction and verification dataset](https://fever.ai)). 

More details about our approach are described in the following paper [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://arxiv.org/abs/2009.12756)


- [Use the trained models](#use-the-trained-models)
    - [Evaluating retrieval](#evaluating-retrieval)
    - [Evaluating QA](#evaluating-qa)
- [Train models from scratch](#train-models-from-scratch)
    - [Retriever training](#retriever-training)
    - [Encoding the corpus for retrieval](#encoding-the-corpus-for-retrieval)
    - [ELECTRA QA model training](#electra-qa-model-training)

## Use the trained models

1. Set up the environment
```bash
conda create --name MDR python=3.6
conda activate MDR
git clone git@github.com:facebookresearch/multihop_dense_retrieval.git
cd multihop_dense_retrieval 
bash setup.sh
```

2. Download the necessary data files and pretrained retrieval models
Simplified data files with **quesitons** and ground-truth **supporting passages**:

```
# save pretrained models to models/ and all processed hotpotQA into data/ 
# models will take about 2GB, and data will take 20GB since the pre-trained wikipedia index are included.
bash ./scripts/download_hotpot.sh
```


### Evaluating retrieval
Evalauting direct retrieval performance (The printed statistics might not adhere to the metric names defined in the paper. **PR**: whether one of the supporting passages is included in all retrieved passages; **P-EM**: whether **both** supporting passages are included in all retrieval passages; **Path Recall**: whether any of the topk retrieved chain extract match the ground-truth supporting passages.) and saving topk retrieved passage chains for downstream QA. 

```
python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    models/q_encoder.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path ${SAVE_RETRIEVAL_FOR_QA}
```
Sevaral important options includes 
* `--beam-size-n`: beam size at each hop; 
* `--topk`: topk passage chains from beam search 
* `--gpu`: move the dense index to GPU, resulting in much faster search


### Evaluating QA
The best answer extraction model is based on the pretrained [ELECTRA](https://arxiv.org/abs/2003.10555), outperforming the **BERT-large-whole-word-masking** by ~2 points answer EM/F1. We construct the training data with the pretrained MDR retriever and always include the ground-truth passage chain if the MDR failed. Each training question is paired with the groundtruth SP passage chain and also 5 (hyperparameter) retrieved chains which do not match the groundtruth. 

As the HotpotQA task requires evaluating the prediction of supporting sentences, we do sentence segmetation on the MDR retrieval result before feeding into the answer extraction models. Follow the script [scripts/add_sp_label.sh](scripts/add_sp_label.sh) to annotate the retrieved chains for train/val data. Supposing we got the top100 retrieved results in `data/hotpot/dev_retrieval_top100_sp.json`: 

```
python scripts/train_qa.py \
    --do_predict \
    --predict_batch_size 200 \
    --model_name google/electra-large-discriminator \
    --fp16 \
    --predict_file data/hotpot/dev_retrieval_top100_sp.json \
    --max_seq_len 512 \
    --max_q_len 64 \
    --init_checkpoint models/qa_electra.pt \
    --sp-pred \
    --max_ans_len 30 \
    --save-prediction hotpot_val_top100.json
```


## Train models from scratch
Our experiments are mostly run on 8 GPUs, however, we observed similar performance when using a smaller performance. 

### Retriever training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train_mhop.py \
    --do_train \
    --prefix ${RUN_ID} \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --train_batch_size 150 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH}  \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --shared-encoder \
    --warmup-ratio 0.1
```
Processed train/validation data for retrieval training:
* `${TRAIN_DATA_PATH}`: data/hotpot/hotpot_train_with_neg_v0.json
* `${DEV_DATA_PATH}`: data/hotpot/hotpot_dev_with_neg_v0.json

### Finetune the question encoder with frozen memory bank
This step happens after the previous training stage and reuses the checkpoint
point.


```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_momentum.py \
    --do_train \
    --prefix {RUN_ID} \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --train_batch_size 150 \
    --learning_rate 1e-5 \
    --fp16 \
    --train_file {TRAIN_DATA_PATH} \
    --predict_file {DEV_DATA_PATH}  \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --momentum \
    --k 76800 \
    --m 0.999 \
    --temperature 1 \
    --init-retriever {CHECKPOINT_PT} 

## Encoding the corpus for retrieval
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/get_embed.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name roberta-base \
    --predict_file ${CORPUS_PATH} \
    --init_checkpoint ${MODEL_CHECKPOINT} \
    --embed_save_path ${SAVE_PATH} \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20 
```
* `${CORPUS_PATH}`: json encoded object ({"title": ..., "text": ...}) per line;
* `${SAVE_PATH}`: path to save the numpy vectors and ID2DOC lookup table.

### ELECTRA QA model training
The ELECTRA-based QA model is sensitive to the learning rate schedule and adding a 10% warmup stage seems necessary to achieve good answer extraction performance. 
```
CUDA_VISIBLE_DEVICES=0 python train_qa.py \
    --do_train \
    --prefix electra_large_debug_sn \
    --predict_batch_size 1024 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 12 \
    --learning_rate 5e-5 \
    --train_file ${QA_TRAIN_DATA} \
    --predict_file ${QA_DEV_DATA} \
    --seed 42 \
    --eval-period 250 \
    --max_seq_len 512 \
    --max_q_len 64 \
    --gradient_accumulation_steps 8 \
    --neg-num 5 \
    --fp16 \
    --use-adam \
    --warmup-ratio 0.1 \
    --sp-weight 0.05 \
    --sp-pred
```
Processed (ran [scripts/add_sp_label.sh](scripts/add_sp_label.sh)) train/validata data for QA training.
* `${QA_TRAIN_DATA}`: data/hotpot/train_retrieval_b100_k100_sp.json
* `${QA_DEV_DATA}`: data/hotpot/dev_retrieval_b30_k30_sp.json



