#!/bin/bash 
# Make data and model folder. 
mkdir data
mkdir models

# Download data 
cd data
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_train_with_neg_v0.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_dev_with_neg_v0.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_qas_val.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_id2doc.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_index.npy
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/train_retrieval_b100_k100_sp.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_b50_k50_sp.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_top100_sp.json
echo "Finished downloading data!"

# Download models
cd ../models
wget https://dl.fbaipublicfiles.com/mdpr/models/doc_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/q_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/qa_electra.pt
