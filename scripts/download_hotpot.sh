#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
#!/bin/bash 
# Make data and model folder. 
mkdir data
mkdir models

# Download data 
cd data
mkdir hotpot
cd hotpot
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_train_with_neg_v0.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_dev_with_neg_v0.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/hotpot_qas_val.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/train_retrieval_b100_k100_sp.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_b50_k50_sp.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_top100_sp.json

cd ..
mkdir hotpot_index
cd hotpot_index
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_id2doc.json
wget https://dl.fbaipublicfiles.com/mdpr/data/hotpot_index/wiki_index.npy

echo "Finished downloading data!"

# Download models
cd ../../models
wget https://dl.fbaipublicfiles.com/mdpr/models/doc_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/q_encoder.pt
wget https://dl.fbaipublicfiles.com/mdpr/models/qa_electra.pt
