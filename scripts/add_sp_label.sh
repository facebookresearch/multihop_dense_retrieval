#!/bin/bash 

# to add SP sentence labels to the retrieved passages for dev data ./scripts/add_sp_label.sh data/hotpot/hotpot_dev_distractor_v1.json data/hotpot/dev_retrieved_top100.json data/hotpot/dev_retrieved_top100_with_sp.json

ORIGINAL_DATA=$1
RETRIEVED_DATA=$2
SAVED_PATH=$3

python mdr/retrieval/utils/mhop_utils.py ${ORIGINAL_DATA} ${RETRIEVED_DATA}
${SASAVED_PATH}
