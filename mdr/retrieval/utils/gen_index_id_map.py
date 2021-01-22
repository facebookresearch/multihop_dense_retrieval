# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import json

mapping = {}
with open('../data/para_doc.db') as f_in:
    for idx, line in enumerate(f_in):
        sample = json.loads(line.strip())
        mapping[idx] = sample['id']
with open('index_data/idx_id.json', 'w') as f_out:
    json.dump(mapping, f_out)

