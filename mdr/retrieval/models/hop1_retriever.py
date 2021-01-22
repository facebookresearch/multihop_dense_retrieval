# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss


class Retriever1hop(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()

        self.bert_q = BertModel.from_pretrained(args.bert_model_name)
        self.bert_c = BertModel.from_pretrained(args.bert_model_name)
        self.hidden_size = config.hidden_size

    def forward(self, batch):
        # representations
        q_hidden_states = self.bert_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'])[0]
        q_cls = q_hidden_states[:,0,:]
        c_hidden_states = self.bert_c(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'])[0]
        c_cls = c_hidden_states[:, 0, :]
        neg_c_cls = self.bert_c(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'])[0][:, 0, :]

        # sentence-level representations
        gather_index = batch["c_sent_offsets"].unsqueeze(2).expand(-1,-1,self.hidden_size) # B x |S| x h
        c_sent_rep = torch.gather(c_hidden_states, 1, gather_index)

        outputs = {'q': q_cls, 'c':c_cls, "neg_c": neg_c_cls, "c_sent_rep": c_sent_rep}

        return outputs

