# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from torch import embedding
from transformers import AutoModel
import torch.nn as nn
import torch


class RobertaRetriever(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)
        return vector

    def forward(self, batch):
        c1 = self.encode_seq(batch['c1_input_ids'], batch['c1_mask'])
        c2 = self.encode_seq(batch['c2_input_ids'], batch['c2_mask'])

        neg_1 = self.encode_seq(batch['neg1_input_ids'], batch['neg1_mask'])
        neg_2 = self.encode_seq(batch['neg2_input_ids'], batch['neg2_mask'])

        q = self.encode_seq(batch['q_input_ids'], batch['q_mask'])
        q_sp1 = self.encode_seq(batch['q_sp_input_ids'], batch['q_sp_mask'])
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors

    def encode_q(self, input_ids, q_mask, q_type_ids):
        return self.encode_seq(input_ids, q_mask)



class RobertaMomentumRetriever(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()

        self.encoder_q = RobertaRetriever(config, args)
        self.encoder_k = RobertaRetriever(config, args)

        if args.init_retriever != "":
            print(f"Load pretrained retriever from {args.init_retriever}")
            self.load_retriever(args.init_retriever)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.k = args.k
        self.m = args.m
        self.register_buffer("queue", torch.randn(self.k, config.hidden_size))
        # add layernorm?
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def load_retriever(self, path):
        state_dict = torch.load(path)
        def filter(x): return x[7:] if x.startswith('module.') else x
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in self.encoder_q.state_dict()}
        self.encoder_q.load_state_dict(state_dict)
        return

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, embeddings):
        """
        memory bank of previous context embeddings, c1 and c2
        """
        # gather keys before updating queue
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.k:
            batch_size = self.k - ptr
            embeddings = embeddings[:batch_size]

        # if self.k % batch_size != 0:
        #     return
        # assert self.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = embeddings

        ptr = (ptr + batch_size) % self.k  # move pointer
        self.queue_ptr[0] = ptr
        return


    def forward(self, batch):
        q = self.encoder_q.encode_seq(batch['q_input_ids'], batch['q_mask'])
        q_sp1 = self.encoder_q.encode_seq(batch['q_sp_input_ids'], batch['q_sp_mask'])

        if self.training:
            with torch.no_grad():
                c1 = self.encoder_k.encode_seq(batch['c1_input_ids'], batch['c1_mask'])
                c2 = self.encoder_k.encode_seq(batch['c2_input_ids'], batch['c2_mask'])

                neg_1 = self.encoder_k.encode_seq(batch['neg1_input_ids'], batch['neg1_mask'])
                neg_2 = self.encoder_k.encode_seq(batch['neg2_input_ids'], batch['neg2_mask'])
        else:
            # whether to use the momentum encoder for inference
            c1 = self.encoder_k.encode_seq(batch['c1_input_ids'], batch['c1_mask'])
            c2 = self.encoder_k.encode_seq(batch['c2_input_ids'], batch['c2_mask'])

            neg_1 = self.encoder_k.encode_seq(batch['neg1_input_ids'], batch['neg1_mask'])
            neg_2 = self.encoder_k.encode_seq(batch['neg2_input_ids'], batch['neg2_mask'])
        
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors


