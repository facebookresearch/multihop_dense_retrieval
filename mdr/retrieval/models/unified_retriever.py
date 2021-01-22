# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
from transformers import AutoModel
import torch.nn as nn
import torch

class UnifiedRetriever(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()
        self.encoder_c = AutoModel.from_pretrained(args.model_name)
        if "roberta" in args.model_name:
            self.roberta = True
            self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        else:
            self.roberta = False
        self.stop = nn.Linear(config.hidden_size, 2)
        self.stop_drop = nn.Dropout(args.stop_drop)

    def encode_seq(self, input_ids, mask, type_ids):
        if self.roberta:
            cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
            vector = self.project(cls_rep)
        else:
            vector = self.encoder_c(input_ids, mask, type_ids)[0][:, 0, :]
        return vector

    def forward(self, batch):
        c1 = self.encode_seq(batch['c1_input_ids'], batch['c1_mask'], batch.get('c1_type_ids', None))
        c2 = self.encode_seq(batch['c2_input_ids'], batch['c2_mask'], batch.get('c2_type_ids', None))
        neg_1 = self.encode_seq(batch['neg1_input_ids'], batch['neg1_mask'], batch.get('neg1_type_ids', None))
        neg_2 = self.encode_seq(batch['neg2_input_ids'], batch['neg2_mask'], batch.get('neg2_type_ids', None))

        q = self.encode_seq(batch['q_input_ids'], batch['q_mask'], batch.get('q_type_ids', None))
        q_sp1 = self.encode_seq(batch['q_sp_input_ids'], batch['q_sp_mask'], batch.get('q_sp_type_ids', None))

        qsp_pooled = self.encoder_c(batch['q_sp_input_ids'], batch['q_sp_mask'], batch.get('q_sp_type_ids', None))[1]
        stop_logits = self.stop(self.stop_drop(qsp_pooled))

        return {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1, "stop_logits": stop_logits}

    def encode_qsp(self, input_ids, q_mask, q_type_ids):
        sequence_output, pooled = self.encoder_c(input_ids, q_mask, q_type_ids)[:2]
        qsp_vector = sequence_output[:,0,:]
        stop_logits = self.stop(pooled)
        return qsp_vector, stop_logits

    def encode_q(self, input_ids, q_mask, q_type_ids):
        return self.encode_seq(input_ids, q_mask, q_type_ids)



class RobertaNQRetriever(nn.Module):

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
        return cls_rep

    def forward(self, batch):
        c = self.encode_seq(batch['c_input_ids'], batch['c_mask'])
        neg = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'])
        q = self.encode_seq(batch['q_input_ids'], batch['q_mask'])
        q_neg1 = self.encode_seq(batch['q_neg1_input_ids'], batch['q_neg1_mask'])
        vectors = {'q': q, 'c': c, "neg": neg, "q_neg1": q_neg1}
        return vectors

    def encode_q(self, input_ids, q_mask, q_type_ids):
        return self.encode_seq(input_ids, q_mask)

class BertNQRetriever(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.model_name)

    def encode_seq(self, input_ids, mask, type_ids):
        cls_rep = self.encoder(input_ids, mask, type_ids)[0][:, 0, :]
        return cls_rep

    def forward(self, batch):
        c = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch.get('c_type_ids', None))
        neg = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch.get('neg_type_ids', None))
        q = self.encode_seq(batch['q_input_ids'], batch['q_mask'], batch.get('q_type_ids', None))
        q_neg1 = self.encode_seq(batch['q_neg1_input_ids'], batch['q_neg1_mask'], batch.get('q_neg1_type_ids', None))
        neg_dense1 = self.encode_seq(batch['dense_neg1_input_ids'], batch['dense_neg1_mask'], batch.get('dense_neg1_type_ids', None))
        neg_dense2 = self.encode_seq(batch['dense_neg2_input_ids'], batch['dense_neg2_mask'], batch.get('dense_neg2_type_ids', None))
        vectors = {'q': q, 'c': c, "neg": neg, "q_neg1": q_neg1, "dense_neg1": neg_dense1, "dense_neg2": neg_dense2}
        return vectors

    def encode_q(self, input_ids, q_mask, q_type_ids):
        return self.encode_seq(input_ids, q_mask, q_type_ids)


class BertNQMomentumRetriever(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()

        self.encoder_q = BertNQRetriever(config, args)
        self.encoder_k = BertNQRetriever(config, args)

        if args.init_retriever != "":
            print(f"Load pretrained retriever from {args.init_retriever}")
            self.load_retriever(args.init_retriever)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.k = args.k
        self.m = args.m
        self.register_buffer("queue", torch.randn(self.k, config.hidden_size))
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

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = embeddings
        ptr = (ptr + batch_size) % self.k  # move pointer
        self.queue_ptr[0] = ptr
        return

    def forward(self, batch):
        q = self.encoder_q.encode_seq(batch['q_input_ids'], batch['q_mask'], batch.get('q_type_ids', None))
        q_neg1 = self.encoder_q.encode_seq(batch['q_neg1_input_ids'], batch['q_neg1_mask'], batch.get('q_neg1_type_ids', None))

        if self.training:
            with torch.no_grad():
                c = self.encoder_k.encode_seq(batch['c_input_ids'], batch['c_mask'], batch.get('c_type_ids', None))
                neg = self.encoder_k.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch.get('neg_type_ids', None))
        else:
            # whether to use the momentum encoder for inference
            c = self.encoder_k.encode_seq(batch['c_input_ids'], batch['c_mask'], batch.get('c_type_ids', None))
            neg = self.encoder_k.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch.get('neg_type_ids', None))
        
        vectors = {'q': q, 'c': c, "neg": neg, "q_neg1": q_neg1}
        return vectors

