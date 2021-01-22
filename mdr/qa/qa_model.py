# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

from transformers import AutoModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class QAModel(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()
        self.model_name = args.model_name
        self.sp_weight = args.sp_weight
        self.sp_pred = args.sp_pred
        self.encoder = AutoModel.from_pretrained(args.model_name)

        if "electra" in args.model_name:
            self.pooler = BertPooler(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.rank = nn.Linear(config.hidden_size, 1) # noan

        if self.sp_pred:
            self.sp = nn.Linear(config.hidden_size, 1)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, batch):

        outputs = self.encoder(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids', None))

        if "electra" in self.model_name:
            sequence_output = outputs[0]
            pooled_output = self.pooler(sequence_output)
        else:
            sequence_output, pooled_output = outputs[0], outputs[1]

        logits = self.qa_outputs(sequence_output)
        outs = [o.squeeze(-1) for o in logits.split(1, dim=-1)]
        outs = [o.float().masked_fill(batch["paragraph_mask"].ne(1), float("-inf")).type_as(o) for o in outs]

        start_logits, end_logits = outs[0], outs[1]
        rank_score = self.rank(pooled_output)

        if self.sp_pred:
            gather_index = batch["sent_offsets"].unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
            sent_marker_rep = torch.gather(sequence_output, 1, gather_index)
            sp_score = self.sp(sent_marker_rep).squeeze(2)
        else:
            sp_score = None

        if self.training:

            rank_target = batch["label"]
            if self.sp_pred:
                sp_loss = F.binary_cross_entropy_with_logits(sp_score, batch["sent_labels"].float(), reduction="none")
                sp_loss = (sp_loss * batch["sent_offsets"]) * batch["label"]
                sp_loss = sp_loss.sum()

            start_positions, end_positions = batch["starts"], batch["ends"]

            rank_loss = F.binary_cross_entropy_with_logits(rank_score, rank_target.float(), reduction="sum")

            start_losses = [self.loss_fct(start_logits, starts) for starts in torch.unbind(start_positions, dim=1)]
            end_losses = [self.loss_fct(end_logits, ends) for ends in torch.unbind(end_positions, dim=1)]
            loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
            log_prob = - loss_tensor
            log_prob = log_prob.float().masked_fill(log_prob == 0, float('-inf')).type_as(log_prob)
            marginal_probs = torch.sum(torch.exp(log_prob), dim=1)
            m_prob = [marginal_probs[idx] for idx in marginal_probs.nonzero()]
            if len(m_prob) == 0:
                span_loss = self.loss_fct(start_logits, start_logits.new_zeros(
                    start_logits.size(0)).long()-1).sum()
            else:
                span_loss = - torch.log(torch.cat(m_prob)).sum()

            if self.sp_pred:
                loss = rank_loss + span_loss + sp_loss * self.sp_weight
            else:
                loss = rank_loss + span_loss
            return loss.unsqueeze(0)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'rank_score': rank_score,
            "sp_score": sp_score
            }
