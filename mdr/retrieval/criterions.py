# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# def loss_single(model, batch, momentum=False):
#     outputs = model(batch)
#     q = outputs['q']
#     c = outputs['c']
#     neg_c = outputs['neg_c']
#     product_in_batch = torch.mm(q, c.t())
#     product_neg = (q * neg_c).sum(-1).unsqueeze(1)
#     product = torch.cat([product_in_batch, product_neg], dim=-1)

#     if momentum:
#         queue_c = model.module.encode_queue_ctx()
#         product_queue = torch.mm(q, queue_c.t())
#         product = torch.cat([product, product_queue], dim=-1)
#         model.module.dequeue_and_enqueue(batch)

#     target = torch.arange(product.size(0)).to(product.device)
#     loss = F.cross_entropy(product, target)
#     return loss


# """
# multi-hop retrieval for NQ, train the model to recover from
# """
# def loss_nq_mhop(model, batch, momentum=False):
#     outputs = model(batch)
#     product_in_batch = torch.mm(outputs['q'], outputs['c'].t())
#     product_neg = (outputs['q'] * outputs['neg']).sum(-1).unsqueeze(1)
#     # product_neg1 = (outputs['q'] * outputs['dense_neg1']).sum(-1).unsqueeze(1)
#     # product_neg2 = (outputs['q'] * outputs['dense_neg2']).sum(-1).unsqueeze(1)
#     scores1 = torch.cat([product_in_batch, product_neg], dim=-1)

#     product_in_batch_from_error = torch.mm(outputs["q_neg1"], outputs['c'].t())
#     dense_neg = torch.cat([outputs["dense_neg1"].unsqueeze(1), outputs["dense_neg2"].unsqueeze(1)], dim=1)
#     product_neg_from_error = torch.bmm(outputs["q_neg1"].unsqueeze(1), dense_neg.transpose(1,2)).squeeze(1)
#     scores2 = torch.cat([product_in_batch_from_error, product_neg_from_error], dim=-1)
#     if momentum:
#         queue_neg_scores_1 = torch.mm(outputs['q'], model.module.queue.clone().detach().t())
#         queue_neg_scores_2 = torch.mm(outputs["q_neg1"], model.module.queue.clone().detach().t())
#         scores1 = torch.cat([scores1, queue_neg_scores_1], dim=1)
#         scores2 = torch.cat([scores2, queue_neg_scores_2], dim=1)
#         model.module.dequeue_and_enqueue(outputs["c"].detach())
#         # model.module.momentum_update_key_encoder()

#     target = torch.arange(scores1.size(0)).to(scores1.device)
#     loss = F.cross_entropy(scores1, target) + F.cross_entropy(scores2, target)
#     # loss = F.cross_entropy(scores1, target)
#     return loss

# def eval_nq_mhop(model, batch):
#     outputs = model(batch)
#     product_in_batch = torch.mm(outputs['q'], outputs['c'].t())
#     product_neg = (outputs['q'] * outputs['neg']).sum(-1).unsqueeze(1)
#     # product_neg1 = (outputs['q'] * outputs['dense_neg1']).sum(-1).unsqueeze(1)
#     # product_neg2 = (outputs['q'] * outputs['dense_neg2']).sum(-1).unsqueeze(1)
#     scores1 = torch.cat([product_in_batch, product_neg], dim=-1)

#     product_in_batch_from_error = torch.mm(outputs["q_neg1"], outputs['c'].t())
#     dense_neg = torch.cat([outputs["dense_neg1"].unsqueeze(1), outputs["dense_neg2"].unsqueeze(1)], dim=1)
#     product_neg_from_error = torch.bmm(outputs["q_neg1"].unsqueeze(1), dense_neg.transpose(1,2)).squeeze(1)
#     scores2 = torch.cat([product_in_batch_from_error, product_neg_from_error], dim=-1)

#     target = torch.arange(scores1.size(0)).to(scores1.device)

#     rrs, rrs_2hop = [], []
#     ranked = scores1.argsort(dim=1, descending=True)
#     ranked_2hop = scores2.argsort(dim=1, descending=True)
#     idx2rank = ranked.argsort(dim=1)
#     for idx, t in enumerate(target.tolist()):
#         rrs.append(1 / (idx2rank[idx][t].item() +1))
#     idx2rank2hop = ranked_2hop.argsort(dim=1)
#     for idx, t in enumerate(target.tolist()):
#         rrs_2hop.append(1 / (idx2rank2hop[idx][t].item() +1))
#     return rrs, rrs_2hop



# def eval_vanilla(outputs):
#     """
#     view the two sp passages as the same, no multi-hop modeling;
#     select the passages from all passages in the batch
#     """
#     rrs = []
#     q = outputs['q']
#     c1 = outputs['c1'] 
#     c2 = outputs['c2']
#     c = torch.cat([c1.unsqueeze(1), c2.unsqueeze(1)], dim=1) # B x 2 x D
#     c = c.view(-1, q.size(-1)) # 2B x D
#     product_in_batch = torch.mm(q, c.t()) # Bx2B
#     neg_c = outputs['neg_c']
#     product_neg = (q * neg_c).sum(-1).unsqueeze(1)
#     product = torch.cat([product_in_batch, product_neg], dim=-1) 
#     target = torch.arange(product.size(0)).to(product.device).unsqueeze(1)
#     target = torch.cat([target*2, target*2+1], dim=1)
#     ranked = product.argsort(dim=1, descending=True)
#     # MRR
#     idx2rank = ranked.argsort(dim=1)
#     for idx, t in enumerate(target):
#         correct_idx = t.tolist()
#         for _ in correct_idx:
#             rrs.append(1 / (idx2rank[idx][_].item() + 1))
#     return rrs



def mhop_loss(model, batch, args):

    outputs = model(batch)
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1) # B x 2 x M x h
    
    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)

    # mask the 1st hop
    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)

    if args.momentum:
        queue_neg_scores_1 = torch.mm(outputs["q"], model.module.queue.clone().detach().t())
        queue_neg_scores_2 = torch.mm(outputs["q_sp1"], model.module.queue.clone().detach().t())

        # queue_neg_scores_1 = queue_neg_scores_1 / args.temperature
        # queue_neg_scores_2 = queue_neg_scores_2 / args.temperature  

        scores_1_hop = torch.cat([scores_1_hop, queue_neg_scores_1], dim=1)
        scores_2_hop = torch.cat([scores_2_hop, queue_neg_scores_2], dim=1)
        model.module.dequeue_and_enqueue(all_ctx.detach())
        # model.module.momentum_update_key_encoder()

    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)

    retrieve_loss = loss_fct(scores_1_hop, target_1_hop) + loss_fct(scores_2_hop, target_2_hop)

    return retrieve_loss

def mhop_eval(outputs, args):
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)


    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)


    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)
    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)

    ranked_1_hop = scores_1_hop.argsort(dim=1, descending=True)
    ranked_2_hop = scores_2_hop.argsort(dim=1, descending=True)
    idx2ranked_1 = ranked_1_hop.argsort(dim=1)
    idx2ranked_2 = ranked_2_hop.argsort(dim=1)
    rrs_1, rrs_2 = [], []
    for t, idx2ranked in zip(target_1_hop, idx2ranked_1):
        rrs_1.append(1 / (idx2ranked[t].item() + 1))
    for t, idx2ranked in zip(target_2_hop, idx2ranked_2):
        rrs_2.append(1 / (idx2ranked[t].item() + 1))
    
    return {"rrs_1": rrs_1, "rrs_2": rrs_2}


def unified_loss(model, batch, args):

    outputs = model(batch)
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)
    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)

    # mask for 1st hop
    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)

    stop_loss = F.cross_entropy(outputs["stop_logits"], batch["stop_targets"].view(-1), reduction="sum")

    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)


    retrieve_loss = F.cross_entropy(scores_1_hop, target_1_hop, reduction="sum") + (F.cross_entropy(scores_2_hop, target_2_hop, reduction="none") * batch["stop_targets"].view(-1)).sum()

    return retrieve_loss + stop_loss

def unified_eval(outputs, batch):
    all_ctx = torch.cat([outputs['c1'], outputs['c2']], dim=0)
    neg_ctx = torch.cat([outputs["neg_1"].unsqueeze(1), outputs["neg_2"].unsqueeze(1)], dim=1)
    scores_1_hop = torch.mm(outputs["q"], all_ctx.t())
    scores_2_hop = torch.mm(outputs["q_sp1"], all_ctx.t())
    neg_scores_1 = torch.bmm(outputs["q"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    neg_scores_2 = torch.bmm(outputs["q_sp1"].unsqueeze(1), neg_ctx.transpose(1,2)).squeeze(1)
    bsize = outputs["q"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(outputs["q"].device)
    scores_1_hop = scores_1_hop.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1_hop)
    scores_1_hop = torch.cat([scores_1_hop, neg_scores_1], dim=1)
    scores_2_hop = torch.cat([scores_2_hop, neg_scores_2], dim=1)
    target_1_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device)
    target_2_hop = torch.arange(outputs["q"].size(0)).to(outputs["q"].device) + outputs["q"].size(0)

    # stop accuracy
    stop_pred = outputs["stop_logits"].argmax(dim=1)
    stop_targets = batch["stop_targets"].view(-1)
    stop_acc = (stop_pred == stop_targets).float().tolist()

    ranked_1_hop = scores_1_hop.argsort(dim=1, descending=True)
    ranked_2_hop = scores_2_hop.argsort(dim=1, descending=True)
    idx2ranked_1 = ranked_1_hop.argsort(dim=1)
    idx2ranked_2 = ranked_2_hop.argsort(dim=1)

    rrs_1_mhop, rrs_2_mhop, rrs_nq = [], [], []
    for t1, idx2ranked1, t2, idx2ranked2, stop in zip(target_1_hop, idx2ranked_1, target_2_hop, idx2ranked_2, stop_targets):
        if stop: # 
            rrs_1_mhop.append(1 / (idx2ranked1[t1].item() + 1))
            rrs_2_mhop.append(1 / (idx2ranked2[t2].item() + 1))
        else:
            rrs_nq.append(1 / (idx2ranked1[t1].item() + 1))

    return {
        "stop_acc": stop_acc, 
        "rrs_1_mhop": rrs_1_mhop,
        "rrs_2_mhop": rrs_2_mhop,
        "rrs_nq": rrs_nq
        }
