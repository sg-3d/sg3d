from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from optim.loss.loss import LOSS_REGISTRY
from optim.loss.query3d_loss import cross_entropy

@LOSS_REGISTRY.register()
class LLMLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self, data_dict):
        llm_logits = data_dict['llm_logits']
        llm_labels = data_dict['llm_labels']
        num_tokens_for_loss = data_dict['num_tokens_for_loss']
        bs = data_dict['bs']
        loss = F.cross_entropy(llm_logits, llm_labels, reduction='none')
        loss = rearrange(loss, '(b t) -> b t', b=bs)
        loss = loss.sum(1) / num_tokens_for_loss   # (B,)
        loss = loss.mean()
        return loss

@LOSS_REGISTRY.register()
class SequentialGroundLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self, data_dict):
        ground_logits = data_dict['ground_logits']
        ground_label = data_dict['ground_label']
        logits = ground_logits.view(-1, ground_logits.shape[-1])
        label = ground_label.view(-1)
        loss = F.cross_entropy(logits, label)
        return loss
        
        