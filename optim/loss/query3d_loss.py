from typing import Any
import torch
import torch.nn.functional as F

from optim.loss.loss import LOSS_REGISTRY

heads = ['ground', 'generation', 'query_cls', 'mv_cls', 'pc_cls', 'voxel_cls', 'txt_cls', 'sem_cls', 'prompt_cls', 'qa']

def cross_entropy(logits, label):
    """ calculate cross entropy along the last dim. """
    logits = torch.clamp(logits, min=-100)
    if label.shape == logits.shape: # label is a 0-1 vector and we use BCE loss.
        logits = logits.view(-1, logits.shape[-1])
        label = label.view(-1, label.shape[-1]).float()
        return F.binary_cross_entropy_with_logits(logits, label)
    else:
        logits = logits.view(-1, logits.shape[-1])
        label = label.view(-1)
        return F.cross_entropy(logits, label)

for head in heads:
    # 'head=head' is the magic to avoid the late-binding issue in lambda functions. Ask ChatGPT about late-binding to learn more.
    loss = lambda cfg, head=head: lambda data_dict: cross_entropy(data_dict[head + '_logits'], data_dict[head + '_label'])
    loss.__name__ = head + '_loss'
    LOSS_REGISTRY.register(loss)

    