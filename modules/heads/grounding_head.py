import torch
import torch.nn as nn

from modules.build import HEADS_REGISTRY
from modules.layers.transformers import TransformerEncoderLayer
from modules.utils import get_mlp_head, layer_repeat


@HEADS_REGISTRY.register()
class GroundHeadV1(nn.Module):
    def __init__(self, cfg, input_size=768, hidden_size=768, sem_cls_size=607, dropout=0.3, detach_all_aux_loss=False):
        super().__init__()
        self.og3d_head = get_mlp_head(
            input_size, hidden_size,
            1, dropout=dropout
        )
        self.txt_clf_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_pre_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.detach_all_aux_loss = detach_all_aux_loss

    def forward(self, txt_embeds, obj_embeds, obj_pre_embeds, obj_masks, **kwargs):
        og3d_logits = self.og3d_head(obj_embeds).squeeze(2)
        og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        if self.detach_all_aux_loss:
            txt_embeds = txt_embeds.detach()
            obj_embeds = obj_embeds.detach()
            obj_pre_embeds = obj_pre_embeds.detach()
        txt_cls_logits = self.txt_clf_head(txt_embeds[:, 0])
        obj_cls_logits = self.obj3d_clf_head(obj_embeds)
        obj_cls_pre_logits = self.obj3d_clf_pre_head(obj_pre_embeds)
        return txt_cls_logits, obj_cls_logits, obj_cls_pre_logits, og3d_logits


@HEADS_REGISTRY.register()
class GroundHead(nn.Module):
    def __init__(self, cfg, input_size=768, hidden_size=768, dropout=0.3):
        super().__init__()
        self.og3d_head = get_mlp_head(
            input_size, hidden_size,
            1, dropout=dropout
        )

    def forward(self, obj_embeds, obj_masks=None, **kwargs):
        og3d_logits = self.og3d_head(obj_embeds).squeeze(2)
        if obj_masks is not None:
            og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        return og3d_logits

@HEADS_REGISTRY.register()
class SequentialGroundHead(nn.Module):
    def __init__(self, cfg, hidden_size=4096, num_attention_heads=32, num_layers=2):
        super().__init__()        
        # grounding head
        self.og3d_head = get_mlp_head(
            hidden_size * 2, hidden_size // 2,
            1, dropout=0.1
        )

    def forward(self, obj_embeds, grd_embdes, obj_masks=None):
        txt_embeds = grd_embdes
        og3d_logits = self.og3d_head(torch.cat((obj_embeds, txt_embeds.repeat(1, obj_embeds.shape[1], 1)), dim=2)).squeeze(2)
        if obj_masks is not None:
            og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf')) 
        return og3d_logits

@HEADS_REGISTRY.register()
class SequentialGroundContrastiveHead(nn.Module):
    def __init__(self, cfg, hidden_size=4096, num_attention_heads=32, num_layers=2):
        super().__init__()        
        self.obj_mapping = get_mlp_head(hidden_size, 768, 768)
        self.grd_mapping = get_mlp_head(hidden_size, 768, 768)

    def forward(self, obj_embeds, grd_embdes, obj_masks=None):
        txt_embeds = self.grd_mapping(grd_embdes.repeat(1, obj_embeds.shape[1], 1))
        obj_embeds = self.obj_mapping(obj_embeds)
        og3d_logits = (txt_embeds * obj_embeds).sum(dim=2)
        if obj_masks is not None:
            og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf')) 
        return og3d_logits
        
@HEADS_REGISTRY.register()
class SequentialVisTAGroundHead(nn.Module):
    def __init__(self, cfg, hidden_size=4096, num_attention_heads=32, num_layers=2):
        super().__init__()
        # transformer encoder
        unified_encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=hidden_size)
        self.unified_encoder = layer_repeat(unified_encoder_layer, num_layers)

        # token embedding
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        # grounding head
        self.og3d_head = get_mlp_head(
            hidden_size, hidden_size // 2,
            1, dropout=0.1
        )

    def forward(self, obj_embeds, grd_embdes, obj_masks=None):
        txt_len = 1
        obj_len = obj_embeds.shape[1]
        txt_embeds = grd_embdes
        txt_masks = torch.ones(grd_embdes.shape[0], 1).bool().to(grd_embdes.device)
        
        for i, unified_layer in enumerate(self.unified_encoder):
            # add embeddings for points
            pc_token_type_ids = torch.ones((obj_embeds.shape[0:2])).long().to(obj_embeds.device)
            pc_type_embeds = self.token_type_embeddings(pc_token_type_ids)
            obj_embeds = obj_embeds + pc_type_embeds

            # add embeddings for languages
            lang_token_type_ids = torch.zeros((txt_embeds.shape[0:2])).long().to(txt_embeds.device)
            lang_type_embeds = self.token_type_embeddings(lang_token_type_ids)
            txt_embeds = txt_embeds + lang_type_embeds

            # fuse embeddings
            joint_embeds = torch.cat((txt_embeds, obj_embeds), dim=1)
            joint_masks = torch.cat((txt_masks, obj_masks), dim=1)

            # transformer
            joint_embeds, self_attn_matrices = unified_layer(joint_embeds,
                                                             tgt_key_padding_mask=joint_masks.logical_not())

            # split
            txt_embeds, obj_embeds = torch.split(joint_embeds, [txt_len, obj_len], dim=1)
        
        og3d_logits = self.og3d_head(obj_embeds).squeeze(2)
        if obj_masks is not None:
            og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf')) 
        return og3d_logits
        



