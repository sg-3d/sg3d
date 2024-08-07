

from modules.build import VISION_REGISTRY
import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from hydra.utils import instantiate
from accelerate.logging import get_logger
import os
import timm
from einops import rearrange
import math
from typing import Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from modules.layers.pointnet import PointNetPP
from modules.layers.transformers import CrossAttentionLayer, TransformerEncoderLayer, TransformerSpatialEncoderLayer
from modules.utils import calc_pairwise_locs, get_activation_fn, layer_repeat
from modules.weights import _init_weights_bert

logger = get_logger(__name__)

def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore
    """
    return self

class PointcloudBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.pcd_net = PointNetPP(
                sa_n_points=[32, 16, None],
                sa_n_samples=[32, 32, None],
                sa_radii=[0.2, 0.4, None],
                sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
            )
        self.backbone_name = cfg.net._target_.split('.')[-1]
        self.out_dim = self.pcd_net.out_dim
        logger.info(f"Build PointcloudBackbone: {self.backbone_name}")

        path = cfg.path
        if path is not None and os.path.exists(path):
            self.pcd_net.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
            logger.info(f"Load {self.backbone_name} weights from {path}")

        self.freeze = cfg.freeze
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
            self.train = disabled_train
            logger.info(f"Freeze {self.backbone_name}")

    def forward_normal(self, obj_pcds):
        # obj_pcds: (batch_size, num_objs, num_points, 6)
        batch_size = obj_pcds.shape[0]
        obj_embeds = self.pcd_net(
            rearrange(obj_pcds, 'b o p d -> (b o) p d')
        )
        obj_embeds = rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        return obj_embeds

    @torch.no_grad()
    def forward_frozen(self, obj_pcds):
        return self.forward_normal(obj_pcds)

    def forward(self, obj_pcds):
        if self.freeze:
            return self.forward_frozen(obj_pcds)
        else:
            return self.forward_normal(obj_pcds)

def generate_fourier_features(pos, num_bands=10, max_freq=15, concat_pos=True, sine_only=False):
    # Input: B, N, C
    # Output: B, N, C'
    batch_size = pos.shape[0]
    device = pos.device

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.linspace(start=min_freq, end=max_freq, steps=num_bands, device=device)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos.unsqueeze(-1).repeat(1, 1, 1, num_bands) * freq_bands
    per_pos_features = torch.reshape(
        per_pos_features, [batch_size, -1, np.prod(per_pos_features.shape[2:])])
    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 k_dim=None, v_dim=None, prenorm=True):
        super().__init__()
        if k_dim is None:
            k_dim = d_model
        if v_dim is None:
            v_dim = d_model
        self.prenorm = prenorm
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, kdim=k_dim, vdim=v_dim
        )
        # Implementation of Feedforward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(
            self, tgt, memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        if self.prenorm:
            tgt2 = self.norm1(tgt2)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        if not self.prenorm:
            tgt = self.norm1(tgt)
        if self.prenorm:
            tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if not self.prenorm:
            tgt = self.norm3(tgt)
        return tgt, cross_attn_matrices


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(
            self, tgt, memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            query=tgt2, key=tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, batch_first=True, dropout=0.1, activation="relu", prenorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.prenorm = prenorm

    def forward(
            self, tgt, tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        if self.prenorm:
            tgt2 = self.norm1(tgt2)
        tgt2, self_attn_matrices = self.self_attn(
            query=tgt2, key=tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        if not self.prenorm:
            tgt = self.norm1(tgt)
        if self.prenorm:
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        if not self.prenorm:
            tgt = self.norm2(tgt)
        return tgt, self_attn_matrices


class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
            self, d_model, n_head, dropout=0.1, spatial_multihead=True, spatial_dim=5,
            spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' % (d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == 'ctx':
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == 'cond':
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        residual = q
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t')
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)
        elif self.spatial_attn_fusion == 'ctx':
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])
        elif self.spatial_attn_fusion == 'cond':
            spatial_weights = self.lang_cond_fc(residual)
            spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head,
                                               d=self.spatial_dim + 1)
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias
            loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        if self.spatial_attn_fusion == 'add':
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        else:
            if self.spatial_attn_fusion in ['mul', 'cond']:
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                fused_attn = loc_attn + attn
            fused_attn = torch.softmax(fused_attn, 3)

        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, fused_attn


class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
            spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
            self, tgt, memory,
            tgt_pairwise_locs: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


class TransformerSpatialEncoderLayer(TransformerEncoderLayer):
    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
            spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
            self, tgt, tgt_pairwise_locs,
            tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, self_attn_matrices
    
@VISION_REGISTRY.register()
class OSE3D(nn.Module):
    # Open-vocabulary, Spatial-attention, Embodied-token, 3D-agent
    def __init__(self, cfg):
        super().__init__()
        self.use_spatial_attn = cfg.use_spatial_attn   # spatial attention
        self.use_embodied_token = cfg.use_embodied_token   # embodied token
        hidden_dim = cfg.hidden_dim

        # pcd backbone
        self.obj_encoder = PointcloudBackbone(cfg.backbone)
        self.obj_proj = nn.Linear(self.obj_encoder.out_dim, hidden_dim)

        # embodied token
        if self.use_embodied_token:
            self.anchor_feat = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.anchor_size = nn.Parameter(torch.ones(1, 1, 3))
        self.orient_encoder = nn.Linear(cfg.fourier_size, hidden_dim)
        self.obj_type_embed = nn.Embedding(2, hidden_dim)

        # spatial encoder
        if self.use_spatial_attn:
            spatial_encoder_layer = TransformerSpatialEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
                spatial_dim=cfg.spatial_encoder.spatial_dim,
                spatial_multihead=cfg.spatial_encoder.spatial_multihead,
                spatial_attn_fusion=cfg.spatial_encoder.spatial_attn_fusion,
            )
        else:
            spatial_encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
            )

        self.spatial_encoder = layer_repeat(
            spatial_encoder_layer,
            cfg.spatial_encoder.num_layers,
        )
        self.pairwise_rel_type = cfg.spatial_encoder.pairwise_rel_type
        self.spatial_dist_norm = cfg.spatial_encoder.spatial_dist_norm
        self.spatial_dim = cfg.spatial_encoder.spatial_dim
        self.obj_loc_encoding = cfg.spatial_encoder.obj_loc_encoding

        # location encoding
        if self.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.obj_loc_encoding == 'diff_all':
            num_loc_layers = cfg.spatial_encoder.num_layers

        loc_layer = nn.Sequential(
            nn.Linear(cfg.spatial_encoder.dim_loc, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.loc_layers = layer_repeat(loc_layer, num_loc_layers)

        logger.info("Build 3D module: OSE3D")

        # only initialize spatial encoder and loc layers
        self.spatial_encoder.apply(_init_weights_bert)
        self.loc_layers.apply(_init_weights_bert)

        if self.use_embodied_token:
            nn.init.normal_(self.anchor_feat, std=0.02)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, data_dict):
        """
        data_dict requires keys:
            obj_fts: (B, N, P, 6), xyz + rgb
            obj_masks: (B, N), 1 valid and 0 masked
            obj_locs: (B, N, 6), xyz + whd
            anchor_locs: (B, 3)
            anchor_orientation: (B, C)
        """

        obj_feats = self.obj_encoder(data_dict['obj_fts'])
        obj_feats = self.obj_proj(obj_feats)
        obj_masks = ~data_dict['obj_masks']   # flipped due to different convention of TransformerEncoder

        B, N = obj_feats.shape[:2]
        device = obj_feats.device

        obj_type_ids = torch.zeros((B, N), dtype=torch.long, device=device)
        obj_type_embeds = self.obj_type_embed(obj_type_ids)

        if self.use_embodied_token:
            # anchor feature
            anchor_orient = data_dict['anchor_orientation'].unsqueeze(1)
            anchor_orient_feat = self.orient_encoder(generate_fourier_features(anchor_orient))
            anchor_feat = self.anchor_feat + anchor_orient_feat
            anchor_mask = torch.zeros((B, 1), dtype=bool, device=device)

            # anchor loc (3) + size (3)
            anchor_loc = torch.cat(
                [data_dict['anchor_locs'].unsqueeze(1), self.anchor_size.expand(B, -1, -1).to(device)], dim=-1
            )

            # anchor type
            anchor_type_id = torch.ones((B, 1), dtype=torch.long, device=device)
            anchor_type_embed = self.obj_type_embed(anchor_type_id)

            # fuse anchor and objs
            all_obj_feats = torch.cat([anchor_feat, obj_feats], dim=1)
            all_obj_masks = torch.cat((anchor_mask, obj_masks), dim=1)

            all_obj_locs = torch.cat([anchor_loc, data_dict['obj_locs']], dim=1)
            all_obj_type_embeds = torch.cat((anchor_type_embed, obj_type_embeds), dim=1)

        else:
            all_obj_feats = obj_feats
            all_obj_masks = obj_masks

            all_obj_locs = data_dict['obj_locs']
            all_obj_type_embeds = obj_type_embeds

        all_obj_feats = all_obj_feats + all_obj_type_embeds

        # call spatial encoder
        if self.use_spatial_attn:
            pairwise_locs = calc_pairwise_locs(
                all_obj_locs[:, :, :3],
                all_obj_locs[:, :, 3:],
                pairwise_rel_type=self.pairwise_rel_type,
                spatial_dist_norm=self.spatial_dist_norm,
                spatial_dim=self.spatial_dim,
            )

        for i, pc_layer in enumerate(self.spatial_encoder):
            if self.obj_loc_encoding == 'diff_all':
                query_pos = self.loc_layers[i](all_obj_locs)
            else:
                query_pos = self.loc_layers[0](all_obj_locs)
            if not (self.obj_loc_encoding == 'same_0' and i > 0):
                all_obj_feats = all_obj_feats + query_pos

            if self.use_spatial_attn:
                all_obj_feats, _ = pc_layer(
                    all_obj_feats, pairwise_locs,
                    tgt_key_padding_mask=all_obj_masks
                )
            else:
                all_obj_feats, _ = pc_layer(
                    all_obj_feats,
                    tgt_key_padding_mask=all_obj_masks
                )

        data_dict['obj_tokens'] = all_obj_feats
        data_dict['obj_masks'] = ~all_obj_masks

        return data_dict

def simple_conv_and_linear_weights_init(m):
    if type(m) in [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        simple_linear_weights_init(m)

def simple_linear_weights_init(m):
    if type(m) == nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Backbone2DWrapper(nn.Module):

    def __init__(self, model, tag, freeze=True):
        super().__init__()
        self.model = model
        self.tag = tag
        self.freeze = freeze
        if 'convnext' in tag:
            self.out_channels = 1024
        elif 'swin' in tag:
            self.out_channels = 1024
        elif 'vit' in tag:
            self.out_channels = 768
        elif 'resnet' in tag:
            self.out_channels = 2048
        else:
            raise NotImplementedError

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            self.train = disabled_train

    def forward_normal(self, x, flat_output=False):
        feat = self.model.forward_features(x)
        if 'swin' in self.tag:
            feat = rearrange(feat, 'b h w c -> b c h w')
        if 'vit_base_32_timm_laion2b' in self.tag or 'vit_base_32_timm_openai' in self.tag:
            # TODO: [CLS] is prepended to the patches.
            feat = rearrange(feat[:, 1:], 'b (h w) c -> b c h w', h=7)
        if flat_output:
            feat = rearrange(feat, 'b c h w -> b (h w) c')
        return feat

    @torch.no_grad()
    def forward_frozen(self, x, flat_output=False):
        return self.forward_normal(x, flat_output)

    def forward(self, x, flat_output=False):
        if self.freeze:
            return self.forward_frozen(x, flat_output)
        else:
            return self.forward_normal(x, flat_output)
        
def convnext_base_laion2b(pretrained=False, freeze=True, **kwargs):
    m = timm.create_model(
        'convnext_base.clip_laion2b',
        pretrained=pretrained
    )
    if kwargs.get('reset_clip_s2b2'):
        logger.debug('Resetting the last conv layer of convnext-base to random init.')
        s = m.state_dict()
        for i in s.keys():
            if 'stages.3.blocks.2' in i and ('weight' in i or 'bias' in i):
                s[i].normal_()
        m.load_state_dict(s, strict=True)

    return Backbone2DWrapper(m, 'convnext_base_laion2b', freeze=freeze)

@VISION_REGISTRY.register()
class GridFeatureExtractor2D(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        init_func_name = '_'.join([cfg.backbone_name, cfg.backbone_pretrain_dataset])
        init_func = globals().get(init_func_name)
        if init_func and callable(init_func):
            self.backbone = init_func(pretrained=cfg.use_pretrain, freeze=cfg.freeze)
        else:
            raise NotImplementedError(f"Backbone2D does not support {init_func_name}")

        self.pooling = cfg.pooling
        if self.pooling:
            if self.pooling == 'avg':
                self.pooling_layers = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1,1)),
                    nn.Flatten()
                )
                self.out_channels = self.backbone.out_channels
            elif self.pooling == 'conv':
                self.pooling_layers = nn.Sequential(
                    nn.Conv2d(self.backbone.out_channels, 64, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 1),
                    nn.Flatten()
                )
                self.pooling_layers.apply(simple_conv_and_linear_weights_init)
                self.out_channels = 32 * 7 * 7   # hardcode for 224x224
            elif self.pooling in ['attn', 'attention']:
                self.visual_attention = nn.Sequential(
                    nn.Conv2d(self.backbone.out_channels, self.backbone.out_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.backbone.out_channels, self.backbone.out_channels, 1),
                )
                self.visual_attention.apply(simple_conv_and_linear_weights_init)
                def _attention_pooling(x):
                    B, C, H, W = x.size()
                    attn = self.visual_attention(x)
                    attn = attn.view(B, C, -1)
                    x = x.view(B, C, -1)
                    attn = attn.softmax(dim=-1)
                    x = torch.einsum('b c n, b c n -> b c', x, x)
                    return x
                self.pooling_layers = _attention_pooling
                self.out_channels = self.backbone.out_channels
            else:
                raise NotImplementedError(f"Backbone2D does not support {self.pooling} pooling")
        else:
            self.out_channels = self.backbone.out_channels

        logger.info(f"Build Backbone2D: {init_func_name}, " +
                    f"pretrain = {cfg.use_pretrain}, freeze = {cfg.freeze}, " +
                    f"pooling = {self.pooling if self.pooling else None}")

    def forward(self, x):
        if self.pooling:
            x = self.backbone(x, flat_output=False)
            x = self.pooling_layers(x).unsqueeze(1)
            return x
        else:
            return self.backbone(x, flat_output=True)