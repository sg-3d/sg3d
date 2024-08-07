import torch
import torch.nn as nn

from modules.build import GROUNDING_REGISTRY
from modules.layers.transformers import (CrossAttentionLayer,
                                         TransformerEncoderLayer)
from modules.utils import layer_repeat, calc_pairwise_locs
from modules.weights import _init_weights_bert


@GROUNDING_REGISTRY.register()
class UnifiedDecoder(nn.Module):
    """
       spatial_dim: spatial feature dim, used to modify attention
       dim_loc:
    """

    def __init__(self, cfg, hidden_size=768, num_attention_heads=12, num_layers=4, dim_loc=6):
        super().__init__()

        # cross attention
        cross_attn_layer = CrossAttentionLayer(hidden_size, num_attention_heads)
        self.cross_attn = layer_repeat(cross_attn_layer, num_layers)

        # unfied encoder
        unified_encoder_layer = TransformerEncoderLayer(hidden_size, num_attention_heads)
        self.unified_encoder = layer_repeat(unified_encoder_layer, num_layers)

        # loc layer
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = layer_repeat(loc_layer, 1)

        # token embedding
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        self.apply(_init_weights_bert)

    def forward(
            self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
            output_attentions=False, output_hidden_states=False, **kwargs
    ):
        txt_len = txt_embeds.shape[1]
        obj_len = obj_embeds.shape[1]

        # add embeddings for points
        query_pos = self.loc_layers[0](obj_locs)
        pc_token_type_ids = torch.ones((obj_embeds.shape[0:2])).long().cuda()
        pc_type_embeds = self.token_type_embeddings(pc_token_type_ids)  # ?
        obj_embeds = obj_embeds + query_pos
        query_pos = query_pos + pc_type_embeds

        # add embeddings for languages
        lang_token_type_ids = torch.zeros((txt_embeds.shape[0:2])).long().cuda()
        lang_type_embeds = self.token_type_embeddings(lang_token_type_ids)
        txt_embeds = txt_embeds + lang_type_embeds

        for i, (cross_attn, unified_layer) in enumerate(zip(self.cross_attn, self.unified_encoder)):
            # cross attention
            query_pos, cross_attn_matrices = cross_attn(query_pos, obj_embeds, 
                                                        tgt_key_padding_mask=obj_masks.logical_not(), 
                                                        memory_key_padding_mask=obj_masks.logical_not())

            # fuse embeddings
            joint_embeds = torch.cat((txt_embeds, query_pos), dim=1)
            joint_masks = torch.cat((txt_masks, obj_masks), dim=1)

            # self attention
            joint_embeds, self_attn_matrices = unified_layer(joint_embeds,
                                                             tgt_key_padding_mask=joint_masks.logical_not())

            # split
            txt_embeds, query_pos = torch.split(joint_embeds, [txt_len, obj_len], dim=1)

        return txt_embeds, query_pos