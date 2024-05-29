import math
import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F

from .deformable_attention.ops.modules import MSDeformAttn


class MSCrossAttnBlock(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=16, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropout=0.1, init_values=0.):
        super().__init__()
        self.select_layer = [_ for _ in range(n_levels)]
        self.query_layer = -1 

        self.cross_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.gamma1 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

        self.norm1 = norm_layer(d_model)
        self.self_attn = MSDeformAttn(d_model=d_model, n_levels=1, n_heads=n_heads, n_points=n_points)
        self.gamma2 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, srcs, masks=None, pos_embeds=None):
        # prepare input feat
        src_flatten = []
        spatial_shapes = []
        for lvl in self.select_layer: 
            src = srcs[lvl]
            _, hw, _ = src.shape
            e = int(math.sqrt(hw))
            spatial_shape = (e, e)
            spatial_shapes.append(spatial_shape)
            src_flatten.append(src)
        feat = torch.cat(src_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # cross attn
        pos = None  # TODO
        query = srcs[self.query_layer]
        query = self.with_pos_embed(query, pos)  # bs, h*w, c
        query_e = int(math.sqrt(query.shape[1]))  # h == w

        reference_points = self.get_reference_points([(query_e, query_e)], device=query.device)
        attn = self.cross_attn(self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes,
                               level_start_index, None)

        # self attn
        attn1 = self.norm1(attn)
        attn_pos = None  # TODO
        spatial_shapes_attn = torch.as_tensor([(query_e, query_e)], dtype=torch.long, device=attn1.device)
        level_start_index_attn = torch.cat(
            (spatial_shapes_attn.new_zeros((1,)), spatial_shapes_attn.prod(1).cumsum(0)[:-1]))
        reference_points_attn = self.get_reference_points(spatial_shapes_attn, device=attn1.device)
        attn2 = self.self_attn(self.with_pos_embed(attn1, attn_pos), reference_points_attn, attn1, spatial_shapes_attn,
                               level_start_index_attn, None)
        attn = attn + self.gamma2 * attn2

        # Residual Connection
        tgt = query + self.gamma1 * attn

        return tgt