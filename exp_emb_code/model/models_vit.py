# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch

import timm.models.vision_transformer
#import model.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.pretrained=True
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :] # without cls token (N, L=14*14, D=768=16*16*3)
            x = x.mean(dim=1) # global average pooling (N, D=768)
            outcome = self.fc_norm(x) # Layer Normalization (N, D=768)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    # borrow from timm
    def forward(self, x, ret_feature=False):
        x = self.forward_features(x)
        feature = x
        if getattr(self, 'head_dist', None) is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # return
        if ret_feature:
            return x, feature
        else:
            return x


# setup model archs
VIT_KWARGS_BASE = dict(mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

VIT_KWARGS_PRESETS = {
    'micro': dict(patch_size=16, embed_dim=96, depth=12, num_heads=2),
    'mini': dict(patch_size=16, embed_dim=128, depth=12, num_heads=2),
    'tiny_d6': dict(patch_size=16, embed_dim=192, depth=6, num_heads=3),
    'tiny': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    'small': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    'base': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'large': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    'huge': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    'giant': dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11),
    'gigantic': dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64/13),
}

def create_vit_model(preset=None, creator=None, **kwargs):
    preset = 'base' if preset is None else preset.lower()
    all_kwargs = dict()
    all_kwargs.update(VIT_KWARGS_BASE)
    all_kwargs.update(VIT_KWARGS_PRESETS[preset])
    all_kwargs.update(kwargs)
    if creator is None:
        creator = VisionTransformer
    return creator(**all_kwargs)

vit_micro_patch16 = partial(create_vit_model, preset='micro')
vit_mini_patch16 = partial(create_vit_model, preset='mini')
vit_tiny_d6_patch16 = partial(create_vit_model, preset='tiny_d6')
vit_tiny_patch16 = partial(create_vit_model, preset='tiny')
vit_small_patch16 = partial(create_vit_model, preset='small')
vit_base_patch16 = partial(create_vit_model, preset='base')
vit_large_patch16 = partial(create_vit_model, preset='large')
vit_huge_patch14 = partial(create_vit_model, preset='huge')
vit_giant_patch14 = partial(create_vit_model, preset='giant')
vit_gigantic_patch14 = partial(create_vit_model, preset='gigantic')



