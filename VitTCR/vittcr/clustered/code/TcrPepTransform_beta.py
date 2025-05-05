from functools import partial
import torch
import torch.nn as nn
from TcrPepTransform_utils import DropPath, trunc_normal_, lecun_normal_
import math
import numpy as np


class PatchEmbed(nn.Module):
    def __init__(self, 
                 input_height=20, 
                 input_width=12, 
                 patch_size=4, 
                 in_chans=3, 
                 embed_dim=768):
        super().__init__()
        self.height = input_height
        self.width = input_width
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (input_height // patch_size) * (input_width // patch_size)
        self.num_patches = num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width, \
            f"Input image size ({H}*{W}) doesn't match model ({self.height}*{self.width})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim, 
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, 
                              attn_drop=attn_drop, 
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TcrPepTransform_single(nn.Module):
    def __init__(self, 
                 input_height=20, 
                 input_width=12, 
                 in_chans=5, 
                 patch_size=4, 
                 num_classes=2, 
                 embed_dim=256, 
                 depth=1, 
                 num_heads=4, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=None,
                 act_layer=nn.GELU):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(input_height=input_height, 
                                      input_width=input_width, 
                                      patch_size=patch_size, 
                                      in_chans=in_chans, 
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim), requires_grad=True) 
        nn.init.normal_(self.pos_embed, 0., math.sqrt(1. / num_patches)) 
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, 
                                            num_heads=num_heads, 
                                            mlp_ratio=mlp_ratio, 
                                            qkv_bias=qkv_bias, 
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate, 
                                            drop_path=dpr[i], 
                                            norm_layer=norm_layer, 
                                            act_layer=act_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity() # for single chain only
        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
               nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for block_layer in self.blocks:
            x = block_layer(x)
        x = self.norm(x)
        x = torch.mean(x, 1)
        x = self.head(x)
        return x