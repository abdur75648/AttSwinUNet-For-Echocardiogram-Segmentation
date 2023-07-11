import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
import copy
from attention_swin_unet.embedding import PatchEmbed
from attention_swin_unet.encoder import Encoder
from attention_swin_unet.decoder import Decoder


class SwinAttentionUnet(nn.Module):
    r""" SwinAttentionUnet
        we use A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
        and edited some parts.
    Args:
        config (int | tuple(int)): Input image size. Default 224
        num_classes (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self,in_chans=3, out_chans=1):
        super().__init__()
        self.img_size        = 224
        self.patch_size      = 4
        self.in_chans        = in_chans
        self.embed_dim       = 96
        self.depths          = [2, 2, 6, 2]
        self.num_heads       = [3, 6, 12, 24]
        self.window_size     = 7
        self.mlp_ratio       = 4
        self.qkv_bias        = True
        self.qk_scale        = None
        self.drop_rate       = 0.0
        self.drop_path_rate  = 0.1
        self.attn_drop_rate  = 0
        self.ape             = False
        self.patch_norm      = True
        self.use_checkpoint  = False
        self.num_classes     = out_chans
        self.num_layers      = len(self.depths)
        self.num_features    = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(self.embed_dim * 2)
        self.mode            = "swin"
        self.skip_num        = 3
        self.operation       = '+'
        self.add_attention   = '1'
        self.final_upsample  = "expand_first"
        self.norm_layer      = nn.LayerNorm
        
        #Build embedding
        self.patch_embed = PatchEmbed(
        img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,
        norm_layer= nn.LayerNorm if self.patch_norm else None)
        self.num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        #Build encoder
        self.encoder = Encoder(embed_dim= self.embed_dim,depths =self.depths,num_heads = self.num_heads,window_size = self.window_size,mlp_ratio = self.mlp_ratio,qkv_bias= self.qkv_bias ,\
        qk_scale  = self.qk_scale,drop_rate = self.drop_rate,attn_drop_rate = self.attn_drop_rate,norm_layer= self.norm_layer,use_checkpoint = self.use_checkpoint,\
        num_layers = self.num_layers,img_size  = self.img_size,ape=False,num_patches =self.num_patches,patch_size=self.patch_size,in_chans=self.in_chans, drop_path_rate=self.drop_path_rate,patch_embed= self.patch_embed)#,args = config)
        #Build decoder
        self.decoder =  Decoder(embed_dim = self.embed_dim,depths = self.depths,num_heads = self.num_heads,img_size =self.img_size,\
        window_size = self.window_size,mlp_ratio = self.mlp_ratio,qkv_bias = self.qkv_bias,qk_scale = self.qk_scale,\
        drop_rate = self.drop_rate,attn_drop_rate = self.attn_drop_rate,norm_layer = self.norm_layer,\
        use_checkpoint = self.use_checkpoint,num_layers = self.num_layers,\
        patch_embed = self.patch_embed,patch_norm=True,final_upsample="expand_first",num_classes=self.num_classes,mode=self.mode,skip_num=self.skip_num,operation=self.operation, add_attention=self.add_attention)#,args = config )
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

   
    def forward(self, x):
        # print(x.shape)
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        # print(x.shape)
        x, x_downsample,x_attention_encoder = self.encoder(x)
        x = self.decoder(x,x_downsample,x_attention_encoder)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


# x = torch.randn((1,1,224,224))
# print("Tensor Created")
# model = SwinAttentionUnet(in_chans=1)
# print("Model Created")
# preds = model(x)
# print("Inference Done")
# print(preds.shape) # torch.randn((1,1,224,224))