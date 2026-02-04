import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

import matplotlib.pyplot as plt
import os
norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

class RPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):

            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = nn.Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)


    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.append(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.append(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            if j < num_blocks -1:
                block.append(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m, distribution="uniform")

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = F.relu(self.blocks[i](x))
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x

class Progressive_RPN(RPN):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(Progressive_RPN, self).__init__(layer_nums,ds_layer_strides,ds_num_filters,us_layer_strides,us_num_filters,num_input_features,norm_cfg, name,logger)

        # S2D module
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(num_input_features, 256, kernel_size=3, stride=2, padding=1),  # Output: [4, 256, 40, 90]
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Output: [4, 256, 20, 45]
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.convnext_block_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,20,45], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.convnext_block_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,20,45], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        
        self.convnext_block_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,20,45], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )


        self.decoder_1 = nn.Sequential( # 94,94,256
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),  
        )

        self.decoder_2 = nn.Sequential( # 188,188,256
            nn.Conv2d(512,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256,num_input_features,4,2,1),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        )
        
        self.fusion_sparse = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        )

        self.fusion_dense = nn.Sequential(
            nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(num_input_features, 640,1,1,0),
            nn.BatchNorm2d(640),
            nn.GELU(),
        )  

        self.generator_1 = nn.Sequential( # N,128,5,188,188
            nn.Conv3d(128,32,1,1,0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32,32,4,2,1), # N,32,10,376,376
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        self.gen_out_4 = nn.Sequential(
            nn.Conv3d(32,3,1,1,0),
        )


        self.gen_mask_4 = nn.Sequential(
            nn.Conv3d(32,1,1,1,0),
        )

        self.generator_2 = nn.Sequential(
            nn.Conv3d(32,16,1,1,0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16,3,4,2,1), # N,16,20,752,752
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )

        self.gen_out_2 = nn.Sequential(
            nn.Conv3d(3,3,1,1,0),
        )

        self.gen_mask_2 = nn.Sequential(
            nn.Conv3d(3,1,1,1,0),
        )


    def forward(self, x):
        
        ups = []

        x_bev = x['bev_feat']
        ######################## S2D Module ###################################
        y_1 = self.encoder_1(x_bev)                        
        y_2 = self.encoder_2(y_1)                      
        att = self.convnext_block_1(y_2) + y_2         
        att = self.convnext_block_2(att) + att                
        att = F.gelu(self.convnext_block_3(att) + att)
        y_3 = torch.cat([self.decoder_1(att) , y_1],1)        # [B, 512, 40, 90]
        F_S_b = self.decoder_2(y_3)                                         #[B, 768, 80, 180]              
        # x['prog_1'] = F_S_b

        y_1 = self.encoder_1(F_S_b)                        
        y_2 = self.encoder_2(y_1)                      
        att = self.convnext_block_1(y_2) + y_2         
        att = self.convnext_block_2(att) + att                
        att = F.gelu(self.convnext_block_3(att) + att)
        y_3 = torch.cat([self.decoder_1(att) , y_1],1)        # [B, 512, 40, 90]
        F_S_b = self.decoder_2(y_3)                                         #[B, 768, 80, 180]              
        # x['prog_2'] = F_S_b
        
        F_S_a = self.fusion_dense(F_S_b) + self.fusion_sparse(x_bev)        #[B, 768, 80, 180]

        for i in range(len(self.blocks)):
            if i == 0:
                x_bev = self.blocks[i](F_S_a)
            else:
                x_bev = self.blocks[i](x_bev)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x_bev)) #[B, 256, 80, 180]
        if len(ups) > 0:
            x_bev = torch.cat(ups, dim=1)
        
        x['bev_layer'] = x_bev
        
        ######################## CBAM ###################################
        # cbam = CBAM(gate_channels=x_bev.shape[1], reduction_ratio=8, pool_types=['avg', 'max']).to(device='cuda')
        # x_cbam = cbam(x_bev)
        # x['bev_cbam'] = x_cbam
        
        
        # feature_map = x_bev.cpu().detach().numpy()
        # output_dir = '/media/oem/c886c17a-87f1-42c0-9d3c-831467b43160/Sseunghyun_1214/K-Radar_KD/heatmap/seq11_154/01p_distill'
        # for i in range(feature_map.shape[1]):
        #     heatmap = feature_map[0, i, :, :]
        #     plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        #     plt.colorbar()
        #     plt.title(f'Channel {i}')
        #     plt.savefig(os.path.join(output_dir, f'heatmap_channel_{i}.png'))
        #     plt.close()
        
        # exit()
        
        return x 