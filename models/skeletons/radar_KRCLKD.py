import torch
import torch.nn as nn
import copy
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.nn import MultiheadAttention

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

###### Squeeze-and-Excitation #######
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y                     # channel-wise scaling

class ConcatConv_SE(nn.Module):
    def __init__(self, in_channels=768, common_dim=128, output_dim=768, se_r=16):
        super().__init__()
        # 각 맵마다 독립적으로 1×1 Conv+BN (성능 ↑, 파라 ↑)
        self.pre = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, common_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(common_dim),
                nn.ReLU(inplace=True),
                SEBlock(common_dim, r=se_r)          # <-- SE 삽입
            ) for _ in range(3)
        ])
        self.merge_conv = nn.Conv2d(common_dim * 3, output_dim, kernel_size=1, bias=True)

    def forward(self, x1, x2, x3):
        x1, x2, x3 = (m(x) for m, x in zip(self.pre, (x1, x2, x3)))
        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.merge_conv(x_cat)
########################################

################ CBAM ##############################
class ChannelGate(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False)
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.mlp(self.avg_pool(x).view(b, c))
        max_ = self.mlp(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg + max_).view(b, c, 1, 1)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * scale

class CBAMBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, r)
        self.spatial_gate = SpatialGate()
    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x

class ConcatConv_CBAM(nn.Module):
    def __init__(self, in_channels=768, common_dim=128, output_dim=768, r=16):
        super().__init__()
        self.pre = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, common_dim, 1, bias=False),
                nn.BatchNorm2d(common_dim),
                nn.ReLU(inplace=True),
                CBAMBlock(common_dim, r=r)           # <-- CBAM 삽입
            ) for _ in range(3)
        ])
        # CBAM을 거친 뒤 concat
        self.merge_conv = nn.Conv2d(common_dim * 3, output_dim, 1, bias=True)
        self.post_cbam = CBAMBlock(output_dim, r=4)  # 최종 CBAM 한 번 더 (선택)

    def forward(self, x1, x2, x3):
        x1, x2, x3 = (m(x) for m, x in zip(self.pre, (x1, x2, x3)))
        out = self.merge_conv(torch.cat([x1, x2, x3], dim=1))
        return self.post_cbam(out)

class ConcatConv_CBAM_2(nn.Module):
    def __init__(self, in_channels=768, common_dim=128, output_dim=768, r=8):
        super().__init__()
        self.pre = nn.ModuleList([
            nn.Sequential(
                # Conv2d
                nn.Conv2d(in_channels, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                #convnext
                nn.Conv2d(256, 256, 1, bias=False),
                nn.LayerNorm([256,80,180], eps=1e-6),
                nn.Conv2d(256,256*3,1,1,0),
                nn.GELU(),
                nn.Conv2d(256*3,256,1,1,0),
                # Conv2d
                nn.Conv2d(256, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])
        # CBAM을 거친 뒤 concat
        self.merge_conv = nn.Conv2d(common_dim * 3, output_dim, 1, bias=True)
        self.post_cbam = CBAMBlock(output_dim, r=8)  # 최종 CBAM 한 번 더 (선택)

    def forward(self, x1, x2, x3):
        x1, x2, x3 = (m(x) for m, x in zip(self.pre, (x1, x2, x3)))
        out = self.merge_conv(torch.cat([x1, x2, x3], dim=1))
        return self.post_cbam(out)

class ConcatConv_CBAM_3(nn.Module):
    def __init__(self, in_channels=768, common_dim=128, output_dim=768, r=8):
        super().__init__()
        self.pre = nn.ModuleList([
            nn.Sequential(
                # Conv2d
                nn.Conv2d(in_channels, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                #convnext
                nn.Conv2d(256, 256, 1, bias=False),
                nn.LayerNorm([256,80,180], eps=1e-6),
                nn.Conv2d(256,256*3,1,1,0),
                nn.GELU(),
                nn.Conv2d(256*3,256,1,1,0),
                # Conv2d
                nn.Conv2d(256, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])
        # CBAM
        self.post_cbam = CBAMBlock(common_dim*3, r=8)  # 최종 CBAM 한 번 더 (선택)
        self.conv_last = nn.Conv2d(common_dim*3, output_dim, 1, bias=True)

    def forward(self, x1, x2, x3):
        x1, x2, x3 = (m(x) for m, x in zip(self.pre, (x1, x2, x3)))
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.post_cbam(out)
        return self.conv_last(out)

class ConcatConv_CBAM_2teacher(nn.Module):
    def __init__(self, in_channels=768, common_dim=128, output_dim=768, r=8):
        super().__init__()
        self.pre = nn.ModuleList([
            nn.Sequential(
                # Conv2d
                nn.Conv2d(in_channels, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                #convnext
                nn.Conv2d(256, 256, 1, bias=False),
                nn.LayerNorm([256,80,180], eps=1e-6),
                nn.Conv2d(256,256*3,1,1,0),
                nn.GELU(),
                nn.Conv2d(256*3,256,1,1,0),
                # Conv2d
                nn.Conv2d(256, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ) for _ in range(2)
        ])
        # CBAM
        self.post_cbam = CBAMBlock(common_dim*2, r=8)  # 최종 CBAM 한 번 더 (선택)
        self.conv_last = nn.Conv2d(common_dim*2, output_dim, 1, bias=True)

    def forward(self, x1, x2):
        x1, x2 = (m(x) for m, x in zip(self.pre, (x1, x2)))
        out = torch.cat([x1, x2], dim=1)
        out = self.post_cbam(out)
        return self.conv_last(out)


#################################################

#### avg + self-attention ####
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class CombineFeatureMaps(nn.Module): # Comparison 1
    def __init__(self):
        super(CombineFeatureMaps, self).__init__()
        self.spatial_attention = SpatialAttentionModule()

    # def forward(self, x1, x2, x3):
    #     combined = (x1 + x2 + x3) / 3
    #     attention_map = self.spatial_attention(combined)
    #     return combined * attention_map, attention_map
        
    def forward(self,x1, x2, x3):
        combined = (x1+ x2 + x3) / 3
        attention_map = self.spatial_attention(combined)
        return combined * attention_map, attention_map
##############################    

#### multi-head attention ####
class PartialAttentionKDModule(nn.Module):
    def __init__(self, in_channel=768, common_dim=128, patch_size=20, embed_dim=2048, n_heads=8, dropout=0.0, bias=False, patch_expansion=1, output_dim=768, out_bias=True):
        super(PartialAttentionKDModule, self).__init__()
        
        self.to_p1 = nn.Sequential(
            nn.Conv2d(in_channel, common_dim, 1, bias=True),
            nn.BatchNorm2d(common_dim)
        )
        self.to_p2 = nn.Sequential(
            nn.Conv2d(in_channel, common_dim, 1, bias=True),
            nn.BatchNorm2d(common_dim)
        )
        self.to_p3 = nn.Sequential(
            nn.Conv2d(in_channel, common_dim, 1, bias=True),
            nn.BatchNorm2d(common_dim)
        )
        
        self.rearr = Rearrange('b d (ph h) (pw w) -> (b h w) (d ph pw)', ph=patch_size, pw=patch_size)
        
        input_dim = common_dim*patch_size*patch_size
        if input_dim != embed_dim: # TODO: seperate if it does not work
            self.to_embed = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, embed_dim, bias=False),
                nn.LayerNorm(embed_dim)
            )
        else:
            self.to_embed = None
        
        n_query = patch_size*patch_size*patch_expansion
        self.kd_query = nn.Parameter(torch.randn(1, n_query, embed_dim), requires_grad=True)
        self.kd_att = nn.MultiheadAttention(embed_dim, n_heads, dropout, bias, batch_first=True)
        self.patch_size = patch_size
        self.patch_expansion = patch_expansion
        
        self.to_feat_ln = nn.LayerNorm(embed_dim)
        self.to_out_feat = nn.Conv2d(embed_dim*patch_expansion, output_dim, 1, bias=out_bias)
        
    def forward(self, x1, x2, x3):
        b, _, h, w = x1.shape
        
        x1 = self.rearr(self.to_p1(x1)) #[8, 768, 80, 180] --> [8, 128, 80, 180] --> [8x4x9, 128x20x20]
        x2 = self.rearr(self.to_p2(x2))
        x3 = self.rearr(self.to_p3(x3))
        
        if self.to_embed is not None:
            x1 = self.to_embed(x1) # [8x4x9, 128x20x20] --> [8x4x9, 2048]
            x2 = self.to_embed(x2)
            x3 = self.to_embed(x3)
        
        x1 = x1.unsqueeze(1) # [8x4x9, 1, 2048]
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        
        kv_feat = torch.concat((x1, x2, x3), dim=1) # [8x4x9, 3, 2048]
        
        bb = int(np.round(b*h*w/(self.patch_size*self.patch_size))) #[8x4x9]
        kd_q = repeat(self.kd_query, 'b n c -> (b bb) n c', bb=bb) #[1, 400, 2048]--> [8x4x9, 400, 2048]
        
        att_feat, att_score = self.kd_att(kd_q, kv_feat, kv_feat)
        att_feat = self.to_feat_ln(att_feat)
        
        att_feat = rearrange(att_feat, '(b h w) (ph pw pex) d -> b (d pex) (ph h) (pw w)', \
            b=b, h=int(np.round(h/self.patch_size)), w=int(np.round(w/self.patch_size)), \
            ph=self.patch_size, pw=self.patch_size, pex=self.patch_expansion)
        
        att_feat = self.to_out_feat(att_feat)
        
        return att_feat, att_score
##############################

####### Concat + Conv ########
class ConcatConvModule(nn.Module):
    def __init__(self, in_channels=768, common_dim=128, output_dim=768):
        super(ConcatConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, common_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(common_dim)
        
        # Conv layer to merge concatenated feature maps
        self.merge_conv = nn.Conv2d(common_dim * 3, output_dim, kernel_size=1, bias=True)
        
    def forward(self, x1, x2, x3):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x2 = F.relu(self.bn1(self.conv1(x2)))
        x3 = F.relu(self.bn1(self.conv1(x3)))
        
        # Concatenate along the channel dimension
        x_cat = torch.cat([x1, x2, x3], dim=1)
        
        # Apply the merging convolution
        out = self.merge_conv(x_cat)
        
        return out
##############################

######## weighted sum ########
class WeightedSumModule(nn.Module):
    def __init__(self, in_channels=768, common_dim=128, output_dim=768):
        super(WeightedSumModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, common_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(common_dim)
        
        # Conv layer to process the weighted sum feature map
        self.output_conv = nn.Conv2d(common_dim, output_dim, kernel_size=1, bias=True)
        
        # Learnable weights for the feature maps
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)
        
    def forward(self, x1, x2, x3):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x2 = F.relu(self.bn1(self.conv1(x2)))
        x3 = F.relu(self.bn1(self.conv1(x3)))
        
        # Apply softmax to weights to ensure they sum to 1
        weights = F.softmax(self.weights, dim=0)
        
        # Compute weighted sum of the feature maps
        x_weighted_sum = weights[0] * x1 + weights[1] * x2 + weights[2] * x3
        
        # Apply the output convolution
        out = self.output_conv(x_weighted_sum)
        
        return out
##############################

class radar_KRCLKD(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_teacher1 = None
        self.model_teacher2 = None
        self.model_teacher3 = None
        
        self.model_student = None

        # self.avg_att = CombineFeatureMaps().cuda()
        # self.conv_cat = ConcatConvModule().cuda()
        # self.kd_att = PartialAttentionKDModule().cuda()
        # self.weigted_sum = WeightedSumModule().cuda()
        # self.se = ConcatConv_SE().cuda()
        self.cbam = ConcatConv_CBAM_3().cuda()
        # self.cbam_2techer = ConcatConv_CBAM_2teacher().cuda()

        # bev_loss cfg
        self.calculate_bev_loss = model_cfg.get('LOSS_BEV', True)
        self.bev_loss_fun = torch.nn.MSELoss(reduction='none')
        self.bev_loss_type = model_cfg.get('LOSS_BEV_TYPE', 'L2')
        self.bev_loss_weight = model_cfg.get('LOSS_BEV_WEIGHT', 32)
        
        self.amp_training = False

    def forward(self, batch_dict, data1, data2, data3):
        
        loss, tb_dict= self.get_training_loss(batch_dict, data1, data2, data3)

        tb_dict['loss'] = loss

        return loss, tb_dict
        
    def get_mask(self, attention_map):
        
        feature_map = attention_map.cpu().detach().numpy()[0, 0]
        mean_value = np.mean(feature_map)
        difference = np.abs(feature_map - mean_value)
        max_difference = np.max(difference)
        normalized_difference = difference / max_difference
        normalized_difference = torch.tensor(normalized_difference).float().cuda()
        
        return normalized_difference

    def get_training_loss(self, batch_dict, data1, data2, data3):

        # forward teacher model
        if self.training:
            with torch.no_grad():
                teacher_dict_1 = self.model_teacher1(data1) 
                teacher_dict_2 = self.model_teacher2(data2)
                teacher_dict_3 = self.model_teacher3(data3)
            
                feat_teacher1 = teacher_dict_1['bev_feat'] #[b, 768, 80, 180]
                feat_teacher2 = teacher_dict_2['bev_feat'] #[b, 768, 80, 180]
                feat_teacher3 = teacher_dict_3['bev_feat'] #[b, 768, 80, 180]
            
            
            #collabo_feature, attention_map = combiner(feat_teacher1, feat_teacher2, feat_teacher3) #[b, 768, 80, 180]

            # collabo_feature, _ = self.kd_att(feat_teacher1, feat_teacher2, feat_teacher3)
            
            # collabo_feature = self.conv_cat(feat_teacher1, feat_teacher2, feat_teacher3)
            
            # collabo_feature = self.weigted_sum(feat_teacher1, feat_teacher2, feat_teacher3)

            # collabo_feature = self.se(feat_teacher1, feat_teacher2, feat_teacher3)

            collabo_feature = self.cbam(feat_teacher1, feat_teacher2, feat_teacher3)
            # collabo_feature = feat_teacher3
            
        # att_mask = self.get_mask(attention_map)

        # feature_map = feat_teacher1.cpu().detach().numpy()
        # output_dir = '/media/oem/c886c17a-87f1-42c0-9d3c-831467b43160/Sseunghyun_1214/K-Radar_KD/heatmap/seq11_154/20p'
        # for i in range(feature_map.shape[1]):
        #     heatmap = feature_map[0, i, :, :]
        #     plt.imshow(heatmap)
        #     plt.colorbar()
        #     plt.title(f'Channel {i}')
        #     plt.savefig(os.path.join(output_dir, f'heatmap_channel_{i}.png'))
        #     plt.close()

        #calculate loss

        # rpn_loss and bev loss
        # pass to student model
        batch_dict = self.model_student(batch_dict)
        loss_rpn, tb_dict = self.model_student.head.loss(batch_dict)
        
        # bev_loss
        student_feature = batch_dict['bev_layer'] #[b, 768, 80, 180]
        gt_mask = batch_dict['heatmap'][0].cuda() #[80, 180]
        loss_bev = torch.tensor(0)

        #gt masking
        if (collabo_feature is not None) and (student_feature is not None) and self.calculate_bev_loss:

            bev_loss_mask = torch.ones((collabo_feature.shape[0], 1, collabo_feature.shape[2], collabo_feature.shape[3]), device = collabo_feature.device)

            bev_loss_mask *= gt_mask

            noralizer = bev_loss_mask.numel() / bev_loss_mask.sum()

            loss_bev = (self.bev_loss_fun(student_feature, collabo_feature)*bev_loss_mask).mean()*noralizer

            loss_bev *= self.bev_loss_weight

        #all loss
        loss = loss_bev + loss_rpn

        loss_rpn = loss_rpn.detach()
        loss_bev = loss_bev.detach()

        tb_dict.update({ 
            "bev_loss":loss_bev.item(),
            "rpn_loss": loss_rpn.item(),
        })
        
        
        return loss, tb_dict
    

