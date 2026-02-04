import torch.nn as nn

from models import pre_processor, backbone_2d, backbone_3d, head, roi_head
from ..neck import S2D_RPN, Progressive_RPN
import logging

import torch

class RadarBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL
        
        self.list_module_names = [
            'pre_processor', 'backbone', 'neck', 'head', 'roi_head'
        ]
        self.list_modules = []
        self.build_radar_detector()

    def build_radar_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module

    def build_backbone(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        if cfg_backbone is None:
            return None
        
        if cfg_backbone.TYPE == '2D':
            return backbone_2d.__all__[cfg_backbone.NAME](self.cfg)
        elif cfg_backbone.TYPE == '3D':
            return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)
        else:
            return None

    def build_neck(self):
        cfg_neck = self.cfg_model.get('NECK', None)
        if cfg_neck is None:
            return None
        
        neck_module = Progressive_RPN(
            layer_nums=cfg_neck.NECK_CONFIG['layer_nums'],
            ds_layer_strides=cfg_neck.NECK_CONFIG['ds_layer_strides'],
            ds_num_filters=cfg_neck.NECK_CONFIG['ds_num_filters'],
            us_layer_strides=cfg_neck.NECK_CONFIG['us_layer_strides'],
            us_num_filters=cfg_neck.NECK_CONFIG['us_num_filters'],
            num_input_features=cfg_neck.NECK_CONFIG['num_input_features'],
            logger=cfg_neck.NECK_CONFIG.get('logger', logging.getLogger("S2D_RPN"))
        )
        return neck_module

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def forward(self, x):
        for module in self.list_modules:
            x = module(x)
            # temp_mem = torch.cuda.memory_allocated()/1024/1024
            # print('module:', module.type)
            # print(temp_mem)
        return x
