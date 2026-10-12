import torch
import torch.nn as nn
import torch.nn.functional as F
from .img_network import UNetEncoder
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from .multimodal_network import MLPo
from einops import rearrange
import math

class SimGro(nn.Module):
    def __init__(self, 
        input_channels: int = 1, 
        num_nodule_classes: int = 5, 
        img_size: tuple = (48, 48, 48),
        num_templates: int = 3,  
        feature_dim: int = 128             
    ):
        super().__init__()
        self.num_nodule_classes = num_nodule_classes
        self.num_templates = num_templates
        self.current_template_id_map = {}
        num_stages = 5
        dim = 3
        conv_op = convert_dim_to_conv_op(dim)
        kwargs = {
                'conv_bias': True,
                'norm_op': get_matching_instancenorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        conv_or_blocks_per_stage = {
            'n_conv_per_stage': [2,2,2,2,2],
            'n_conv_per_stage_decoder': [2,2,2,2]
        }
        self.input_channels = input_channels
        base_channels = 32 
        max_channels = 256 
        self.uni_channels = 256
        num_seg_classes = 2
        self.num_nodule_classes = num_nodule_classes
        self.img_z = img_size[0]
        self.img_hw = img_size[1]
        conv_kernel_sizes = [[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]
        pool_op_kernel_sizes = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
        encoded_shapes = []
        shapez, shapeh, shapew = self.img_z, self.img_hw, self.img_hw
        for stagei in range(num_stages):
            shapez = math.ceil(shapez/pool_op_kernel_sizes[stagei][0])
            shapeh = math.ceil(shapeh/pool_op_kernel_sizes[stagei][1])
            shapew = math.ceil(shapew/pool_op_kernel_sizes[stagei][2])
            encoded_shapes.append([shapez, shapeh, shapew])
        
        self.img_encoder = UNetEncoder(
            input_channels = 1,
            n_stages = num_stages,
            features_per_stage= [min(base_channels * 2 ** i, max_channels) for i in range(num_stages)], 
            conv_op = conv_op,
            kernel_sizes = conv_kernel_sizes,
            strides=pool_op_kernel_sizes,
            num_classes=num_seg_classes,
            deep_supervision=False,
            dim_uni = self.uni_channels,
            min_zhw = encoded_shapes,
            **conv_or_blocks_per_stage,
            **kwargs
        )

        flatten_dim = encoded_shapes[-1][0] * encoded_shapes[-1][1] * encoded_shapes[-1][2] * max_channels

        self.mapper = nn.Sequential(
            MLPo(flatten_dim, self.uni_channels, 512, dropout=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

        self.register_buffer('templates', torch.randn(num_nodule_classes, num_templates, feature_dim))
        self.templates = F.normalize(self.templates, p=2, dim=-1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, img_current):
        deep_features = self.img_encoder(img_current)
        flat_features = rearrange(deep_features, 'b (c p) z h w -> b (p c z h w)', p=1)
        features = self.mapper(flat_features)
        return F.normalize(features, p=2, dim=1)

    def classify_from_features(self, features_norm, tau=0.1):
        templates_detach = self.templates.detach()
        all_templates = rearrange(templates_detach, 'c t d -> (c t) d')
        sim_matrix = torch.mm(features_norm, all_templates.t())
        sim_matrix = rearrange(sim_matrix, 'b (c t) -> b c t', c=self.num_nodule_classes)
        class_scores, _ = torch.max(sim_matrix, dim=2)
        return class_scores / tau, class_scores

    def forward_with_features(self, img_current, tau=0.1):
        features_norm = self.extract_features(img_current)
        logits, class_scores = self.classify_from_features(features_norm, tau=tau)
        return logits, features_norm, class_scores

    def forward(self, img_current, tau=0.1, labels=None, margin=0.2):
        logits, features_norm, _ = self.forward_with_features(img_current, tau=tau)
        if self.training:
            return logits, features_norm
        else:
            return logits