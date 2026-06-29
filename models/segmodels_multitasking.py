import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
import sys
from typing import List, Tuple, Union
import numpy as np

sys.path.append('./')
from models.base import BaseModel
from models.upernet_head import UPerHead
from models.segformer_head import SegFormerHead
from torch.distributions.uniform import Uniform

#INCASE USING FEATURE PERTURBATION
def Dropout(x, p=0.5): #Applies standard random dropout of features
    x = torch.nn.functional.dropout(x, p)
    return x
def FeatureDropout(x): #Applies feature dropout based on attention map and spatial aware, dropout salient regions from each channel
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    # x = x.mul(drop_mask)
    return x * drop_mask
class FeatureNoise(nn.Module): #Applies feature noise perturbation
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        # x_noise = x * (1 + noise_vector) #x.mul(noise_vector) + x
        return x * (1 + noise_vector)

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x
# ---------------------------------
# Multi-Scale Feature Regularizer
# ---------------------------------
class MultiScaleFeatureRegularizer(nn.Module):
    def __init__(self, use_dropout=True, use_feature_dropout=True, use_feature_noise=True, dropout_p=0.5, noise_range=0.3):
        super(MultiScaleFeatureRegularizer, self).__init__()
        self.use_dropout = use_dropout
        self.use_feature_dropout = use_feature_dropout
        self.use_feature_noise = use_feature_noise
        self.dropout_p = dropout_p
        self.feature_noise = FeatureNoise(uniform_range=noise_range)

    def forward(self, features):
        """
        features: tuple or list of tensors from different stages
        Example: (f1, f2, f3, f4) where each fi is (B, C, H, W)
        """
        out_features = []
        for x in features:
            if self.use_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            if self.use_feature_dropout:
                x = FeatureDropout(x)
            if self.use_feature_noise:
                x = self.feature_noise(x)
            out_features.append(x)
        return tuple(out_features)  # return same format
# Apply feature regularization on all scales
# regularizer = MultiScaleFeatureRegularizer(use_dropout=True, use_feature_dropout=True, use_feature_noise=True)
# features = regularizer(features)
# seg_logits = decoder(features)

class CONVNEXTMODEL(BaseModel):
    """
    ConvNeXt backbone model with UPerHead, boundary decoder, and classification head.
    """
    def __init__(self, backbone: str = 'ConvNeXt-T', num_classes: int = 4):
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 128, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor, Tensor]:
        y = self.backbone(x)
        inp = self.decode_head(y)
        out = F.interpolate(inp, size=x.shape[2:], mode='bilinear', align_corners=False)

        if self.training:
            return y, inp, out
        return y, inp, out

class SEGFORMER(BaseModel):
    """
    SegFormer-based model with boundary and classification heads.
    """
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 4):
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 128, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor, Tensor]:
        y = self.backbone(x)        
        # y_drop = Dropout(y, p=0.5)
        # y_featdrop = FeatureDropout(y)
        # feat_noise = FeatureNoise(uniform_range=0.3)
        # y_featnoise = feat_noise(y)
        #Outputs from different feature perturbation modules can be used for consistency loss computation
        inp = self.decode_head(y)
        # inp_drop = self.decode_head(y_drop)
        # inp_featdrop = self.decode_head(y_featdrop)
        # inp_featnoise = self.decode_head(y_featnoise)
        out = F.interpolate(inp, size=x.shape[2:], mode='bilinear', align_corners=False)
        # out_drop = F.interpolate(inp_drop, size=x.shape[2:], mode='bilinear', align_corners=True)
        # out_featdrop = F.interpolate(inp_featdrop, size=x.shape[2:], mode=' bilinear', align_corners=True)
        # out_featnoise = F.interpolate(inp_featnoise, size=x.shape[2:], mode='bilinear', align_corners=True)

        if self.training:
            return y, inp, out
        return y, inp, out
