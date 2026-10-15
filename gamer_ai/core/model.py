# -*- coding: utf-8 -*-
"""模型构建 —— 训练端与部署端必须共用这一份，否则输入通道对不上。"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision import models


def build_model(in_frames: int, num_classes: int, pretrained: bool = True,
                weights_path: str = None):
    """EfficientNet-B0 + 首层卷积扩展到 3*K 通道（早融合，用 ImageNet 权重平均初始化）。

    in_frames 是喂给模型的连续帧数，输入张量形状 (3*K, H, W)。
    """
    if weights_path and os.path.isfile(weights_path):
        m = models.efficientnet_b0(weights=None)
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        m.load_state_dict(sd)
    else:
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)

    old = m.features[0][0]                      # Conv2d(3, 32, 3, stride=2)
    new = nn.Conv2d(3 * in_frames, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(1, in_frames, 1, 1) / in_frames)
    m.features[0][0] = new
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return m
