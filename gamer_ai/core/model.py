# -*- coding: utf-8 -*-
"""模型构建 —— 训练端与部署端必须共用这一份，否则输入通道对不上。"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision import models


def build_model(in_frames: int, num_classes: int, pretrained: bool = True,
                weights_path: str = None, in_channels: int = None):
    """EfficientNet-B0 + 首层卷积扩展到 in_channels（早融合，用 ImageNet 权重平均初始化）。

    in_frames 是喂给模型的连续帧数，输入张量默认形状 (3*K, H, W)。
    in_channels 可以不等于 3*K —— 2026-11-05 起会用 (3*K + 1)：
    最后那个通道是**速度标量**（整张图填同一个值），让模型知道此刻跑多快。
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
    base_n = 3 * in_frames
    in_ch = int(in_channels) if in_channels else base_n
    new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=False)
    with torch.no_grad():
        # 前 3*K 个通道：ImageNet 权重按帧数平均复制（和原来一样，收敛快）
        new.weight.zero_()
        new.weight[:, :base_n].copy_(old.weight.repeat(1, in_frames, 1, 1) / in_frames)
        # 额外通道（速度标量）初始为 0：一开始完全不影响，训练中自己学权重
    m.features[0][0] = new
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return m
