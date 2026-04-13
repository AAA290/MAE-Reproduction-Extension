# 分割任务encoder+decoder
import math
import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    # 轻量级渐进式分割解码器
    def __init__(self, embed_dim=768, num_classes=3):
        super().__init__()
        # 第一级放大：14x14 -> 56x56
        self.up1 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 使用双线性插值放大 4 倍，比起反卷积不容易产生棋盘格伪影
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        # 第二级放大：56x56 -> 224x224
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        # 最终投影：把 64 维压缩到 3 维（对应背景、宠物、边缘 3 个类别）
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.head(x)
        return x

class ViTSegmentation(nn.Module):
    # 完整的分割模型 (Encoder + Decoder)
    def __init__(self, backbone, embed_dim=768, num_classes=3):
        """
        backbone: 预训练好的模型实例
        embed_dim: ViT Base 默认是 768
        num_classes: Oxford Pet 任务默认是 3
        """
        super().__init__()
        self.backbone = backbone

        self.backbone.head = None

        self.decoder = SimpleDecoder(embed_dim, num_classes)

    def forward(self, x):
        # 过 Backbone，拿到所有 token 的特征
        patches = self.backbone(x)

        # 空间重塑 (一维序列 -> 二维网格)
        B, N, C = patches.shape
        H = W = int(math.sqrt(N)) # N=196, 所以 H=W=14

        # permute: [B, 196, 768] -> [B, 768, 196]
        patches = patches.permute(0, 2, 1)
        # view: [B, 768, 196] -> [B, 768, 14, 14]
        patches = patches.view(B, C, H, W)

        # 送入解码器
        logits = self.decoder(patches) # 输出形状: [B, 3, 224, 224]

        return logits