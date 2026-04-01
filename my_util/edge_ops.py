import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelPatchScorer(nn.Module):
    # 基于 Sobel 算子的 Patch 边缘强度打分
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        # 1 定义卷积核
        # 水平方向，检测垂直边缘
        # 维度为 [out_channel, in_channel, kernel_H, kernel_W] 即 [1, 1, 3, 3]
        sobel_x = torch.tensor([[-1.,  0.,  1.],
                                [-2.,  0.,  2.],
                                [-1.,  0.,  1.]]).view(1, 1, 3, 3)

        # 垂直方向，检测水平边缘
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        # 2 注册为 Buffer，使用 register_buffer 而不是 nn.Parameter，因为不需要更新这些权重
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # 3 定义RGB转灰度的权重 (公式: Y = 0.299R + 0.587G + 0.114B)
        grayscale_weight = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.register_buffer('grayscale_weight', grayscale_weight)

    def forward(self, imgs):
        #前向传播计算边缘得分
        """
        :param imgs: [B, 3, H, W] 原始高分辨率图像
        :return: [B, L] 每个 Patch 的平均边缘强度 (L = H*W / P^2)
        """
        # 1 转灰度 输出维度[B, 1, H, W]
        gray_imgs = (imgs * self.grayscale_weight).sum(dim=1, keepdim=True)

        # 2 边缘提取
        grad_x = F.conv2d(gray_imgs, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray_imgs, self.sobel_y, padding=1)

        # 3 边缘强度=梯度的总幅值 公式: G = sqrt(Gx^2 + Gy^2)
        # 加上1e-6防止在平坦区域(Gx=Gy=0)强行开方导致反向传播时出现NaN梯度
        edge_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

        # 4 池化，从像素降维到Patch 输出维度[B, 1, H/P, W/P]
        pooled_edges = F.avg_pool2d(edge_magnitude, kernel_size=self.patch_size, stride=self.patch_size)

        # 5 展平张量，对齐的序列长度 L
        # [B, 1, H/P, W/P] -> [B, (H/P * W/P)]=[B, L]
        patch_scores = pooled_edges.flatten(1)

        return patch_scores