import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 导入你写好的模型
from my_vit import MyVit
from my_mae import MyMaskedAutoencoder

def get_args_parser():
    parser = argparse.ArgumentParser('MAE Visualization Demo', add_help=False)
    parser.add_argument('--image_path', type=str, required=True, help='输入测试图片的路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='预训练好的模型权重路径 (.pth)')
    parser.add_argument('--save_path', type=str, default='vis_result.png', help='可视化结果的保存路径')
    parser.add_argument('--img_size', default=224, type=int, help='模型输入的图片大小')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch 大小')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='掩码率')
    parser.add_argument('--mask_type', default='random', type=str, help='掩码策略: random 或 edge')
    return parser

def unpatchify(x, patch_size=16):
    """
    将模型输出的 patch 序列还原成完整的 2D 图像张量
    输入 x: [Batch, Num_Patches, Patch_Size**2 * Channels]
    输出 imgs: [Batch, Channels, H, W]
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1], "Patch 数量必须是一个完全平方数"

    # 1. Reshape 回 [B, H, W, P, P, 3]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    # 2. 调整维度顺序到 [B, 3, H, P, W, P]
    x = torch.einsum('nhwpqc->nchpwq', x)
    # 3. 压平得到最终的图像 [B, 3, H*P, W*P]
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def prepare_model(checkpoint_path, args, device='cuda'):
    """加载模型结构与预训练权重"""
    # 1. 实例化 Encoder 和 MAE
    encoder = MyVit(img_size=args.img_size, patch_size=args.patch_size, embed_dim=768, depth=12, num_heads=12)
    model = MyMaskedAutoencoder(
        encoder=encoder,
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio,
        decoder_dim=512, decoder_depth=8, decoder_num_heads=16
    )

    # 2. 加载权重 (处理 DDP 训练可能带来的 'module.' 前缀)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded checkpoint with msg: {msg}")

    model.to(device)
    model.eval() # 开启评估模式，关闭 Dropout 等
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("1. 正在加载模型...")
    model = prepare_model(args.checkpoint, args, device)

    print("2. 正在处理图像...")
    # 读取图像并将其 Resize 到 224x224
    img = Image.open(args.image_path).convert('RGB')
    img = img.resize((args.img_size, args.img_size))

    # 转换为 Tensor，范围调整到 [0, 1]，增加 Batch 维度，并转移到 GPU
    x = torch.tensor(np.array(img)) / 255.0
    x = x.unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device) # [1, 3, 224, 224]

    print("3. 正在进行前向推理...")
    with torch.no_grad():
        # 如果你的 mask_type 是 edge，记得把 imgs(原始图片) 传进去
        loss, mean, var, pred, mask = model(x)

    print("4. 正在还原与反标准化图像...")
    # 🌟 核心逻辑 1：反标准化 (Anti-normalization)
    # 因为预训练时开启了 norm_pix_loss=True，预测出的 pred 是无色彩分布的标准化数据
    pred = pred * (var + 1.e-6)**.5 + mean

    # 将模型输出的序列变为图像 [1, 3, 224, 224]
    pred_img = unpatchify(pred, args.patch_size)
    pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()

    # 🌟 核心逻辑 2：制作 Mask 覆盖图
    # mask 的维度是 [1, 196]，1 表示被盖住的，0 表示保留的
    # 将 mask 扩展到像素级别 [1, 196, 256] -> unpatchify
    mask_pixels = mask.unsqueeze(-1).repeat(1, 1, args.patch_size**2 * 3)
    mask_img_tensor = unpatchify(mask_pixels, args.patch_size) # [1, 3, 224, 224]
    mask_img_np = mask_img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    # 原始图像
    original_img = x.squeeze().permute(1, 2, 0).cpu().numpy()

    # 被掩码的图像 (保留部分 + 灰色遮挡)
    # 当 mask_img_np == 1 时，用 0.5 (灰色) 替代原图像素
    im_masked = original_img * (1 - mask_img_np) + 0.5 * mask_img_np

    # 最终拼接出来的重建图 (保留部分用原图，预测部分用 pred)
    im_paste = original_img * (1 - mask_img_np) + pred_img * mask_img_np

    # 裁切数值到 [0, 1] 避免 Matplotlib 报错
    original_img = np.clip(original_img, 0, 1)
    im_masked = np.clip(im_masked, 0, 1)
    im_paste = np.clip(im_paste, 0, 1)

    print("5. 正在绘制可视化结果...")
    plt.rcParams['figure.figsize'] = [12, 4]
    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(im_masked)
    axes[1].set_title(f"Masked ({args.mask_ratio * 100}%) - {args.mask_type.capitalize()}")
    axes[1].axis('off')

    axes[2].imshow(im_paste)
    axes[2].set_title("Reconstructed")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches='tight', dpi=300)
    print(f"✅ 可视化完成！结果已保存至 {args.save_path}")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)