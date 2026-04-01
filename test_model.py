# AI生成，只用于测试model中维度在传播过程中是否对齐
import torch
from my_vit import MyVit
from my_mae import MyMaskedAutoencoder

def test_mae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # 1. 实例化你的模型 (使用较小的配置以快速测试)
    encoder = MyVit(img_size=224, patch_size=16, embed_dim=384, depth=4, num_heads=6)
    model = MyMaskedAutoencoder(encoder=encoder, decoder_dim=256, decoder_depth=2, decoder_num_heads=4)
    model.to(device)

    # 2. 伪造一个 Batch 的图片 [B, C, H, W]
    dummy_img = torch.randn(2, 3, 224, 224).to(device)

    # 3. 前向传播
    print("Running Forward Pass...")
    loss, mean, var, pred, mask = model(dummy_img)
    print(f"Forward Success! Loss: {loss.item():.4f}")
    print(f"Pred shape: {pred.shape}, Mask shape: {mask.shape}")

    # 4. 反向传播测试 (验证计算图是否断裂)
    print("Running Backward Pass...")
    loss.backward()
    print("Backward Success! No gradient breakage.")

if __name__ == "__main__":
    test_mae()