# 分割任务的数据增强与读取

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
# 导入 torchvision 新版的 v2 API
from torchvision.transforms import v2
from torchvision import tv_tensors

class OxfordPetSegmentation(Dataset):
    def __init__(self, root, is_train=True, img_size=224):
        """
        root: 数据集的根目录
        """
        super().__init__()
        self.is_train = is_train
        self.img_size = img_size

        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")

        # 读取训练/测试划分文件
        split_file = "trainval.txt" if is_train else "test.txt"
        split_path = os.path.join(root, "annotations", split_file)

        self.filenames = []
        with open(split_path, 'r') as f:
            for line in f:
                name = line.strip().split()[0]
                self.filenames.append(name)

        # ==========================================
        # 构建联合数据增强 (Transforms V2)
        # ==========================================
        if is_train:
            self.transforms = v2.Compose([
                # RandomResizedCrop 在分割中容易把物体切没，所以一般用 Resize + 稍微的 Crop
                v2.Resize((img_size, img_size), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                # 只有图片需要归一化，Mask 不能归一化
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 验证集不翻转，直接 Resize
            self.transforms = v2.Compose([
                v2.Resize((img_size, img_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # 构建路径并读取
        img_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir, name + ".png")

        image = Image.open(img_path).convert("RGB")
        # Mask 必须是 'L' 模式（单通道灰度图），里面存的是类别索引 1, 2, 3
        mask = Image.open(mask_path).convert("L")

        # 将 Mask 显式包裹为 Mask 对象
        mask = tv_tensors.Mask(mask)

        # 同时传入图片和 Mask，它们会经历完全一样的随机变换
        image, mask = self.transforms(image, mask)

        # 返回结果 (这里的 mask 返回出去后，在 engine 里再减 1 变成 0,1,2)
        return image, mask

# ==========================================
# 辅助函数：暴露给 main_segmentation.py 调用
# ==========================================
def build_dataset(is_train, args):
    # 逻辑判断，增强代码的严谨性
    if args.dataset == 'pet_seg':
        dataset = OxfordPetSegmentation(
            root=args.data_path,
            is_train=is_train,
            img_size=args.input_size
        )
    else:
        # 留个报错出口，防止以后加了新数据没反应
        raise ValueError(f"Unknown dataset: {args.dataset}. Currently only 'pet_seg' is supported.")

    print(f"Dataset created: {args.dataset}, Split: {'train' if is_train else 'test'}, Size: {len(dataset)}")
    return dataset