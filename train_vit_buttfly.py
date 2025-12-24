# /// script
# dependencies = [
#   "accelerate",
#   "vit-pytorch",
#   "wandb",
#   "pandas",
#   "Pillow"
# ]
# ///

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os

import torchvision.transforms as T

# constants
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10
DECORR_LOSS_WEIGHT = 1e-1
TRACK_EXPERIMENT_ONLINE = True

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 自定义数据集类：从 CSV + 图像文件夹加载
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class ButterflyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): 路径到 Training_set.csv
            img_dir (str): 图像所在文件夹（如 './train'）
            transform (callable, optional): 可选的图像变换
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 构建标签到索引的映射（确保 label 是 int）
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # 确保是 RGB

        label_str = self.annotations.iloc[idx]['label']
        label = self.label_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 数据预处理 & 加载
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

transform = T.Compose([
    T.Resize((224, 224)),          # ViT 通常用 224x224（原图可能不是 32x32！）
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建数据集
dataset = ButterflyDataset(
    csv_file="data/Training_set.csv",
    img_dir="data/train",
    transform=transform
)

NUM_CLASSES = dataset.num_classes  # 应该是 75
print(f"Number of classes: {NUM_CLASSES}")

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 模型（修改 num_classes 和 image_size）
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

from vit_pytorch.vit_with_decorr import ViT

vit = ViT(
    dim=128,
    num_classes=NUM_CLASSES,       # ← 改为 75
    image_size=224,                # ← 改为 224（或你 Resize 的尺寸）
    patch_size=16,                 # ← 常见设置：224/16=14 patches per side
    depth=6,
    heads=8,
    dim_head=64,
    mlp_dim=128 * 4,
    decorr_sample_frac=1.0
)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 优化器 & Accelerate
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

from torch.optim import Adam
optim = Adam(vit.parameters(), lr=LEARNING_RATE)

from accelerate import Accelerator
accelerator = Accelerator()

vit, optim, dataloader = accelerator.prepare(vit, optim, dataloader)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# WandB 初始化
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

import wandb
wandb.init(
    project='vit-butterfly',
    mode='online' if TRACK_EXPERIMENT_ONLINE else 'disabled',
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "num_classes": NUM_CLASSES,
        "image_size": 224,
        "patch_size": 16,
        "model_dim": 128
    }
)
wandb.run.name = 'butterfly-vit-base'

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 训练循环
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

for epoch in range(EPOCHS):
    vit.train()
    for batch_idx, (images, labels) in enumerate(dataloader):
        logits, decorr_aux_loss = vit(images)
        loss = F.cross_entropy(logits, labels)

        total_loss = loss + decorr_aux_loss * DECORR_LOSS_WEIGHT

        accelerator.backward(total_loss)
        optim.step()
        optim.zero_grad()

        # 日志（每 50 步记录一次，避免太频繁）
        if batch_idx % 50 == 0:
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "decorr_loss": decorr_aux_loss.item(),
                "total_loss": total_loss.item()
            })
            accelerator.print(f"Epoch {epoch}, Batch {batch_idx}: "
                              f"Loss={loss.item():.4f}, Decorr={decorr_aux_loss.item():.4f}")

# 保存模型（可选）
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(vit)
    torch.save(unwrapped_model.state_dict(), "vit_butterfly_final.pth")
    wandb.save("vit_butterfly_final.pth")

wandb.finish()