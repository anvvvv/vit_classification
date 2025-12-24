# visualize_results.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from vit_pytorch.vit_with_decorr import ViT

# ======================
# 配置
# ======================
MODEL_PATH = "vit_butterfly_final.pth"
CSV_FILE = "data/Training_set.csv"
IMG_DIR = "data/train"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 75  # 蝴蝶类别数

# ======================
# 自定义数据集（同训练时）
# ======================
class ButterflyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label_str = self.annotations.iloc[idx]['label']
        label = self.label_to_idx[label_str]
        if self.transform:
            image = self.transform(image)
        return image, label, img_name  # 返回文件名用于可视化

# ======================
# 图像变换（必须和训练一致！）
# ======================
from torchvision import transforms as T
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ======================
# 加载数据 & 模型
# ======================
dataset = ButterflyDataset(CSV_FILE, IMG_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ViT(
    dim=128,
    num_classes=NUM_CLASSES,
    image_size=IMAGE_SIZE,
    patch_size=16,
    depth=6,
    heads=8,
    dim_head=64,
    mlp_dim=128 * 4,
    decorr_sample_frac=1.0
)

# 加载权重
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# ======================
# 推理：获取所有预测
# ======================
all_preds = []
all_labels = []
all_probs = []
all_img_names = []

with torch.no_grad():
    for images, labels, img_names in dataloader:
        logits, _ = model(images)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_img_names.extend(img_names)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# 计算准确率
acc = (all_preds == all_labels).mean()
print(f"Validation Accuracy: {acc:.2%}")

# ======================
# 1. 随机展示 9 个预测样本
# ======================
plt.figure(figsize=(12, 12))
indices = np.random.choice(len(dataset), size=9, replace=False)

for i, idx in enumerate(indices):
    img_name = dataset.annotations.iloc[idx]['filename']
    true_label = dataset.annotations.iloc[idx]['label']
    pred_label = dataset.idx_to_label[all_preds[idx]]
    prob = all_probs[idx][all_preds[idx]]

    img_path = os.path.join(IMG_DIR, img_name)
    img = Image.open(img_path).convert("RGB")

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    color = "green" if true_label == pred_label else "red"
    plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {prob:.2f}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig("result/sample_predictions.png", dpi=150)
plt.show()

# ======================
# 2. 混淆矩阵
# ======================
cm = confusion_matrix(all_labels, all_preds, normalize='true')
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("result/confusion_matrix.png", dpi=150)
plt.show()

# ======================
# 3. 错误案例分析：展示前 12 个错误预测
# ======================
error_indices = np.where(all_preds != all_labels)[0]
plt.figure(figsize=(15, 10))
num_show = min(12, len(error_indices))

for i in range(num_show):
    idx = error_indices[i]
    img_name = all_img_names[idx]
    true_label = dataset.idx_to_label[all_labels[idx]]
    pred_label = dataset.idx_to_label[all_preds[idx]]
    prob = all_probs[idx][all_preds[idx]]

    img_path = os.path.join(IMG_DIR, img_name)
    img = Image.open(img_path).convert("RGB")

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {prob:.2f}", color='red')
    plt.axis('off')

plt.tight_layout()
plt.savefig("result/error_cases.png", dpi=150)
plt.show()

print(f"\n✅ 可视化完成！图片已保存：")
print("- sample_predictions.png")
print("- confusion_matrix.png")
print("- error_cases.png")