import os
import random  # 用于生成随机噪声大小
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# 定义数据集路径
original_path = 'fish_dateset/bright/Black Sea Sprat/Black Sea Sprat/00032.png'
mask_path = 'fish_dateset/bright/Black Sea Sprat/Black Sea Sprat GT/00032.png'
fish_output_path = "temple2.png"

# 图像转换函数
transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()



# 加载图片
original_img = transform(Image.open(original_path))  # 原始图片
mask_img = transform(Image.open(mask_path).convert("L"))  # 掩膜图片

# 将掩膜二值化
binary_mask = (mask_img > 0.5).float()  # 二值化掩膜

# 应用掩膜抠出鱼
fish_img = original_img * binary_mask



# 转换为图片格式并保存
fish_img_pil = transforms.ToPILImage()(fish_img)  # 带噪声的鱼图片


fish_img_pil.save(fish_output_path)