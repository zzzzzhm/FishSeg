import os
import random  # 用于生成随机噪声大小
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# 定义数据集路径
dataset_folder = 'dataset/Fish_Dataset/Fish_Dataset'
output_folder = 'fish_dateset/'
background_folder = 'back ground'


# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 图像转换函数
transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def add_gaussian_noise(tensor, mean=0, std=0.1):
    """
    给图片张量添加高斯噪声
    :param tensor: 输入的图片张量 (C, H, W)
    :param mean: 高斯噪声均值
    :param std: 高斯噪声标准差
    :return: 添加噪声后的图片张量
    """
    noise = torch.randn_like(tensor) * std + mean  # 生成与输入张量形状一致的噪声
    noisy_tensor = tensor + noise  # 添加噪声
    return torch.clamp(noisy_tensor, 0, 1)  # 限制范围到 [0, 1]

def place_fish_on_background(background, fish_img, mask, position):
    """
    将鱼图片放置到背景上的指定位置
    :param background: 背景图片张量 (C, H, W)
    :param fish_img: 鱼图片张量 (C, h, w)
    :param mask: 掩膜张量 (1, h, w)
    :param position: (x, y) 左上角坐标
    :return: 放置后的图片张量
    """
    x, y = position
    h, w = fish_img.shape[1:]  # 鱼图片的高度和宽度
    background[:, y:y+h, x:x+w] = (
        background[:, y:y+h, x:x+w] * (1 - mask) +
        fish_img * mask
    )
    return background


# 遍历整个数据集的子文件夹
for fish_type in os.listdir(dataset_folder):
    fish_type_path = os.path.join(dataset_folder, fish_type)

    # 原始图片文件夹和掩膜文件夹路径
    original_folder = os.path.join(fish_type_path, fish_type)  # 原始图片文件夹
    mask_folder = os.path.join(fish_type_path, f"{fish_type} GT")  # 掩膜文件夹

    # 检查这两个文件夹是否存在
    if not os.path.isdir(original_folder) or not os.path.isdir(mask_folder):
        print(f"Skipping {fish_type}, missing original or mask folder...")
        continue

    # 创建当前鱼类的输出文件夹及其掩膜子文件夹
    fish_output_folder = os.path.join(output_folder, fish_type, fish_type)
    fish_gt_output_folder = os.path.join(output_folder, fish_type, f"{fish_type} GT")
    os.makedirs(fish_output_folder, exist_ok=True)
    os.makedirs(fish_gt_output_folder, exist_ok=True)

    # 遍历原始图片
    for original_name in tqdm(os.listdir(original_folder), desc=f"Processing {fish_type}"):
        original_path = os.path.join(original_folder, original_name)
        mask_path = os.path.join(mask_folder, original_name)

        # 检查掩膜文件是否存在
        if not os.path.exists(mask_path):
            print(f"Mask not found for {original_name}, skipping...")
            continue

        # 加载图片
        original_img = transform(Image.open(original_path))  # 原始图片
        mask_img = transform(Image.open(mask_path).convert("L"))  # 掩膜图片

        # 将掩膜二值化
        binary_mask = (mask_img > 0.5).float()  # 二值化掩膜

        # 应用掩膜抠出鱼
        fish_img = original_img * binary_mask

         # 随机选择一张背景图
        background_name = random.choice(os.listdir(background_folder))
        background_path = os.path.join(background_folder, background_name)
        background_img = transform(Image.open(background_path))  # 背景图片 (C, H, W)

        # 缩小鱼图片和掩膜
        scale_factor = random.uniform(0.75, 0.85)  # 随机缩小比例
        new_size = (int(fish_img.shape[2] * scale_factor), int(fish_img.shape[1] * scale_factor))
        fish_img_resized = transforms.Resize(new_size)(to_pil(fish_img))
        mask_resized = transforms.Resize(new_size)(to_pil(binary_mask))
        fish_img = transform(fish_img_resized)
        binary_mask = transform(mask_resized)

        # 随机生成鱼在背景上的位置
        bg_height, bg_width = background_img.shape[1:]  # 背景图高度和宽度
        fish_height, fish_width = fish_img.shape[1:]  # 鱼图片的高度和宽度
        x = random.randint(0, bg_width - fish_width)
        y = random.randint(0, bg_height - fish_height)

        # 合成图片
        composite_img = place_fish_on_background(background_img.clone(), fish_img, binary_mask, (x, y))


        # 生成随机噪声强度（标准差）
        random_std = random.uniform(0.01, 0.2)  # 随机生成 0.01 到 0.2 的标准差
        noisy_composite_img = add_gaussian_noise(composite_img, mean=0, std=random.uniform(0.01, 0.2))


        # 转换为图片格式并保存
        fish_img_pil = transforms.ToPILImage()(noisy_composite_img)  # 带噪声的鱼图片
        fish_mask_pil = transforms.ToPILImage()(binary_mask)  # 掩膜图片

        # 保存带噪声的鱼图片
        fish_output_path = os.path.join(fish_output_folder, original_name)
        fish_img_pil.save(fish_output_path)

        # 保存掩膜图片
        fish_gt_output_path = os.path.join(fish_gt_output_folder, original_name)
        fish_mask_pil.save(fish_gt_output_path)

print("所有鱼类图片已添加随机高斯噪声并按结构保存到输出文件夹！")
