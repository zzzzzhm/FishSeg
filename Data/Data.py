import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt  # 用于可视化
import torch.nn as nn
from PIL import Image

class FishDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []  # 存储所有图片路径
        self.masks = []   # 存储所有掩码路径
        self.transform = transform

        # 遍历所有类别文件夹
        for fish_type in os.listdir(root_dir):
            fish_dir = os.path.join(root_dir, fish_type)
            if os.path.isdir(fish_dir):
                images_dir = os.path.join(fish_dir, fish_type)
                masks_dir = os.path.join(fish_dir, fish_type + " GT")

                # 获取当前类别的所有图片和掩码路径
                image_files = sorted(os.listdir(images_dir))
                mask_files = sorted(os.listdir(masks_dir))

                # 确保图片和掩码一一对应
                for img_file, mask_file in zip(image_files, mask_files):
                    self.images.append(os.path.join(images_dir, img_file))
                    self.masks.append(os.path.join(masks_dir, mask_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图片和掩码
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # image = cv2.imread(img_path)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        

        if self.transform:
            # image = self.transform(image)
            # mask = self.transform(mask)
            image = self.transform(Image.open(img_path))  # 原始图片
            mask = self.transform(Image.open(mask_path).convert("L"))  # 掩膜图片

        # # # 返回图像和掩码
        # return torch.tensor(image, dtype=torch.float32), \
        #        torch.tensor(mask, dtype=torch.long)
        return image, \
               mask

# 数据增强
train_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((1024, 1024))
    transforms.Resize((256, 256)),
])

# 数据集路径
root_dir = "fish_dateset/noise"

# 加载数据集
dataset = FishDataset(root_dir, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# # # 检查数据并可视化
# for images, masks in dataloader:
#     print(f"Batch of images: {images.shape}, Batch of masks: {masks.shape}")
#     fish_img_pil = transforms.ToPILImage()(images[0])  # 带噪声的鱼图片
#     # fish_mask_pil = transforms.ToPILImage()(mask)  # 掩膜图片
#     print(fish_img_pil)
#     fish_img_pil.show()
#     masks = transforms.ToPILImage()(masks[0])  # 带噪声的鱼图片
#     # fish_mask_pil = transforms.ToPILImage()(mask)  # 掩膜图片
#     print(masks)
#     masks.show()

#     break  # 只显示一个 batch 的数据

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512)
        )
        self.middle = self.conv_block(512, 1024)
        # self.decoder = nn.Sequential(
        #     self.upconv_block(1024, 512),
        #     self.upconv_block(512, 256),
        #     self.upconv_block(256, 128),
        #     self.upconv_block(128, 64)
        # )
        self.decoder = nn.Sequential(
        self.upconv_block(1536, 512),  # 修改输入通道数为 1536（1024 + 512）
        self.upconv_block(768, 256),   # 修改为 768（512 + 256）
        self.upconv_block(384, 128),   # 修改为 384（256 + 128）
        self.upconv_block(192, 64)     # 修改为 192（128 + 64）
)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        print(f"Batch of images: {x.shape}")
        enc1 = self.encoder[0](x)
        # print(f"Batch of images: {x.shape}")
        enc2 = self.encoder[1](enc1)
        # print(f"Batch of images: {x.shape}")
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        # enc4 = self.encoder(x)
        print(f"Batch of images: {enc4.shape}")

        middle = self.middle(enc4)
        print(f"Batch of images: {middle.shape}")

        dec4 = self.decoder[0](torch.cat([self.center_crop(enc4, middle), enc4], dim=1))
        dec3 = self.decoder[1](torch.cat([self.center_crop(enc3, dec4), enc3], dim=1))
        dec2 = self.decoder[2](torch.cat([self.center_crop(enc2, dec3), enc2], dim=1))
        dec1 = self.decoder[3](torch.cat([self.center_crop(enc1, dec2), enc1], dim=1))

        return self.final(self.center_crop(x, dec1))
    
    @staticmethod
    def center_crop(target_tensor, tensor):
        """裁剪 tensor 使其大小与 target_tensor 一致"""
        _, _, H, W = target_tensor.size()
        return tensor[:, :, :H, :W]  # 假设过大的部分都在右下角


import torch.optim as optim

# 模型实例化
model = UNet(in_channels=3, out_channels=1)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类问题的损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)


from tqdm import tqdm  # 导入 tqdm

# 训练过程
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # 二分类分割任务
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    # 使用 tqdm 包裹 DataLoader
    loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)
        print(f"Batch of images: {outputs.shape}")
        loss = criterion(outputs, masks.float())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())  # 在进度条上显示当前 loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")



