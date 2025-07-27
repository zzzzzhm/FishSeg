import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


average = 85.0
transform = transforms.ToTensor()
output_path = "temple.png"
original_path = "fish_dateset/noise/Black Sea Sprat/Black Sea Sprat/00004.png"


# 调整亮度
def adjust_brightness_tensor(tensor, factor):
    """
    调整 PyTorch 张量的亮度
    :param tensor: 图片张量 (C, H, W)
    :param factor: 亮度调整因子 (>1 增亮，<1 变暗)
    :return: 调整后的张量
    """
    return torch.clamp(tensor * factor, 0, 1)



def detect_brightness(image):
    # 如果是彩色图像，将其转换为灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # 计算平均亮度
    avg_brightness = np.mean(gray_image)
    
    # # 设置亮度阈值
    # if avg_brightness > 127:  # 假设 127 为亮度中值
    #     return "Bright"
    # else:
    #     return "Dark"
    return avg_brightness


# def detect_gaussian_noise_frequency(image):
#     # 如果是彩色图像，转换为灰度图
#     if len(image.shape) == 3:
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray_image = image

#     # 傅里叶变换
#     f = np.fft.fft2(gray_image)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # 防止log(0)

#     # 分析高频区域
#     rows, cols = gray_image.shape
#     center_row, center_col = rows // 2, cols // 2
#     high_freq_energy = np.sum(magnitude_spectrum[:center_row - 10, :]) + \
#                        np.sum(magnitude_spectrum[center_row + 10:, :]) + \
#                        np.sum(magnitude_spectrum[:, :center_col - 10]) + \
#                        np.sum(magnitude_spectrum[:, center_col + 10:])

#     # 设置阈值判断噪声是否存在
#     if high_freq_energy > 1e9:  # 假设高频能量阈值
#         return "Gaussian noise detected"
#     else:
#         return "No significant Gaussian noise"
    

def detect_gaussian_noise_residual(image):
    # 如果是彩色图像，转换为灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 应用高斯平滑
    smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

    # 计算残差
    residual = gray_image.astype(np.float32) - smoothed_image.astype(np.float32)

    # 分析残差分布（标准差是否较大）
    residual_std = np.std(residual)
    if residual_std > 20:  # 根据实验调整阈值
        return "Gaussian noise detected", smoothed_image
    else:
        return "No significant Gaussian noise", smoothed_image


def denoise_image_rgb(image):
    """
    对彩色图像的每个通道进行高斯滤波，保留颜色
    Args:
        image (numpy.ndarray): 输入彩色图像（BGR 格式）

    Returns:
        numpy.ndarray: 去噪后的彩色图像
    """
    # 拆分图像的 BGR 通道
    b_channel, g_channel, r_channel = cv2.split(image)
    
    # 对每个通道进行高斯滤波
    b_denoised = cv2.GaussianBlur(b_channel, (5, 5), 5)
    g_denoised = cv2.GaussianBlur(g_channel, (5, 5), 5)
    r_denoised = cv2.GaussianBlur(r_channel, (5, 5), 5)
    
    # 合并通道
    denoised_image = cv2.merge([b_denoised, g_denoised, r_denoised])
    return denoised_image

def denoise_image_hsv(image):
    """
    在 HSV 空间中对图像进行去噪
    Args:
        image (numpy.ndarray): 输入彩色图像（BGR 格式）

    Returns:
        numpy.ndarray: 去噪后的彩色图像
    """
    # 转换为 HSV 空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 拆分 HSV 通道
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    
    # 对亮度通道（V）进行高斯滤波
    v_denoised = cv2.GaussianBlur(v_channel, (3, 3), 20)
    
    # 合并通道
    denoised_hsv = cv2.merge([h_channel, s_channel, v_denoised])
    
    # 转换回 BGR 格式
    denoised_image = cv2.cvtColor(denoised_hsv, cv2.COLOR_HSV2BGR)
    return denoised_image



# 示例：读取图像并检测高斯噪声
image = cv2.imread(original_path)
result, outimage = detect_gaussian_noise_residual(image)
print(result)
cv2.imwrite(output_path, denoise_image_rgb(image))
# outimage.show()













# # 示例：读取图像并检测亮度
# image = cv2.imread(original_path)
# brightness = detect_brightness(image)
# print(f"The image is {brightness}.")
# original_img = transform(Image.open(original_path))
# fish_img = adjust_brightness_tensor(original_img, factor=float(average/brightness))
# fish_img = transforms.ToPILImage()(fish_img)  # 掩膜图片
# fish_img.save(output_path)
# fish_img.show()
# print(f"The image is {detect_brightness(cv2.imread(output_path))}.")
