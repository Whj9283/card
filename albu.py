import cv2
import numpy as np
import os
import random

# 输入文件夹和输出文件夹
image_folder = '/data2/zhangzifan/code_dir/2023-8-28-01/datasets/STARE/train/images'
mask_folder = '/data2/zhangzifan/code_dir/2023-8-28-01/datasets/STARE/train/mask'
output_image_folder = '/data2/zhangzifan/code_dir/2023-8-28-01/datasets/STARE/train/images1'
output_mask_folder = '/data2/zhangzifan/code_dir/2023-8-28-01/datasets/STARE/train/mask1'

# 创建输出文件夹
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# 获取image文件夹中的所有文件
image_files = os.listdir(image_folder)

# 定义增强操作函数
def augment(image, mask):
    # 随机旋转
    if random.random() > 0.5:
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    # 随机镜像
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # 随机裁剪
    if random.random() > 0.5:
        crop_size = (random.randint(200, image.shape[0]), random.randint(200, image.shape[1]))
        x = random.randint(0, image.shape[1] - crop_size[1])
        y = random.randint(0, image.shape[0] - crop_size[0])
        image = image[y:y+crop_size[0], x:x+crop_size[1]]
        mask = mask[y:y+crop_size[0], x:x+crop_size[1]]

    # 随机pad
    if random.random() > 0.5:
        pad_height = random.randint(0, 100)
        pad_width = random.randint(0, 100)
        image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image, mask

# 遍历image文件夹中的图像
for image_file in image_files:
    # 读取图像和相应的mask
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, image_file)
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 执行增强操作并保存结果
    for i in range(50):
        augmented_image, augmented_mask = augment(image.copy(), mask.copy())

        # 保存增强后的图像和mask
        output_image_path = os.path.join(output_image_folder, f"{image_file.split('.')[0]}_{i}.png")
        output_mask_path = os.path.join(output_mask_folder, f"{image_file.split('.')[0]}_{i}.png")

        cv2.imwrite(output_image_path, augmented_image)
        cv2.imwrite(output_mask_path, augmented_mask)

print("数据增强完成")
