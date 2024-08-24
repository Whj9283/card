from PIL import Image
import os
import cv2


png_folder = '/data2/zhangzifan/code_dir/2023-10-25-01/datasets/test/images'
save_folder = '/data2/zhangzifan/code_dir/2023-10-25-01/datasets/test/images1024'

# 获取JPG文件夹中所有文件的列表
files_list = os.listdir(png_folder)

# 遍历JPG文件夹中的所有文件
for file_name in files_list:
    image = cv2.imread(os.path.join(png_folder, file_name))
    image = cv2.resize(image, (1024, 1024))
    cv2.imwrite(os.path.join(save_folder, file_name), image)


