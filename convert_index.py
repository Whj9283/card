import os
import cv2
import multiprocessing
from tqdm import tqdm


# 定义图像转换函数
def convert_index(img):
    for row in range(img.shape[0]):  # 遍历高
        for col in range(img.shape[1]):  # 遍历宽
            # 背景
            if img[row, col, 0] == 255 and img[row, col, 1] == 255 and img[row, col, 2] == 255:
                img[row, col] = [1, 1, 1]
            else:
                img[row, col] = [0, 0, 0]
    return img


# 定义进程函数
def process_file(file, mask_path, save_path):
    try:
        img = cv2.imread(os.path.join(mask_path, file))
        image = convert_index(img)
        cv2.imwrite(os.path.join(save_path, file), image)
    except:
        print(file)


if __name__ == '__main__':
    mask_path = "/home/zhangzifan/MaintoCode/2023-8-28-01/datasets/DRIVE/train/mask/"
    save_path = "/home/zhangzifan/MaintoCode/2023-8-28-01/datasets/DRIVE/train/labels/"
    file_list = os.listdir(mask_path)

    # 设置进程池大小，这里设置为4
    pool = multiprocessing.Pool(processes=10)

    # 使用进程池处理文件
    for file in tqdm(file_list):
        pool.apply_async(process_file, args=(file, mask_path, save_path))

    # 关闭进程池，等待所有进程完成
    pool.close()
    pool.join()
