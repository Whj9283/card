import numpy as np
import cv2
import os


dir_path = "/data2/zhangzifan/code_dir/mouse_data/train_npz"
save_path = "/data2/zhangzifan/code_dir/mouse_data/train"
file_list = os.listdir("/data2/zhangzifan/code_dir/mouse_data/train_npz")


count = 1
for file in file_list:
    data = np.load(os.path.join(dir_path, file))
    cv2.imwrite(os.path.join(save_path, 'images', 'train_' + str(count) + '.png'), data['image'])
    cv2.imwrite(os.path.join(save_path, 'labels', 'train_' + str(count) + '.png'), data['label'])
    count += 1

