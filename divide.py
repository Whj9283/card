import os
import random

file_path = "./datasets/images"
train_path = "./datasets/train/images"
test_path = "./datasets/test/images"

count = 0
file_list = os.listdir(file_path)
random.shuffle(file_list)
for file in file_list:
    file = f"'{file}'"
    if count < int(len(file_list) * 0.8):
        os.system(f'cp {os.path.join(file_path, file)} {os.path.join(train_path, file)}')
        os.system(f'cp {os.path.join(file_path.replace("images", "labels"), file)} {os.path.join(train_path.replace("images", "labels"), file)}')
    else:
        os.system(f'cp {os.path.join(file_path, file)} {os.path.join(test_path, file)}')
        os.system(f'cp {os.path.join(file_path.replace("images", "labels"), file)} {os.path.join(test_path.replace("images", "labels"), file)}')
    count += 1