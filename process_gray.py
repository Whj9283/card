import mmcv
import os
import cv2
from PIL import Image
import numpy as np
import os.path as osp

root = "./datasets/maskNew/"
save = "./datasets/labelsNew/"
PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
for file in mmcv.scandir(osp.join(root), suffix='.png'):
    seg_map = cv2.imread(osp.join(root, file), cv2.IMREAD_GRAYSCALE)
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(PALETTE, dtype=np.uint8))
    seg_img.save(osp.join(save, file))
