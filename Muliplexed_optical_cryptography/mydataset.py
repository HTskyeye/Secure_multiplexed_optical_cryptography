"""
Read plaintext from dataset

DATE: 2025/11/6
"""
import os
from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
from cv2 import cv2
import numpy as np
__all__ = [cv2]


class MyDataset(Dataset):
    """
    DATASET LOADER

    init: image_path, size, patch_size
    input: item
    output: changed_image, length
    """
    def __init__(self, changed_image_path, patch_size, size):
        self.changed_image_path = changed_image_path
        self.size = size
        self.patch_size = patch_size

    def __getitem__(self, item):
        changed_images = os.listdir(self.changed_image_path)

        changed_images.sort()

        changed_image_path = os.path.join(str(Path(self.changed_image_path)), changed_images[item])

        changed_image = cv2.imread(changed_image_path, cv2.IMREAD_GRAYSCALE)

        sp = changed_image.shape
        rows = sp[0]
        cols = sp[1]
        if rows >= cols:
            shorter = cols
        else:
            shorter = rows
        cropped = changed_image[0:shorter, (cols-shorter)//2:(cols+shorter)//2]

        changed_image = cv2.resize(cropped, self.patch_size)

        changed_image = torch.from_numpy(np.expand_dims(changed_image, axis=0)) / 255

        return changed_image, item

    def __len__(self):
        return self.size
