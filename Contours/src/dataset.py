import os
from typing import override
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch._dynamo.variables import base
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BSDS500Dataset(Dataset):
    def __init__(self, data_dir, split='train', transform = None):
        """
        Custom PyTorch Dataset for BSDS500 to load RGB images and .mat edge annotations.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # defining paths based on BSDS500 standard folder layout
        self.image_dir = os.path.join(data_dir, "BSR", "images" ,split)
        self.groundTruth_dir = os.path.join(data_dir, "BSR", "groundTruth", split)

        # get all images filenames
        if os.path.exists(self.image_dir):
            self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
        else:
            self.images = []

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) :
        # Load image
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # load groundtruth edge map
        base_name = os.path.splitext(img_name)[0]
        mat_path = os.path.join(self.groundTruth_dir, base_name + ".mat")

        mat_data = sio.loadmat(mat_path)

        # BSDS500 .mat structure stores ground truth annotations in a nested struct
        # We grab the first human annotator's boundary map: groundTruth[0][0]['Boundaries'][0,0]
        # and convert it to a binary float tensor (0.0 or 1.0)

        gt_bindings = mat_data['groundTruth'][0][0]['Boundaries'][0,0]
        label = Image.fromarray((gt_bindings > 0).astype(np.uint8)*255)

        # Apply sychronized transforms
        if self.transform:
            # ensuring spatial alignment between Image and ground truth through transformations
            seed = torch.randint(0,2**32, (1,)).item()

            torch.manual_seed(seed)
            image = self.transform(image)

            torch.manual_seed(seed)
            label = self.transform(label)

        return image, label