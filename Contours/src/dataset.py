import os
import random
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class BSDS500Dataset(Dataset):
    def __init__(self, data_dir, split='train', transform = None, image_size=(256,256)):
        """
        Custom PyTorch Dataset for BSDS500 to load RGB images and .mat edge annotations.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.augment = (split == 'train')

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

    def _load_consensus_label(self, mat_path):
        mat_data = sio.loadmat(mat_path)
        annotators = mat_data['groundTruth'][0]
        num_annotators = len(annotators)

        vote_sum = None
        for i in range(num_annotators):
            boundaries = annotators[i]['Boundaries'][0, 0]
            try:
                binary = (boundaries > 0).astype(np.uint8)
            except Exception as e:
                raise RuntimeError(
                    f"Failed on {mat_path}, annotator {i}: "
                    f"type={type(boundaries)}, "
                    f"shape={getattr(boundaries, 'shape', None)}, "
                    f"dtype={getattr(boundaries, 'dtype', None)}"
                ) from e
            vote_sum = binary if vote_sum is None else vote_sum + binary

        consensus_threshold = min(3, num_annotators)
        return (vote_sum >= consensus_threshold).astype(np.uint8) * 255

    def _augment(self, image, label):
        # horizontal flips
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # vertical flip
        if random.random() < 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # rotation
        angle = random.uniform(-30,30)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=[0])
        label = TF.rotate(label, angle, interpolation=InterpolationMode.NEAREST, fill=[0])

        return image, label

    def __getitem__(self, index) :
        # Load image
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # load groundtruth edge map
        base_name = os.path.splitext(img_name)[0]
        mat_path = os.path.join(self.groundTruth_dir, base_name + ".mat")
        label = Image.fromarray(self._load_consensus_label(mat_path))

        if self.augment: 
            image, label = self._augment(image, label)

        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        label = TF.resize(label, self.image_size, interpolation=InterpolationMode.NEAREST)
        
        return TF.to_tensor(image), TF.to_tensor(label)