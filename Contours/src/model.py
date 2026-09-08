import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class HED(nn.Module):
    def __init__(self) -> None:
        super(HED, self).__init__()

        # Pre-trained VGG-16 features as the backbone for contour extraction
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        features = list(vgg.features.children())

        # Split VGG into the 5 stages that sit between the pooling layers
        self.slice1 = nn.Sequential(*features[:4])      # Conv1
        self.slice2 = nn.Sequential(*features[4:9])     # Conv2
        self.slice3 = nn.Sequential(*features[9:16])    # Conv3
        self.slice4 = nn.Sequential(*features[16:23])   # Conv4
        self.slice5 = nn.Sequential(*features[23:30])   # Conv5

        # Side-output convolutions, shrinking each stage's channels down to 1
        self.score_ds1 = nn.Conv2d(64, 1, kernel_size=1)
        self.score_ds2 = nn.Conv2d(128, 1, kernel_size=1)
        self.score_ds3 = nn.Conv2d(256, 1, kernel_size=1)
        self.score_ds4 = nn.Conv2d(512, 1, kernel_size=1)
        self.score_ds5 = nn.Conv2d(512, 1, kernel_size=1)

        # Final fusion layer combining the 5 side outputs into one prediction
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

        self._init_heads()

    def _init_heads(self):
        """
        The new layers start out random, which makes the first predictions noise.
        Small side-output weights mean each stage starts near "no opinion", and
        equal fusion weights (1/5 each) mean the fused output starts as a plain
        average of the five stages instead of an arbitrary blend.
        """
        for layer in [self.score_ds1, self.score_ds2, self.score_ds3,
                      self.score_ds4, self.score_ds5]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0)

        nn.init.constant_(self.fuse.weight, 0.2)
        nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x):
        input_shape = x.shape[2:]

        # Stage 1
        h1 = self.slice1(x)
        d1 = self.score_ds1(h1)
        d1 = F.interpolate(d1, size=input_shape, mode="bilinear", align_corners=False)

        # Stage 2
        h2 = self.slice2(h1)
        d2 = self.score_ds2(h2)
        d2 = F.interpolate(d2, size=input_shape, mode="bilinear", align_corners=False)

        # Stage 3
        h3 = self.slice3(h2)
        d3 = self.score_ds3(h3)
        d3 = F.interpolate(d3, size=input_shape, mode="bilinear", align_corners=False)

        # Stage 4
        h4 = self.slice4(h3)
        d4 = self.score_ds4(h4)
        d4 = F.interpolate(d4, size=input_shape, mode="bilinear", align_corners=False)

        # Stage 5
        h5 = self.slice5(h4)
        d5 = self.score_ds5(h5)
        d5 = F.interpolate(d5, size=input_shape, mode="bilinear", align_corners=False)

        # Concatenate the side outputs and fuse them
        fuse = self.fuse(torch.cat([d1, d2, d3, d4, d5], dim=1))

        # NOTE: these are raw logits. Sigmoid is applied by the loss during
        # training and by the inference script at prediction time -- never here.
        return [d1, d2, d3, d4, d5, fuse]