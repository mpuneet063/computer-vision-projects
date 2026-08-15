import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class HED(nn.Module):
    def __init__(self) -> None:
        super(HED, self).__init__()

        # load pre-trained VGG-16 features as backbone for contours extraction
        vgg = vgg16(weights=VGG16_Weights.DEFAULT) 
        features = list(vgg.features.children())

        # split VGG into 5 stages corresponding to the 5 pooling blocks
        self.slice1 = nn.Sequential(*features[:4])  #Conv1
        self.slice2 = nn.Sequential(*features[4:9])     #Conv2
        self.slice3 = nn.Sequential(*features[9:16])    #Conv3
        self.slice4 = nn.Sequential(*features[16:23])   #Conv4
        self.slice5 = nn.Sequential(*features[23:30])   #Conv5

        # side-output convolutions to shrink each stage's channels down to 1
        self.score_ds1 = nn.Conv2d(64, 1, kernel_size=1)
        self.score_ds2 = nn.Conv2d(128, 1, kernel_size=1)
        self.score_ds3 = nn.Conv2d(256, 1, kernel_size=1)
        self.score_ds4 = nn.Conv2d(512, 1, kernel_size=1)
        self.score_ds5 = nn.Conv2d(512, 1, kernel_size=1)

        # final fusion layer combining the 4 side-outputs into a single prediction
        self.fuse = nn.Conv2d(5,1, kernel_size=1)

        # unfreeze the vgg params
        for param in vgg.parameters():
            param.requires_grad = True

    def forward(self, x):
        input_shape = x.shape[2:]

        # stage 1
        h1 = self.slice1(x)
        d1 = self.score_ds1(h1)
        d1 = F.interpolate(d1, size=input_shape, mode = 'bilinear', align_corners=False)

        # stage 2
        h2 = self.slice2(h1)
        d2 = self.score_ds2(h2)
        d2 = F.interpolate(d2, size=input_shape, mode = 'bilinear', align_corners=False)

        # stage 3 
        h3 = self.slice3(h2)
        d3 = self.score_ds3(h3)
        d3 = F.interpolate(d3, size=input_shape, mode = 'bilinear', align_corners=False)

        # stage 4
        h4 = self.slice4(h3)
        d4 = self.score_ds4(h4)
        d4 = F.interpolate(d4, size=input_shape, mode = 'bilinear', align_corners=False)

        # stage 5
        h5 = self.slice5(h4)
        d5 = self.score_ds5(h5)
        d5 = F.interpolate(d5, size=input_shape, mode = 'bilinear', align_corners=False)

        # concatante side outputs and fuse them
        fuse = self.fuse(torch.cat([d1, d2, d3, d4, d5], dim=1))

        return [d1, d2, d3, d4, d5, fuse]