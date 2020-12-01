# FCN_2s:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN_resnet_bn_skp(nn.Module):

    def __init__(self):
        super(FCN_resnet_bn_skp, self).__init__()
        
        resnet18 = models.resnet18(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])

        # For skip connections
        self.pool3 = feature_extractor[:6]
        self.pool4 = resnet18.layer3
        self.pool5 = resnet18.layer4
        
        self.upconv4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256)
        )

        self.upconv3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128)
        )

        self.upconv = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(256),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            # nn.ConvTranspose2d(32, 21, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(32, 21, 1)
        )
   

    def forward(self, x):
        
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        # print("Pool 3:", pool3.shape)
        # print("Pool 4:", pool4.shape)
        # print("Pool 5:", pool5.shape)

        # print("Up Conv 4: ", self.upconv4(pool5).shape)
        up4 = self.upconv4(pool5) + pool4

        # print("Up Conv 3: ", self.upconv3(up4).shape)

        up3 = self.upconv3(up4) + pool3

        # print("Up Conv: ", self.upconv(up3).shape)

        return self.upconv(up3)

if __name__ == "__main__":
    fcn = FCN_resnet_bn_skp()
    x = torch.autograd.Variable(torch.randn(1, 3, 224, 224))   
    print(fcn.forward(x).shape)