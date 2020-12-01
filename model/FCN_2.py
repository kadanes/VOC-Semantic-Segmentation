# FCN_2s:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN_2(nn.Module):

    def __init__(self):
        super(FCN_2, self).__init__()
        vgg16 = models.vgg16(pretrained=True).features
        input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))

        # For skip connections

        self.pool4 = vgg16[:24]
        self.pool5 = vgg16[24:]
        
        self.upconv4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512)
        )


        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            # nn.ConvTranspose2d(32, 21, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(32, 21, 1)
        )
   

    def forward(self, x):
        
        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)
        up4 = self.upconv4(pool5) + pool4

        return self.upconv(up4)

if __name__ == "__main__":
    fcn = FCN_2()
    x = torch.autograd.Variable(torch.randn(1, 3, 224, 224))   
    print(fcn.forward(x).shape)


