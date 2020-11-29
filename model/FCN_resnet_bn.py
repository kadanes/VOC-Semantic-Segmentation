# FCN_2s:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN_resnet_bn(nn.Module):

    def __init__(self):
        super(FCN_resnet_bn, self).__init__()
        
        resnet18 = models.resnet18(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
        resnet18 = feature_extractor[:8]

        # For skip connections
        self.pool4 = resnet18
        
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
        # pool5 = self.pool5(pool4)
        up4 = self.upconv4(pool4)

        return self.upconv(up4)

if __name__ == "__main__":
    fcn = FCN_resnet_bn()
    x = torch.autograd.Variable(torch.randn(1, 3, 224, 224))   
    print(fcn.forward(x).shape)


