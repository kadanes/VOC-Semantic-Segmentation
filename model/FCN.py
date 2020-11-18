import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True).features
        # print(self.vgg16)
        
        self.flattent = nn.Sequential(
            # nn.Conv2d(512, 4096, 7),
            # nn.Conv2d(4096, 4096, 1),
        )

        # self.classes = nn.Conv2d(4096, 21, 1)

        self.upconv = nn.Sequential(
            # nn.ConvTranspose2d(21, 21, 3, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(21, 21, 3, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(21, 21, 3, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(21, 21, 9, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(21, 21, 9, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(21, 21, 4, stride=4, bias=False),
            # nn.ConvTranspose2d(21, 21, 8, stride=4, bias=False),
            # nn.ConvTranspose2d(21, 21, 16, stride=4, bias=False),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(32, 21, 3, stride=2, padding=1, output_padding=1),
        )

        self.fcn16 = nn.Sequential(
            self.vgg16,
            # self.flattent,
            # self.classes,
            self.upconv
        )
        

        # input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
        
        # output = self.fcn16(input)
        # print("Output: ", output.shape)
        # pass
        # print(self.fcn16)

    def forward(self, x):
        return self.fcn16(x)


if __name__ == "__main__":
    fcn = FCN()
    