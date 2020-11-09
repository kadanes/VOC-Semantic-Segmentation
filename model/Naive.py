import torch
import torch.nn as nn
import torch.nn.functional as F

# https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
# https://github.com/yassouali/pytorch_segmentation/blob/master/models/unet.py

class Naive(nn.Module):

    def __init__(self):
        super(Naive, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
        )

        # self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        # self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.decorder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 21, 3, stride=2, padding=1, output_padding=1),
        )

    # ReLU and Tanh cause output to be all 0

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decorder(encoded)
        return decoded

