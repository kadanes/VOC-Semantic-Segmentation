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
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
            # nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(),
            # nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
        )


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
        transpose = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),

        encoded = self.encoder(x)

        pool = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        print("Pool:", pool(x).shape)


        decoded = self.decorder(encoded)
        return decoded

