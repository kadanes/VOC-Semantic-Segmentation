import torch
import torch.nn as nn
import torch.nn.functional as F

class Skip(nn.Module):

    def __init__(self):
        super(Skip, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d()
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d()
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(),
            # nn.MaxPool2d(2),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout2d(),
        )


        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 21, 3, stride=2, padding=1, output_padding=1)
        )

    # ReLU and Tanh cause output to be all 0

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        # print("Down 1: ", down1.shape)
        # print("Down 2: ", down2.shape)
        # print("Down 3: ", down3.shape)
        # print("Down 4: ", down4.shape)


        up4 = self.up4(down4)

        # print("Up 4: ", up4.shape)

        up3 = self.up3(up4 + down3)

        # print("Up 3: ", up3.shape)

        up2 = self.up2(up3 + down2)

        # print("Up 2: ", up2.shape)

        up1 = self.up1(up2)

        # print("Up 1: ", up1.shape)

        return up1

        # Down 1:  torch.Size([128, 64, 112, 112])
        # Down 2:  torch.Size([128, 128, 56, 56])
        # Down 3:  torch.Size([128, 256, 28, 28])
        # Down 4:  torch.Size([128, 512, 14, 14])
        # Up 4:  torch.Size([128, 256, 28, 28])
        # Up 3:  torch.Size([128, 128, 56, 56])
        # Up 2:  torch.Size([128, 64, 112, 112])
        # Up 1:  torch.Size([128, 21, 224, 224])
