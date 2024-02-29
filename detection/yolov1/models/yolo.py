import torch
import torch.nn as nn


class YoloV1(nn.Module):
    def __init__(self, img_size, in_channels, num_classes, S, B):
        super(YoloV1, self).__init__()
        # 初始化参数
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.S = S
        self.B = B

        # 卷积层  N = (W − F+ 2P)/S+1
        self.conv_layers = nn.Sequential(
            # layers1 448*448*3 -> 111*111*192
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=1),  # layer1
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # layers2 111*111*192 ->55*55*256
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),                    # layer2
            nn.BatchNorm2d(num_features=192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # layers3 55*55*256 -> 27*27*512
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),                              # layer3
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),                   # layer4
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),                              # layer5
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),                   # layer6
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # layers4 27*27*512 -> 13*13*1024
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),                              # layer7
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),                   # layer8
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),                              # layer9
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),                   # layer10
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),                              # layer11
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),                   # layer12
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),                              # layer13
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),                   # layer14
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),                              # layer15
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),                  # layer16
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # layers5 13*13*1024 -> 7*7*1024
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),                             # layer17
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),                  # layer18
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),                             # layer19
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),                  # layer20
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),                 # layer21
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),       # layer22
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),

            # layers6 7*7*1024 -> 7*7*1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),                 # layer23
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),                 # layer24
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=self.S * self.S * 1024, out_features=4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view((out.shape[0], -1))
        out = self.fc_layers(out)
        out = out.view((out.shape[0]), self.S, self.S, (self.B * 5 + self.num_classes))
        return out


if __name__ == '__main__':
    img_data = torch.randn(1, 3, 448, 448)
    yolo = YoloV1(img_size=448, in_channels=3, num_classes=20, S=7, B=2)
    out = yolo(img_data)
    print(img_data.shape, out.shape)





