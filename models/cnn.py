import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 2), padding=(1, 0))
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 1), padding=(2, 0))
        self.conv_head = nn.Conv2d(in_channels=512, out_channels=self.vocab_size, kernel_size=(7, 1), padding=(3, 0))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.bn1(x)

        x = F.leaky_relu(self.conv4(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.bn2(x)

        x = F.leaky_relu(self.conv6(x))
        x = F.pad(x, (0, 0, 1, 0))
        x = self.pool3(x)
        x = F.leaky_relu(self.conv7(x))
        x = self.bn3(x)

        x = F.leaky_relu(self.conv8(x))
        x = F.pad(x, (0, 0, 1, 0))
        x = self.pool4(x)

        x = F.leaky_relu(self.conv9(x))
        x = F.leaky_relu(self.conv10(x))
        out = self.conv_head(x)
        return out.squeeze(-1)


if __name__ == "__main__":
    model = CNN(vocab_size=33)
    x = torch.randn((2, 3, 500, 32))
    out = model(x)
    print(out.shape)
    from torchsummary import summary
    # print(summary(model, (1, 500, 32)))
