import torch
from torch import nn
import torch.nn.functional as fun

# defining the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class ADNet(nn.Module):
    def __init__(self):
        super(ADNet, self).__init__()

        self.filter_size = (3, 3, 3)
        self.padding = (1, 1, 1)
        self.stride = (1, 1, 1)
        self.pool_size = (2, 2, 2)
        self.pool_stride = (2, 2, 2)

        #---------------------------- CONVOLUTION AND BN LAYERS ----------------------------

        self.features = nn.Sequential(
            # conv1
            nn.Conv3d(1, 32, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, padding=(1, 0, 1)),

            # conv2
            nn.Conv3d(32, 64, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, padding=(1, 1, 0)),

            # conv3
            nn.Conv3d(64, 128, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, padding=(1, 0, 1)),

            # conv4
            nn.Conv3d(128, 256, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, padding=(1, 1, 0)),

            # conv5
            nn.Conv3d(256, 256, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=self.filter_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, 3),
            nn.BatchNorm1d(3),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6 * 6)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for param in self.parameters():
            if param.ndim != 1:
                nn.init.xavier_uniform_(param)


class LeNet3D(nn.Module):
    def __init__(self):
        super(LeNet3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 6, kernel_size=(5, 5, 5))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(16 * 44 * 36 * 36, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(fun.relu(self.conv1(x)))
        x = self.pool(fun.relu(self.conv2(x)))
        x = x.view(-1, 16 * 44 * 36 * 36)
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet3DBn(nn.Module):
    def __init__(self):
        super(LeNet3DBn, self).__init__()

        self.conv1 = nn.Conv3d(1, 6, kernel_size=(5, 5, 5))
        self.bn1 = nn.BatchNorm3d(6)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.bn2 = nn.BatchNorm3d(16)
        self.fc1 = nn.Linear(16 * 44 * 36 * 36, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(fun.relu(self.bn1(self.conv1(x))))
        x = self.pool(fun.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 44 * 36 * 36)
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = fun.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=3):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = fun.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

