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

        self.convb1 = self.conv_block(1, 32, (1, 0, 1))
        self.convb2 = self.conv_block(32, 64, (1, 1, 0))
        self.convb3 = self.double_conv_block(64, 128, (1, 0, 1))
        self.convb4 = self.double_conv_block(128, 256, (1, 1, 0))
        self.convb5 = self.double_conv_block(256, 256, 0)

        #---------------------------- FULLY CONNECTED LAYERS ----------------------------

        self.fc1 = self.fc_block(256 * 6 * 6 * 6, 512)
        self.fc2 = self.fc_block(256 * 6 * 6 * 6, 512)
        self.fc3 = self.output_layer(512, 3)

    def forward(self, x):
        # convolutions
        x = self.convb1(x)
        x = self.convb2(x)
        x = self.convb3(x)
        x = self.convb4(x)
        x = self.convb5(x)
        x = x.view(-1, 256 * 6 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def conv_block(self, in_channels, out_channels, pool_pad):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, self.filter_size, self.stride, self.padding),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(self.pool_size, self.pool_stride, pool_pad)
        )

    def double_conv_block(self, in_channels, out_channels, pool_pad):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, self.filter_size, self.stride, self.padding),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, self.filter_size, self.stride, self.padding),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(self.pool_size, self.pool_stride, pool_pad)
        )

    def fc_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
            nn.Dropout1d()
        )

    def output_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Softmax()
        )

    def init_weights(self):
        for param in self.parameters():
            if param.ndim != 1:
                nn.init.xavier_uniform_(param)

class LeNet3D(nn.Module):
    def __init__(self):
        super(LeNet3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 6, kernel_size=(5,5,5))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(16 * 39 * 42 * 47, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(fun.relu(self.conv1(x)))
        x = self.pool(fun.relu(self.conv2(x)))
        x = x.view(-1, 16 * 39 * 42 * 47)
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x
