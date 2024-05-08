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
            nn.Conv3d(1, 32, (3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, return_indices=True, padding=(1, 0, 1)),

            # conv2
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, return_indices=True, padding=(1, 1, 0)),

            # conv3
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, padding=(1, 0, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, return_indices=True, padding=(1, 0, 1)),

            # conv4
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, return_indices=True, padding=(1, 1, 0)),

            # conv5
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, return_indices=True, padding=0),
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
