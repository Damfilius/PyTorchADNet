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

# defining the architecture of the model
# model expects an input of size (4, 1, 91, 91, 109)
class ADNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_filters = {'l1': 32,
                          'l2': 64,
                          'l3': 128,
                          'l4': 256,
                          'l5': 256,
                          'fc': 512}

        self.filter_size = (3,3,3)
        self.padding = (1,1,1)
        self.stride = (1,1,1)
        self.pool_size=(2,2,2)
        self.pool_stride=(2,2,2)

        #---------------------------- CONVOLUTION AND BN LAYERS ----------------------------

        # layer 1
        self.conv1 = nn.Conv3d(1, self.n_filters['l1'], self.filter_size, self.stride, self.padding, device=device)
        self.bn1 = nn.BatchNorm3d(self.n_filters['l1'], device=device)
        self.pool1 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,0,1))

        # layer 2
        self.conv2 = nn.Conv3d(self.n_filters['l1'], self.n_filters['l2'], self.filter_size, self.stride, self.padding, device=device)
        self.bn2 = nn.BatchNorm3d(self.n_filters['l2'], device=device)
        self.pool2 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,1,0))

        # layer 3
        self.conv3a = nn.Conv3d(self.n_filters['l2'], self.n_filters['l3'], self.filter_size, self.stride, self.padding, device=device)
        self.conv3b = nn.Conv3d(self.n_filters['l3'], self.n_filters['l3'], self.filter_size, self.stride, self.padding, device=device)
        self.bn3 = nn.BatchNorm3d(self.n_filters['l3'], device=device)
        self.pool3 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,0,1))

        # layer 4
        self.conv4a = nn.Conv3d(self.n_filters['l3'], self.n_filters['l4'], self.filter_size, self.stride, self.padding, device=device)
        self.conv4b = nn.Conv3d(self.n_filters['l4'], self.n_filters['l4'], self.filter_size, self.stride, self.padding, device=device)
        self.bn4 = nn.BatchNorm3d(self.n_filters['l4'], device=device)
        self.pool4 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,1,0))

        # layer 5
        self.conv5a = nn.Conv3d(self.n_filters['l4'], self.n_filters['l5'], self.filter_size, self.stride, self.padding, device=device)
        self.conv5b = nn.Conv3d(self.n_filters['l5'], self.n_filters['l5'], self.filter_size, self.stride, self.padding, device=device)
        self.bn5 = nn.BatchNorm3d(self.n_filters['l5'], device=device)
        self.pool5 = nn.MaxPool3d(self.pool_size, self.pool_stride)

        #---------------------------- FULLY CONNECTED LAYERS ----------------------------

        # layer 6 and 7
        self.fc1 = nn.Linear(6912, self.n_filters['fc'], device=device)
        self.fc2 = nn.Linear(self.n_filters['fc'], self.n_filters['fc'], device=device)
        self.bn6 = nn.BatchNorm1d(self.n_filters['fc'], device=device)

        #---------------------------- OUTPUT LAYERS ----------------------------

        # layer 8
        self.fc3 = nn.Linear(self.n_filters['fc'], 3, device=device)
        self.bn7 = nn.BatchNorm1d(3, device=device)

    def forward(self, x):
        # convolutions
        x = self.pool1(self.bn1(fun.relu(self.conv1(x))))
        x = self.pool2(self.bn2(fun.relu(self.conv2(x))))
        x = self.pool3(self.bn3(fun.relu(self.conv3b(self.bn3(fun.relu(self.conv3a(x)))))))
        x = self.pool4(self.bn4(fun.relu(self.conv4b(self.bn4(fun.relu(self.conv4a(x)))))))
        x = self.pool5(self.bn5(fun.relu(self.conv5b(self.bn5(fun.relu(self.conv5a(x)))))))

        # fully connected layers
        x = x.view(x.size(0),-1)
        x = fun.dropout(self.bn6(fun.relu(self.fc1(x))))
        x = fun.dropout(self.bn6(fun.relu(self.fc2(x))))
        x = fun.softmax(self.bn7(self.fc3(x)))

        return x