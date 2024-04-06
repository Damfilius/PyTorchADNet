import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import nibabel as nib
from torchvision.transforms import ToTensor

# labels map
labels = {
    0: "CN",
    1: "AD",
    2: "MCI",
}

#TODO: Create separate training, validation and testing datasets

# defining the dataset
class MriDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        mri_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # convert from Nifti1 to numpy - this can be done as part of the transform
        mri = nib.load(mri_path)
        mri = np.array(mri.dataobj)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            mri = self.transform(mri)
        if self.target_transform:
            label = self.target_transform(label)
        return mri, label

datasetdir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/ShortDataset/"
full_dataset = MriDataset(datasetdir+"labels.csv", datasetdir, ToTensor(), None)

# defining the dataloader
dataloader = DataLoader(full_dataset, batch_size=10,  shuffle=True, num_workers=0)

# sanity check
import matplotlib.pyplot as plt

sample_mri, label = full_dataset.__getitem__(0)
print(f"Extracted MRI has target value: {labels[label]}")
sample_mri = sample_mri.numpy()

plt.imshow(sample_mri[96], cmap='bone')
plt.axis('off')
plt.show()

# splitting the dataset into training and testing datasets (cross-validation for validation)
train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

# defining the model
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

        #################################### CONVOLUTION AND BN LAYERS ####################################
        self.conv1 = nn.Conv3d(1, self.n_filters['l1'], self.filter_size, self.stride, self.padding, device=device)
        self.bn1 = nn.BatchNorm3d(self.n_filters['l1'], device=device)
        self.pool1 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,0,1))

        self.conv2 = nn.Conv3d(self.n_filters['l1'], self.n_filters['l2'], self.filter_size, self.stride, self.padding, device=device)
        self.bn2 = nn.BatchNorm3d(self.n_filters['l2'], device=device)
        self.pool2 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,1,0))

        self.conv3_a = nn.Conv3d(self.n_filters['l2'], self.n_filters['l3'], self.filter_size, self.stride, self.padding, device=device)
        self.conv3_b = nn.Conv3d(self.n_filters['l3'], self.n_filters['l3'], self.filter_size, self.stride, self.padding, device=device)
        self.bn3 = nn.BatchNorm3d(self.n_filters['l3'], device=device)
        self.pool3 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,0,1))

        self.conv4_a = nn.Conv3d(self.n_filters['l3'], self.n_filters['l4'], self.filter_size, self.stride, self.padding, device=device)
        self.conv4_b = nn.Conv3d(self.n_filters['l4'], self.n_filters['l4'], self.filter_size, self.stride, self.padding, device=device)
        self.bn4 = nn.BatchNorm3d(self.n_filters['l4'], device=device)
        self.pool4 = nn.MaxPool3d(self.pool_size, self.pool_stride, (1,1,0))

        self.conv5_a = nn.Conv3d(self.n_filters['l4'], self.n_filters['l5'], self.filter_size, self.stride, self.padding, device=device)
        self.conv5_b = nn.Conv3d(self.n_filters['l5'], self.n_filters['l5'], self.filter_size, self.stride, self.padding, device=device)
        self.bn5 = nn.BatchNorm3d(self.n_filters['l5'], device=device)
        self.pool5 = nn.MaxPool3d(self.pool_size, self.pool_stride)

        #################################### FULLY CONNECTED LAYERS ####################################
        self.fc1 = nn.Linear(self.n_filters['l5'], self.n_filters['fc'], device=device)
        self.fc2 = nn.Linear(self.n_filters['l5'], self.n_filters['fc'], device=device)
        self.bn6 = nn.BatchNorm3d(self.n_filters['fc'], device=device)

        #################################### OUTPUT LAYERS ####################################
        self.fc3 = nn.Linear(self.n_filters['l5'], 3, device=device)
        self.bn7 = nn.BatchNorm3d(3, device=device)

        self.softmax = nn.Softmax()
        self.dpo = nn.Dropout3d()
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1st layer group
        x = self.pool1(self.bn1(self.conv1(x)))

        x = self.pool2(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3_a(x))
        x = self.pool3(self.bn3(self.conv3_b(x)))

        x = self.bn4(self.conv4_a(x))
        x = self.pool4(self.bn4(self.conv4_b(x)))

        x = self.bn5(self.conv5_a(x))
        x = self.pool5(self.bn5(self.conv5_b(x)))

        x = self.bn6(self.relu(self.fc1(x)))
        x = self.dpo(x)
        x = self.bn6(self.relu(self.fc2(x)))
        x = self.dpo(x)

        x = self.bn7(self.fc3(x))
        x = self.softmax(x)

        return x