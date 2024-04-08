import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import nibabel as nib

# labels map
labels = {
    0: "CN",
    1: "AD",
    2: "MCI",
}

#TODO: Create separate training, validation and testing datasets

############################## DATASET DEFINITION ##############################
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
        # mri = mri[np.newaxis, ...] # adds the channel dimension
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            mri = self.transform(mri)
        if self.target_transform:
            label = self.target_transform(label)
        return mri, label

datasetdir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/RegShortDataset/"
full_dataset = MriDataset(datasetdir+"labels.csv", datasetdir, ToTensor(), None)
train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

# defining the dataloaders
# trainloader = DataLoader(train_dataset, batch_size=3,  shuffle=True, num_workers=0)
# testloader = DataLoader(test_dataset, batch_size=3,  shuffle=True, num_workers=0)

sample_features, sample_labels = next(iter(trainloader))
print(f"Length of train loader: {len(trainloader)}") # returns the num of train MRIs div by batch size
print(f"Feature batch shape: {sample_features.size()}")
print(f"Labels batch shape: {sample_labels.size()}")

# sanity check
import matplotlib.pyplot as plt

sample_mri, label = full_dataset.__getitem__(5)
print(f"Sample MRI has shape: {sample_mri.shape}")
print(f"Extracted MRI has target value: {labels[label]}")
# sample_mri = sample_mri.numpy()

# plt.imshow(sample_mri[45], cmap='bone')
# plt.axis('off')
# plt.show()

# splitting the dataset into training and testing datasets (cross-validation for validation)


############################## MODEL DEFINITION ##############################
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# model expects an input of size (3, 1, 91, 91, 109)
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

        #---------------------------- FULLY CONNECTED LAYERS ----------------------------
        self.fc1 = nn.Linear(self.n_filters['l5'], self.n_filters['fc'], device=device)
        self.fc2 = nn.Linear(self.n_filters['l5'], self.n_filters['fc'], device=device)
        self.bn6 = nn.BatchNorm3d(self.n_filters['fc'], device=device)

        #---------------------------- OUTPUT LAYERS ----------------------------
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


############################## TRAINING ALGORITHM ##############################

num_epochs = 5
num_folds = 5
batch_size = 3
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter("/home/damfil/Uni/FYP/PyTorchADNet/sample_logs")

def train_one_epoch(model, dataloader, epoch_idx, sum_writer):
    model.train(True)

    running_loss = 0.
    avg_loss = 0.
    num_correct = 0

    for i, data in enumerate(dataloader):
        input, labels = data

        # clear the gradients
        opt_fn.zero_grad()

        # generate the output
        input = np.expand_dims(input, axis=1) # adds the channel dimension
        output = model(input)

        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        running_loss += loss.item()
        loss.backward()

        # Adjust learning weights
        opt_fn.step()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(label.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader)

    sum_writer.add_scalar('Loss/train', avg_loss, epoch_idx)

    print(f"TRAIN Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}]")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, epoch_idx, sum_writer):
    model.eval()

    running_loss = 0.
    avg_loss = 0.
    num_correct=0

    for i, data in enumerate(dataloader):
        input, label = data

        input = np.expand_dims(input, axis=1) # adds the channel dimension
        output = model(input)

        loss = loss_fn(output, input)
        running_loss += loss.item()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(label.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader)

    sum_writer.add_scalar('Loss/val', avg_loss, epoch_idx)

    print(f"VALIDATION Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}]")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return avg_loss, accuracy


kf = StratifiedKFold(num_folds)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    trainloader = DataLoader(test_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    valloader = DataLoader(test_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    adnet = ADNet().to(device)
    opt_fn = torch.optim.Adam(adnet.parameters(), 1e-4)

    for i in range(num_epochs):
        # train the model
        train_one_epoch(adnet, trainloader, i, writer)

        # validate the model on the parameters
        validate_one_epoch(adnet, valloader, i, writer)
            