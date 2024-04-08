import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import nibabel as nib
import datetime

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
        mri = np.array(mri.dataobj,dtype=np.ubyte)
        if mri.ndim == 4:
            mri = mri[:, :, :, 0]
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
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, self.n_filters['l1'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l1'], device=device),
            nn.MaxPool3d(self.pool_size, self.pool_stride, (1,0,1))
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(self.n_filters['l1'], self.n_filters['l2'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l2'], device=device),
            nn.MaxPool3d(self.pool_size, self.pool_stride, (1,1,0))
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(self.n_filters['l2'], self.n_filters['l3'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l3'], device=device),
            nn.Conv3d(self.n_filters['l3'], self.n_filters['l3'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l3'], device=device),
            nn.MaxPool3d(self.pool_size, self.pool_stride, (1,0,1))
        )

        self.layer4 = nn.Sequential(
            nn.Conv3d(self.n_filters['l3'], self.n_filters['l4'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l4'], device=device),
            nn.Conv3d(self.n_filters['l4'], self.n_filters['l4'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l4'], device=device),
            nn.MaxPool3d(self.pool_size, self.pool_stride, (1,1,0))
        )

        self.layer5 = nn.Sequential(
            nn.Conv3d(self.n_filters['l4'], self.n_filters['l5'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l5'], device=device),
            nn.Conv3d(self.n_filters['l5'], self.n_filters['l5'], self.filter_size, self.stride, self.padding, device=device),
            nn.ReLU(),
            nn.BatchNorm3d(self.n_filters['l5'], device=device),
            nn.MaxPool3d(self.pool_size, self.pool_stride)
        )

        #---------------------------- FULLY CONNECTED LAYERS ----------------------------
        self.layer6 = nn.Sequential(
            nn.Linear(6912, self.n_filters['fc'], device=device),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_filters['fc'], device=device),
            nn.Dropout()
        )

        self.layer7 = nn.Sequential(
            nn.Linear(self.n_filters['fc'], self.n_filters['fc'], device=device),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_filters['fc'], device=device),
            nn.Dropout()
        )

        #---------------------------- OUTPUT LAYERS ----------------------------
        self.layer8 = nn.Sequential(
            nn.Linear(self.n_filters['fc'], 3, device=device),
            nn.BatchNorm1d(3, device=device),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0),-1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x


############################## TRAINING ALGORITHM ##############################

num_epochs = 5
num_folds = 5
batch_size = 3
best_loss = 999
kf = KFold(num_folds, shuffle=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter("/home/damfil/Uni/FYP/PyTorchADNet/sample_logs")

def train_one_epoch(model, dataloader, epoch_idx, sum_writer):
    model.train(True)

    running_loss = 0.
    avg_loss = 0.
    num_correct = 0

    for i, data in enumerate(dataloader):
        input, labels = data[0].to(device), data[1].to(device)

        # clear the gradients
        opt_fn.zero_grad()

        # generate the output
        input = input.unsqueeze(1)
        output = model(input)

        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        running_loss += loss.item()
        loss.backward()

        # Adjust learning weights
        opt_fn.step()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader)

    sum_writer.add_scalar('Loss/train', avg_loss, epoch_idx)

    print(f"TRAIN Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}]")
    print("--------------------------------------------------------------------------------------------\n")

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
    print("--------------------------------------------------------------------------------------------\n")

    return avg_loss, accuracy

print("############### STARTED TRAINING ###############\n\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"######## FOLD [{fold}] ########\n")

    trainloader = DataLoader(train_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    valloader = DataLoader(train_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    adnet = ADNet().to(device)
    opt_fn = torch.optim.Adam(adnet.parameters(), 1e-4)

    for i in range(num_epochs):
        print(f"~~~~~~~~~EPOCH [{i}]~~~~~~~~~\n")

        # train the model
        t_loss, t_acc  = train_one_epoch(adnet, trainloader, i, writer)

        # validate the model on the parameters
        v_loss, v_acc = validate_one_epoch(adnet, valloader, i, writer)

        writer.add_scalars('Training vs. Validation Loss', { 'Training' : t_loss, 'Validation' : v_loss }, i)
        writer.flush()

    if v_loss < best_loss:
        best_loss = v_loss
        model_path = 'model_{}_{}'.format(timestamp, i)
        torch.save(adnet.state_dict(), model_path) 

print("\n\n\n############### FINISHED TRAINING ###############")