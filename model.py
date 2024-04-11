import os
import torch
from torch import nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
import nibabel as nib
import datetime
from tqdm import tqdm

# labels map
labels = {
    0: "CN",
    1: "AD",
    2: "MCI",
}

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

    def get_labels(self):
        labels = np.array([])
        for i in range(len(self.img_labels)):
            label = self.img_labels.iloc[i,1]
            labels = np.append(labels, label)

        return labels

datasetdir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/RegShortDataset/"
full_dataset = MriDataset(datasetdir+"labels.csv", datasetdir, ToTensor(), None)

labels = full_dataset.get_labels()
train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, shuffle=True, stratify=labels)
train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

############################## MODEL DEFINITION ##############################
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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


############################## TRAINING ALGORITHM ##############################

num_epochs = 5
num_folds = 5
batch_size = 4
train_losses = np.array([]); val_losses = np.array([])
train_accs = np.array([]); val_accs = np.array([])

params_pre_0 = []
params_post_0 = []
params_pre_1 = []
params_post_1 = []

adnet = ADNet().to(device)
opt_fn = torch.optim.Adam(adnet.parameters(), 2e-3)
loss_fn = nn.CrossEntropyLoss()

kf = KFold(num_folds, shuffle=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter("/home/damfil/Uni/FYP/PyTorchADNet/sample_logs")


def save_params(parameters, arr):
    for param in adnet.parameters():
        arr.append(param)

def check_params():
    if len(params_pre_0) != len(params_post_0):
        print(f"Number of weights does not stay the same before and after epoch 0")
    
    if len(params_post_0) != len(params_pre_1):
        print(f"Number of weights changes between the end of epoch 0 and start of epoch 1")
        return

    for i in range(len(params_pre_1)):
        if not torch.equal(params_pre_1[i], params_post_0[i]):
            print(f"The weights are not same after epoch 0 finishes and epoch 1 starts")
            break
        
    return

def get_length(parameters):
    counter = 0
    for param in parameters:
        counter += 1

    print(f"Mum. of parameters: [{counter}]")
    return counter


def train_one_epoch(model, dataloader, epoch_idx, sum_writer, optimizer, loss_fn):
    model.train(True)
    if epoch_idx == 0:
        save_params(model.parameters(), params_pre_0)
    elif epoch_idx == 1:
        save_params(model.parameters(), params_pre_1)

    running_loss = 0.
    num_correct = 0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, labels = data[0].to(device), data[1].to(device)

        # clear the gradients
        optimizer.zero_grad()

        # generate the output
        input = input.unsqueeze(1)
        output = model(input)

        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        running_loss += loss.item()
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader.dataset)

    sum_writer.add_scalar('Loss/train', avg_loss, epoch_idx)
    print(f"[TRAIN]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, epoch_idx, sum_writer):
    model.eval()

    running_loss = 0.
    avg_loss = 0.
    num_correct=0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, labels = data[0].to(device), data[1].to(device)

        input = input.unsqueeze(1)
        output = model(input)

        loss = loss_fn(output, labels)
        running_loss += loss.item()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader.dataset)

    sum_writer.add_scalar('Loss/val', avg_loss, epoch_idx)

    print(f"[VAL]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy


def train_model():
    best_loss = 999

    print(f"Number of training samples: [{len(train_dataset)}]")
    print(f"Number of testing samples: [{len(test_dataset)}]")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):

        trainloader = DataLoader(train_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        valloader = DataLoader(train_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

        v_loss = 0
        for i in range(num_epochs):

            # train the model
            t_loss, t_acc  = train_one_epoch(adnet, trainloader, i, writer, opt_fn, loss_fn)
            if i == 0:
               save_params(adnet.parameters(), params_post_0)
            elif i == 1:
               save_params(adnet.parameters(), params_post_1)
               check_params()

            # validate the model on the parameters
            v_loss, v_acc = validate_one_epoch(adnet, valloader, i, writer)

            np.append(train_losses, t_loss)
            np.append(train_accs, t_acc)
            np.append(val_losses, v_loss)
            np.append(val_accs, v_acc)

            writer.add_scalars('Training vs. Validation Loss', { 'Training' : t_loss, 'Validation' : v_loss }, i)
            writer.flush()

        if v_loss < best_loss:
            best_loss = v_loss
            model_path = 'model_{}_{}'.format(timestamp, fold)
            torch.save(adnet.state_dict(), model_path) 

        print("--------------------------------------------------------------------------------------------\n")

train_model()