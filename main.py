import numpy as np
from torch import nn, optim
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from Utils import get_distribution
from DatasetHandler import MriDataset
from Model import ADNet, device
from TrainingAlgorithm import train_model

# labels map
labels = {
    0: "CN",
    1: "AD",
    2: "MCI",
}

# dataset
datasetdir = "/home/damfil/Uni/FYP/resources/mri/ad/dataset/NiftiDataset/RegShortDataset/"
full_dataset = MriDataset(datasetdir+"labels.csv", datasetdir, ToTensor(), None)

labels = full_dataset.get_labels()
train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, shuffle=True, stratify=labels)
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

get_distribution(train_dataset)
get_distribution(test_dataset)

# model
adnet = ADNet().to(device)
adam = optim.Adam(adnet.parameters(), 2e-3)
cross_entropy = nn.CrossEntropyLoss()
train_labels = full_dataset.get_labels(train_idx)
num_epochs = 5
num_folds = 5
batch_size = 4

# training
train_model(adnet, adam, cross_entropy, train_dataset, train_labels, batch_size, num_epochs, device)
