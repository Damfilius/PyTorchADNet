import sys
import numpy as np
from torch import nn, optim
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from Utils import parse_args
from DatasetHandler import MriDataset
from Model import ADNet, device
from TrainingAlgorithm import train_model

# labels map
label_map = {
    0: "CN",
    1: "AD",
    2: "MCI",
}

def main(arguments):
    args = parse_args(arguments)

    # dataset
    if args.dataset[-1] != '/':
        args.dataset += '/'

    dataset_dir = args.dataset
    full_dataset = MriDataset(dataset_dir+"labels.csv", dataset_dir, ToTensor(), None)

    labels = full_dataset.get_labels()
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, shuffle=True, stratify=labels)
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # model
    adnet = ADNet().to(device)
    adam = optim.Adam(adnet.parameters(), 0.001)
    cross_entropy = nn.CrossEntropyLoss()
    train_labels = full_dataset.get_labels(train_idx)
    num_epochs = args.epochs
    num_folds = 5
    batch_size = 4

    # training
    train_model(adnet, adam, cross_entropy, train_dataset, train_labels, batch_size, num_epochs, num_folds, device)

# starting point
if __name__ == '__main__':
    main(sys.argv[1:])
