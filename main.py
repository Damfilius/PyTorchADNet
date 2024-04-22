import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Utils import parse_args
from DatasetHandler import MriDataset
from Model import ADNet, LeNet3D, device
from TrainingAlgorithm import train_model, test_model

# labels map

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
    test_labels = full_dataset.get_labels(test_idx)

    # model
    adnet = ADNet().to(device)
    # lenet = LeNet3D().to(device)

    adam = optim.Adam(adnet.parameters(), 0.0001)
    # sgd = optim.SGD(adnet.parameters(), lr=0.0001, momentum=0.9)
    # adam2 = optim.Adam(lenet.parameters(), 0.0001)
    # sgd2 = optim.SGD(lenet.parameters(), lr=0.0001, momentum=0.9)

    cross_entropy = nn.CrossEntropyLoss()
    train_labels = full_dataset.get_labels(train_idx)
    num_epochs = args.epochs
    num_folds = 5
    batch_size = 4

    # training
    train_model(adnet, adam, cross_entropy, train_dataset, train_labels, batch_size, num_epochs, num_folds, device)
    print("FINISHED TRAINING - STARTED TESTING")
    # lenet.load_state_dict(torch.load("model_20240421_180514_4"))
    # print("Successfully loaded the model...")
    # testing
    avg_loss, conf_mat, f1_scores = test_model(adnet, cross_entropy, test_dataset, test_labels, batch_size, device)
    print("FINISHED TESTING")

    print("CONFUSION MATRIX")
    print(np.matrix(conf_mat))
    print("F1 SCORES")
    print(f1_scores)

# starting point
if __name__ == '__main__':
    main(sys.argv[1:])
