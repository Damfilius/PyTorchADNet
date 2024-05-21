import datetime
import sys
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from Utils import parse_args, calculate_distribution, prepare_directory
from DatasetHandler import MriDataset
from Model import ADNet, device, LeNet3D
from TrainingAlgorithm import train_model, test_model


# labels map

def main(arguments):
    args = parse_args(arguments)

    # dataset - ensuring that the directory has a trailing /
    if args.dataset[-1] != '/':
        args.dataset += '/'

    # model path - ensuring that the directory does not have a trailing /
    if args.model_path[-1] == '/':
        args.model_path = args.model_path[:-1]

    dataset_dir = args.dataset
    full_dataset = MriDataset(dataset_dir + "labels.csv", dataset_dir, ToTensor(), None)

    labels = full_dataset.get_labels()
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, shuffle=True, stratify=labels)

    tr_dist = calculate_distribution(labels[train_idx], [0, 1, 2])
    te_dist = calculate_distribution(labels[test_idx], [0, 1, 2])

    if np.intersect1d(train_idx, test_idx).size == 0:
        print("No overlap between training and testing sets...\n")

    print(" CONFIRMING TESTING AND TRAINING SPLITS ")
    print("----------------------------------------")
    print(f" TRAIN SET LENGTH: [{len(train_idx)}]")
    print(f" TRAIN DISTRIBUTION: {tr_dist}")
    print(f" TEST SET LENGTH: [{len(test_idx)}]")
    print(f" TEST DISTRIBUTION: {te_dist}")
    print("----------------------------------------")

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    test_labels = full_dataset.get_labels(test_idx)

    # model
    # adnet = ADNet().to(device)
    lenet = LeNet3D().to(device)

    adam = optim.Adam(lenet.parameters(), 0.0001)
    cross_entropy = nn.CrossEntropyLoss()
    train_labels = full_dataset.get_labels(train_idx)
    num_epochs = args.epochs
    num_folds = 6
    batch_size = 1

    # training
    prepare_directory(args.model_path)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_weights = train_model(lenet, adam, cross_entropy, train_dataset, train_labels, batch_size, num_epochs, num_folds, device, timestamp, args.model_path)

    print("FINISHED TRAINING - LOADING THE MODEL AND STARTING TESTING")
    lenet.load_state_dict(torch.load(model_weights))
    avg_loss, conf_mat, f1_scores = test_model(lenet, cross_entropy, test_dataset, test_labels, batch_size, device, timestamp, args.model_path)
    print("FINISHED TESTING")

    print("CONFUSION MATRIX")
    print(np.matrix(conf_mat))
    print("F1 SCORES")
    print(f1_scores)


# starting point
if __name__ == '__main__':
    main(sys.argv[1:])
