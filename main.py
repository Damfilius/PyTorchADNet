import datetime
import sys
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from Utils import parse_args, calculate_distribution, prepare_directory
from DatasetHandler import MriDataset, generate_folds
from Model import ADNet, device, LeNet3D
from TrainingAlgorithm import train_model, train_model_2, test_model


# labels map

def main(arguments):
    args = parse_args(arguments)

    # dataset - ensuring that the directory has a trailing /
    if args.dataset[-1] != '/':
        args.dataset += '/'

    # model path - ensuring that the directory does not have a trailing /
    if args.model_path[-1] == '/':
        args.model_path = args.model_path[:-1]

    folds_dir = args.dataset
    folds_arr = generate_folds(folds_dir)



    # model
    lenet = LeNet3D().to(device)
    adam = optim.Adam(lenet.parameters(), 0.0001)
    cross_entropy = nn.CrossEntropyLoss()
    num_epochs = args.epochs
    num_folds = 6
    batch_size = 1

    # training and evaluation
    prepare_directory(args.model_path)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_weights = train_model_2(lenet, adam, cross_entropy, folds_arr, batch_size, num_epochs, device, timestamp, args.model_path)

# starting point
if __name__ == '__main__':
    main(sys.argv[1:])
