import datetime
import sys
from torch import nn, optim
import random

from Utils import parse_args, prepare_directory, test_and_train_split
from DatasetHandler import generate_folds
from Model import device, LeNet3D, LeNet3DBn, ADNet
from TrainingAlgorithm import train_model_2, test_models


# labels map

def create_model(has_bn, volume):
    if has_bn:
        return LeNet3DBn(volume).to(device)

    return LeNet3D(volume).to(device)


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
    test_folds, train_folds = test_and_train_split(folds_arr)

    # model
    # lenet = create_model(args.batch_norm, args.volume)
    adnet = ADNet(args.volume).to(device)
    adam = optim.Adam(adnet.parameters(), 0.0001)
    cross_entropy = nn.CrossEntropyLoss()
    num_epochs = args.epochs
    batch_size = args.batch
    patience = args.patience

    # training and evaluation
    prepare_directory(args.model_path)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = train_model_2(adnet, adam, cross_entropy, train_folds, batch_size, num_epochs, device, timestamp,
                                  args.model_path, patience=patience)
    test_models(adnet, models_dir, cross_entropy, test_folds, batch_size, device, timestamp, args.model_path)


# starting point
if __name__ == '__main__':
    main(sys.argv[1:])
