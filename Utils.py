import argparse
import os

import numpy as np
from torch import equal
import datetime


def get_distribution(dataset):
    counterAD = 0
    counterCN = 0
    counterMCI = 0

    for i, item in enumerate(dataset):
        sample, label = item
        if label == 0:
            counterCN += 1
        elif label == 1:
            counterAD += 1
        else:
            counterMCI += 1

    size = len(dataset)
    ad = 100 * counterAD / size
    cn = 100 * counterCN / size
    mci = 100 * counterMCI / size

    print(f"Distributions of the classes are following:\nAD - [{ad}%]\nCN - [{cn}%]\nMCI - [{mci}%]\n")
    return counterAD, counterCN, counterMCI


def has_overlap(train_idx, test_idx):
    for i in test_idx:
        for j in train_idx:
            if i == j:
                return True

    return False


def save_params(parameters, arr):
    for param in parameters:
        arr.append(param)


def check_params(params1, params2):
    if len(params1) != len(params2):
        print(f"Number of weights changes between the end of epoch 0 and start of epoch 1")
        return

    for i in range(len(params1)):
        if not equal(params1[i], params2[i]):
            print(f"The weights are not same after epoch 0 finishes and epoch 1 starts")
            break

    return


def get_length(parameters):
    counter = 0
    for param in parameters:
        counter += 1

    print(f"Mum. of parameters: [{counter}]")
    return counter


def parse_args(arguments):
    parser = argparse.ArgumentParser(prog="ADNet Model",
                                     description="Model Training and Diagnosis of AD through MRI Scans")
    parser.add_argument('dataset')
    parser.add_argument('-e', '--epochs', type=int, default=150)
    args = parser.parse_args(arguments)
    return args


def save_metrics_to_file(confusion_matrix, f1_scores, output_scores, cm_file, f1_file, out_file):
    np.savetxt(cm_file, confusion_matrix, delimiter=",")
    np.savetxt(f1_file, f1_scores, delimiter=",")
    np.savetxt(out_file, output_scores, delimiter=",")


def save_accs_and_losses(train_losses, train_accs, val_losses, val_accs, fold, timestamp):
    np.savetxt(f"AccsAndLosses/TrainLosses{fold}_{timestamp}.csv", train_losses, delimiter=",")
    np.savetxt(f"AccsAndLosses/TrainAccs{fold}_{timestamp}.csv", train_accs, delimiter=",")
    np.savetxt(f"AccsAndLosses/ValLosses{fold}_{timestamp}.csv", val_losses, delimiter=",")
    np.savetxt(f"AccsAndLosses/ValAccs{fold}_{timestamp}.csv", val_accs, delimiter=",")


def empty_logs(dirname="logs/"):
    try:
        with os.scandir(dirname) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


def calculate_distribution(arr, classes):
    dist = np.array([])
    for cl in classes:
        dist = np.append(dist, (arr == cl).sum())

    return dist


def print_datasets_into(labels, train_idx, test_idx):
    tr_dist = calculate_distribution(labels[train_idx], [0, 1, 2])
    val_dist = calculate_distribution(labels[test_idx], [0, 1, 2])

    if np.intersect1d(train_idx, test_idx).size == 0:
        print("No overlap between training and testing/valid sets...\n")

    print(" CONFIRMING TESTING AND TRAINING SPLITS ")
    print("----------------------------------------")
    print(f" TRAIN SET LENGTH: [{len(train_idx)}]")
    print(f" TRAIN DISTRIBUTION: {tr_dist}")
    print(f" TEST SET LENGTH: [{len(test_idx)}]")
    print(f" TEST DISTRIBUTION: {val_dist}")
    print("----------------------------------------")
