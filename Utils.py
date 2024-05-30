import argparse
import os

import numpy as np
from torch import equal
import sys
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
    parser = argparse.ArgumentParser(prog="LeNet3D Model",
                                     description="Model Training and Diagnosis of AD through MRI Scans")
    parser.add_argument('dataset')
    parser.add_argument('model_path')
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-n', '--batch_norm', type=bool, default=False)
    parser.add_argument('-v', '--volume', type=int, default=100000)
    args = parser.parse_args(arguments)
    return args


def save_metrics_to_file(confusion_matrix, f1_scores, output_scores, cm_file, f1_file, out_file):
    np.savetxt(cm_file, confusion_matrix, delimiter=",")
    np.savetxt(f1_file, f1_scores, delimiter=",")
    np.savetxt(out_file, output_scores, delimiter=",")


def save_accs_and_losses(train_losses, train_accs, val_losses, val_accs, fold, timestamp, path):
    np.savetxt(f"{path}/AccsAndLosses/TrainLosses{fold}_{timestamp}.csv", train_losses, delimiter=",")
    np.savetxt(f"{path}/AccsAndLosses/TrainAccs{fold}_{timestamp}.csv", train_accs, delimiter=",")
    np.savetxt(f"{path}/AccsAndLosses/ValLosses{fold}_{timestamp}.csv", val_losses, delimiter=",")
    np.savetxt(f"{path}/AccsAndLosses/ValAccs{fold}_{timestamp}.csv", val_accs, delimiter=",")


def save_benchmarks_to_file(train_times_per_epoch, val_times_per_epoch, num_folds, pred_times, benchmarks_file):
    total_time = np.sum(train_times_per_epoch) + np.sum(val_times_per_epoch)
    avg_time_per_fold = total_time / num_folds
    avg_train_time_per_epoch = np.mean(train_times_per_epoch)
    avg_val_time_per_epoch = np.mean(val_times_per_epoch)
    avg_pred_time = np.mean(pred_times)

    print(f"Benchmarking Results:\n"
          f"Total Time: {total_time}s\n",
          f"Average Time / Fold: {avg_time_per_fold}s\n"
          f"Average Training Time / Epoch: {avg_train_time_per_epoch}s\n"
          f"Average Validation Time / Epoch: {avg_val_time_per_epoch}s\n"
          f"Average Prediction Time: {avg_pred_time}",
          file=benchmarks_file)


def save_performance_metrics_to_file(test_losses, test_accuracies, performance_file):
    mean_test_loss = np.mean(test_losses)
    mean_test_accuracy = np.mean(test_accuracies)

    print(f"test losses: {test_losses}",
          f"mean test loss: {mean_test_loss}",
          f"test accuracies: {test_accuracies}",
          f"mean test accuracies: {mean_test_accuracy}",
          file=performance_file)


def save_train_accs_and_losses(train_losses, train_accuracies, fold, timestamp, path):
    np.savetxt(f"{path}/AccsAndLosses/TrainLosses{fold}_{timestamp}.csv", train_losses, delimiter=",")
    np.savetxt(f"{path}/AccsAndLosses/TrainAccs{fold}_{timestamp}.csv", train_accuracies, delimiter=",")


def prepare_directory(directory_path):
    if not os.path.isdir(f"{directory_path}/AccsAndLosses"):
        os.makedirs(f"{directory_path}/AccsAndLosses")

    if not os.path.isdir(f"{directory_path}/Benchmarks"):
        os.makedirs(f"{directory_path}/Benchmarks")

    if not os.path.isdir(f"{directory_path}/logs"):
        os.makedirs(f"{directory_path}/logs")

    if not os.path.isdir(f"{directory_path}/Models"):
        os.makedirs(f"{directory_path}/Models")

    if not os.path.isdir(f"{directory_path}/PerformanceMetrics"):
        os.makedirs(f"{directory_path}/PerformanceMetrics")

    if not os.path.isdir(f"{directory_path}/ROCCurves"):
        os.makedirs(f"{directory_path}/ROCCurves")


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


def print_datasets_into(labels, train_idx, test_idx, is_valid=False):
    eval_set = "VALIDATION" if is_valid else "TEST"

    tr_dist = calculate_distribution(labels[train_idx], [0, 1, 2])
    val_dist = calculate_distribution(labels[test_idx], [0, 1, 2])

    if np.intersect1d(train_idx, test_idx).size == 0:
        print("No overlap between training and testing/valid sets...\n")

    print(" CONFIRMING TESTING AND TRAINING SPLITS ")
    print("----------------------------------------")
    print(f" TRAIN SET LENGTH: [{len(train_idx)}]")
    print(f" TRAIN SET DISTRIBUTION: {tr_dist}")
    print(f" {eval_set} SET LENGTH: [{len(test_idx)}]")
    print(f" {eval_set} SET DISTRIBUTION: {val_dist}")
    print("----------------------------------------")


def create_dir(path_to_dir):
    if os.path.isdir(path_to_dir):
        return

    os.mkdir(path_to_dir)


def replace_zeros_with_min_float(arr):
    if 0 not in arr:
        return arr

    arr = np.where(arr == 0, sys.float_info.min, arr)
    return arr


def test_and_train_split(fold_arr):
    num_folds = len(fold_arr)
    random_indices = np.random.choice(num_folds, 2, replace=False)
    test_folds = fold_arr[random_indices]
    train_folds = np.delete(fold_arr, random_indices)
    return test_folds, train_folds


def reset_model_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()