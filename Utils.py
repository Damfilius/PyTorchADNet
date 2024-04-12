import argparse
from torch import equal


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
