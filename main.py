import sys
import numpy as np
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Utils import parse_args
from DatasetHandler import MriDataset
from Model import ADNet, LeNet3D, device
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
    # adnet = ADNet().to(device)
    lenet = LeNet3D().to(device)

    # adam = optim.Adam(adnet.parameters(), 0.0001)
    # sgd = optim.SGD(adnet.parameters(), lr=0.0001, momentum=0.9)
    adam2 = optim.Adam(lenet.parameters(), 0.0001)
    sgd2 = optim.SGD(lenet.parameters(), lr=0.0001, momentum=0.9)

    cross_entropy = nn.CrossEntropyLoss()
    train_labels = full_dataset.get_labels(train_idx)
    num_epochs = args.epochs
    num_folds = 5
    batch_size = 4

    # training
    # train_model(lenet, adam2, cross_entropy, train_dataset, train_labels, batch_size, num_epochs, num_folds, device)

    full_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True, drop_last=True)
    lenet.train(True)
    for epoch in range(num_epochs):
        running_loss = 0
        num_correct = 0
        for i, data in tqdm(enumerate(full_dataloader, 0), total=len(full_dataloader)):
            inputs, labels = data
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            adam2.zero_grad()

            outputs = lenet(inputs)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            adam2.step()
            running_loss += loss.item()

            prediction = outputs.argmax(dim=1, keepdim=True)
            num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

        print(f"Avg. Loss per Batch: [{running_loss / len(full_dataloader)}]")
        print(f"Accuracy: [{100 * num_correct / len(full_dataloader.dataset)}%]")

# starting point
if __name__ == '__main__':
    main(sys.argv[1:])
