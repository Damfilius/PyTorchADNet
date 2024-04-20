import numpy as np
import torch
from torch import save
from torch.utils.data import DataLoader, SubsetRandomSampler
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime


def train_one_epoch(model, dataloader, epoch_idx, sum_writer, opt_fn, loss_fn, device):
    model.train(True)

    running_loss = 0.
    num_correct = 0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        mri, labels = data[0].to(device), data[1].to(device)

        # zero out the gradients
        model.zero_grad(set_to_none=False)

        # generate the output
        mri = mri.unsqueeze(1)
        output = model(mri)

        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        loss.backward()

        # Adjust learning weights
        opt_fn.step()

        running_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader.dataset)

    sum_writer.add_scalar('Loss/train', avg_loss, epoch_idx)
    print(f"[TRAIN]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy


def validate_one_epoch(model, loss_fn, dataloader, epoch_idx, sum_writer, device):
    model.train(False)

    running_loss = 0.
    num_correct = 0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        mri, labels = data[0].to(device), data[1].to(device)

        mri = mri.unsqueeze(1)
        output = model(mri)

        loss = loss_fn(output, labels)
        running_loss += loss.item()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader.dataset)

    sum_writer.add_scalar('Loss/val', avg_loss, epoch_idx)

    print(f"[VAL]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy


def train_model(model, opt_fn, loss_fn, dataset, train_labels, batch_size, num_epochs, num_folds, device):
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    writer = SummaryWriter("/home/damfil/Uni/FYP/PyTorchADNet/sample_logs")

    train_losses = np.array([])
    val_losses = np.array([])
    train_accs = np.array([])
    val_accs = np.array([])

    best_loss = 999
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for fold, (training_idx, val_idx) in enumerate(kf.split(dataset, train_labels)):

        trainloader = DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(training_idx))
        valloader = DataLoader(dataset, batch_size, sampler=SubsetRandomSampler(val_idx))

        v_loss = 0
        for i in range(num_epochs):
            # train the model
            t_loss, t_acc = train_one_epoch(model, trainloader, i, writer, opt_fn, loss_fn, device)

            # validate the model on the parameters
            v_loss, v_acc = validate_one_epoch(model, loss_fn, valloader, i, writer, device)

            train_losses = np.append(train_losses, t_loss)
            val_losses = np.append(val_losses, v_loss)
            train_accs = np.append(train_accs, t_acc)
            val_accs = np.append(val_accs, v_acc)

            writer.add_scalars('Training vs. Validation Loss', {'Training': t_loss, 'Validation': v_loss}, i)
            writer.flush()

        if v_loss < best_loss:
            best_loss = v_loss
            model_path = 'model_{}_{}'.format(timestamp, fold)
            save(model.state_dict(), model_path)

        print("--------------------------------------------------------------------------------------------\n")


def test_model(model, loss_fn, test_dataset, test_labels, batch_size, device):
    model.train(False)
    test_loader = DataLoader(test_dataset, batch_size)
    running_loss = 0
    confusion_matrix = torch.zeros(3, 3)
    mri_scores = np.array([])

    # preprocessing the labels for multiclass ROC curves
    cn_class = 0
    ad_class = 1
    mci_class = 2
    label_binarizer = LabelBinarizer().fit(test_labels)
    test_one_hot_encoded = label_binarizer.transform(test_labels)

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs = inputs.usqueeze(1)
        outputs = model(inputs)
        mri_scores = np.append(mri_scores, outputs)

        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        prediction = outputs.argmax(dim=1, keepdim=True)
        confusion_matrix += multiclass_confusion_matrix(prediction, labels, 3)

    avg_loss = running_loss / len(test_loader)
    num_correct = confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[2, 2]
    accuracy = num_correct / len(test_loader.dataset)

    # computing the ROC curve
    display_ad = RocCurveDisplay.from_predictions(test_one_hot_encoded[:, ad_class],
                                                  mri_scores[:, ad_class],
                                                  name=f"AD vs the rest",
                                                  plot_chance_level=True)
    _ = display_ad.ax_.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                           title="One-vs-Rest ROC curves:\nAD vs (CN & MCI)")

    display_cn = RocCurveDisplay.from_predictions(test_one_hot_encoded[:, cn_class],
                                                  mri_scores[:, cn_class],
                                                  name=f"CN vs the rest",
                                                  plot_chance_level=True)
    _ = display_cn.ax_.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                           title="One-vs-Rest ROC curves:\nCN vs (AD & MCI)")

    display_mci = RocCurveDisplay.from_predictions(test_one_hot_encoded[:, mci_class],
                                                   mri_scores[:, mci_class],
                                                   name=f"MCI vs the rest",
                                                   plot_chance_level=True)
    _ = display_mci.ax_.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                            title="One-vs-Rest ROC curves:\nMCI vs (CN & AD)")

    print(f"[TEST]: Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, confusion_matrix
