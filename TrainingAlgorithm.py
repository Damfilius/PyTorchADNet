import numpy as np
import torch
from torch import save
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from Utils import save_metrics_to_file, save_accs_and_losses, print_datasets_into
import time

label_map = {
    0: "CN",
    1: "AD",
    2: "MCI",
}


def train_one_epoch(model, dataloader, opt_fn, loss_fn, epoch_idx, device):
    model.train(True)

    running_loss = 0.
    num_correct = 0
    epoch_start = time.time()
    avg_time_per_batch = 0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        mri, labels = data[0].to(device), data[1].to(device)

        # zero out the gradients
        model.zero_grad(set_to_none=False)

        # generate the output
        mri = mri.unsqueeze(1)
        pred_start = time.time()
        output = model(mri)
        pred_end = time.time()
        avg_time_per_batch += pred_end - pred_start

        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        loss.backward()

        # Adjust learning weights
        opt_fn.step()

        running_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    epoch_end = time.time()
    epoch_elapsed = epoch_end - epoch_start
    avg_time_per_batch /= len(dataloader)

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader.dataset)

    print(f"[TRAIN]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy, epoch_elapsed, avg_time_per_batch


def validate_one_epoch(model, dataloader2, loss_fn, epoch_idx, device):
    model.train(False)

    running_loss = 0.
    num_correct = 0
    epoch_start = time.time()
    avg_time_per_batch = 0

    for i, data in tqdm(enumerate(dataloader2), total=len(dataloader2)):
        mri, labels = data[0].to(device), data[1].to(device)

        mri = mri.unsqueeze(1)

        pred_start = time.time()
        output = model(mri)
        pred_end = time.time()
        avg_time_per_batch += pred_end - pred_start

        loss = loss_fn(output, labels)
        running_loss += loss.item()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    epoch_end = time.time()
    epoch_elapsed = epoch_end - epoch_start
    avg_time_per_batch /= len(dataloader2)

    avg_loss = running_loss / len(dataloader2)
    accuracy = 100 * num_correct / len(dataloader2.dataset)

    # sum_writer.add_scalar('Loss/val', avg_loss, epoch_idx)

    print(f"[VAL]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy, epoch_elapsed, avg_time_per_batch


def write_scalars(writer, t_loss, t_acc, v_loss, v_acc, fold, epoch):
    # reporting for testing
    writer.add_scalar(f'Loss_Fold{fold}/train', t_loss, epoch)
    writer.add_scalar(f'Accuracy_Fold{fold}/train', t_acc, epoch)

    # reporting for validation
    writer.add_scalar(f'Loss_Fold{fold}/val', v_loss, epoch)
    writer.add_scalar(f'Accuracy_Fold{fold}/val', v_acc, epoch)

    # for the overlay graph
    writer.add_scalars('Training vs. Validation Loss', {'Training': t_loss, 'Validation': v_loss}, epoch)
    writer.flush()


def train_model(model, opt_fn, loss_fn, dataset, train_labels, batch_size, num_epochs, num_folds, device):
    # cross validation and saving metrics
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    writer = SummaryWriter("logs/")
    save(model.state_dict(), "Models/init_model")

    # benchmarks
    benchmarks_file = open("Benchmarks/training.txt", "a")
    total_time = 0
    train_epoch_time = np.array([])
    val_epoch_time = np.array([])
    pred_times = np.array([])

    # saving the model and calculating average performance
    best_loss = 999
    best_losses = np.array([])
    best_accuracies = np.array([])
    total_fold_times = np.array([])
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for fold, (training_idx, val_idx) in enumerate(kf.split(dataset, train_labels)):
        # for loss and train across time
        train_losses = np.array([])
        val_losses = np.array([])
        train_accs = np.array([])
        val_accs = np.array([])

        # preparing the dataloaders
        train_dataset = Subset(dataset, training_idx)
        val_dataset = Subset(dataset, val_idx)
        trainloader = DataLoader(train_dataset, batch_size, drop_last=True)
        valloader = DataLoader(val_dataset, batch_size, drop_last=True)

        print(f"FOLD {fold}")
        print_datasets_into(train_labels, training_idx, val_idx)

        # benchmarking
        total_fold_time = 0

        # training the model
        for i in range(num_epochs):
            t_loss, t_acc, train_time, avg_train_pred = train_one_epoch(model, trainloader, opt_fn, loss_fn, i, device)
            v_loss, v_acc, val_time, avg_val_pred = validate_one_epoch(model, valloader, loss_fn, i, device)
            write_scalars(writer, t_loss, t_acc, v_loss, v_acc, fold, i)

            if v_loss < best_loss:
                best_loss = v_loss
                print("Saving the model...")
                model_path = 'Models/model_{}'.format(timestamp)
                save(model.state_dict(), model_path)

            # total time - excludes all the summary writing and numpy appending
            total_fold_time += train_time + val_time
            train_epoch_time = np.append(train_epoch_time, train_time)
            val_epoch_time = np.append(val_epoch_time, val_time)
            pred_times = np.append(pred_times, [avg_train_pred, avg_val_pred])

            # add values to the accuracy and loss arrays
            train_losses = np.append(train_losses, t_loss)
            train_accs = np.append(train_accs, t_acc)
            val_losses = np.append(val_losses, v_loss)
            val_accs = np.append(val_accs, v_acc)

        save_accs_and_losses(train_losses, train_accs, val_losses, val_accs, fold)
        print("--------------------------------------------------------------------------------------------\n")

        # store the best loss recorded for a model of a specific fold
        fold_best_loss = np.min(val_losses)
        fold_best_acc = np.max(val_accs)
        best_losses = np.append(best_losses, fold_best_loss)
        best_accuracies = np.append(best_accuracies, fold_best_acc)

        # benchmarking
        total_fold_times = np.append(total_fold_times, total_fold_time)
        total_time += total_fold_time

        # resetting the model
        model.load_state_dict(torch.load("Models/init_model"))


    # performance
    average_loss = np.mean(best_losses)
    average_accuracy = np.mean(best_accuracies)
    print(f"Losses: {best_losses}")
    print(f"Average Loss: {average_loss}")
    print(f"Accuracies: {best_accuracies}")
    print(f"Average Accuracy: {average_accuracy}")

    # benchmarking
    avg_time_per_fold = np.mean(total_fold_times)
    avg_train_time = np.mean(train_epoch_time)
    avg_val_time = np.mean(val_epoch_time)
    avg_pred_time = np.mean(pred_times)

    print(f"Benchmarking Results:\n"
          f"Total Time: {total_time}s\n"
          f"Average Time / Fold: {avg_time_per_fold}s\n"
          f"Average Training Time / Epoch: {avg_train_time}s\n"
          f"Average Validation Time / Epoch: {avg_val_time}s\n"
          f"Average Prediction Time: {avg_pred_time}",
          file=benchmarks_file)

    benchmarks_file.close()


def train_model_2(model, opt_fn, loss_fn, dataset, train_labels, batch_size, num_epochs, device):
    train_idx, val_idx = train_test_split(np.arange(len(train_labels)), test_size=0.2, shuffle=True,
                                          stratify=train_labels)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    trainloader = DataLoader(train_dataset, batch_size, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size, drop_last=True)

    print_datasets_into(train_labels, train_idx, val_idx)

    for i in range(num_epochs):
        train_one_epoch(model, trainloader, opt_fn, loss_fn, i, device)
        validate_one_epoch(model, valloader, loss_fn, i, device)


def compute_f1_scores(confusion_matrix):
    sum_horiz = np.sum(confusion_matrix, axis=1)
    sum_vert = np.sum(confusion_matrix, axis=0)
    true_positives = confusion_matrix.diagonal()

    precisions = true_positives / sum_vert
    recalls = true_positives / sum_horiz

    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    return f1_scores


def compute_ROC_curves(output_scores, test_labels):
    # in case that the dataloader dropped the last non-full batch
    len_scores = len(output_scores)
    len_labels = len(test_labels)

    if len_scores < len_labels:
        test_labels = test_labels[:len_scores]

    label_binarizer = LabelBinarizer().fit(test_labels)
    test_one_hot_encoded = label_binarizer.transform(test_labels)

    for i in range(3):
        fpr, tpr, thresholds = metrics.roc_curve(test_one_hot_encoded[:, i], output_scores[:, i])
        auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='AD estimator')
        display.plot()
        plt.savefig(f"ROCCurves/{label_map[i]}_ROC_curve.png")


def update_confusion_matrix(confusion_matrix, prediction, labels):
    for i in range(len(prediction)):
        confusion_matrix[labels[i].item(), prediction[i].item()] += 1

    return confusion_matrix


def test_model(model, loss_fn, test_dataset, test_labels, batch_size, device):
    model.train(False)
    test_loader = DataLoader(test_dataset, batch_size, drop_last=True)
    running_loss = 0
    confusion_matrix = np.zeros((3, 3))
    output_scores = np.array([])

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        output_scores = np.append(output_scores, outputs.cpu().detach().numpy())

        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        prediction = outputs.argmax(dim=1, keepdim=True)
        confusion_matrix = update_confusion_matrix(confusion_matrix, prediction, labels)

    avg_loss = running_loss / len(test_loader)
    num_correct = confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[2, 2]
    accuracy = 100 * num_correct / len(test_loader.dataset)
    print(f"[TEST]: Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    f1_scores = compute_f1_scores(confusion_matrix)

    # computing the ROC curve
    output_scores = np.reshape(output_scores, (-1, 3))
    compute_ROC_curves(output_scores, test_labels)

    save_metrics_to_file(confusion_matrix, f1_scores, output_scores,
                         "PerformanceMetrics/ConfusionMatrix.csv",
                         "PerformanceMetrics/F1Scores.csv",
                         "PerformanceMetrics/OutputScores.csv")

    return avg_loss, confusion_matrix, f1_scores
