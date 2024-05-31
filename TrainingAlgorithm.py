import numpy as np
import torch
from torch import save
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from Utils import save_metrics_to_file, save_accs_and_losses, print_datasets_into, reset_model_weights, save_benchmarks_to_file, save_performance_metrics_to_file, test_and_train_split
from Utils import generate_random_index_pairs, save_validation_performance_to_file
import time
import os

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
    model.eval()

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

def write_training(writer, t_loss, t_acc, fold, epoch):
    writer.add_scalar(f'Loss_Fold{fold}/train', t_loss, epoch)
    writer.add_scalar(f'Accuracy_Fold{fold}/train', t_acc, epoch)


def train_model(model, opt_fn, loss_fn, dataset, train_labels, batch_size, num_epochs, num_folds, device, timestamp, path):
    # cross validation and saving metrics and model
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    writer = SummaryWriter(f"{path}/logs/")
    model_path = f"{path}/Models/model_{timestamp}"
    save(model.state_dict(), f"{path}/Models/init_model_{timestamp}")

    # benchmarks
    benchmarks_file = open(f"{path}/Benchmarks/training_{timestamp}.txt", "a")
    total_time = 0
    train_epoch_time = np.array([])
    val_epoch_time = np.array([])
    pred_times = np.array([])

    # saving the model and calculating average performance
    best_loss = 999
    model_losses = np.array([])
    model_accuracies = np.array([])
    total_fold_times = np.array([])

    for fold, (training_idx, val_idx) in enumerate(kf.split(dataset, train_labels)):
        # preparing the dataloaders
        train_dataset = Subset(dataset, training_idx)
        val_dataset = Subset(dataset, val_idx)
        trainloader = DataLoader(train_dataset, batch_size, drop_last=True)
        valloader = DataLoader(val_dataset, batch_size, drop_last=True)

        # for loss and train across time
        train_losses = np.array([])
        val_losses = np.array([])
        train_accs = np.array([])
        val_accs = np.array([])

        print(f"FOLD {fold}")
        print_datasets_into(train_labels, training_idx, val_idx, is_valid=True)

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

        save_accs_and_losses(train_losses, train_accs, val_losses, val_accs, fold, timestamp, path)
        print("--------------------------------------------------------------------------------------------\n")

        # store the best achieved loss and accuracy
        min_loss_idx = np.argmin(val_losses)
        fold_best_loss = val_losses[min_loss_idx]
        corr_acc = val_accs[min_loss_idx]
        model_losses = np.append(model_losses, fold_best_loss)
        model_accuracies = np.append(model_accuracies, corr_acc)

        # benchmarking
        total_fold_times = np.append(total_fold_times, total_fold_time)
        total_time += total_fold_time

        # resetting the model
        model.load_state_dict(torch.load(f"{path}/Models/init_model_{timestamp}"))

    # performance
    average_loss = np.mean(model_losses)
    average_accuracy = np.mean(model_accuracies)
    print(f"Losses: {model_losses}")
    print(f"Average Loss: {average_loss}")
    print(f"Accuracies: {model_accuracies}")
    print(f"Average Accuracy: {average_accuracy}")

    # benchmarking
    avg_time_per_fold = np.mean(total_fold_times)
    avg_train_time = np.mean(train_epoch_time)
    avg_val_time = np.mean(val_epoch_time)
    avg_pred_time = np.mean(pred_times)

    print(f"Benchmarking Results:\n"
          f"Total Time: {total_time}s\n",
          f"Fold Times: {total_fold_times}"
          f"Average Time / Fold: {avg_time_per_fold}s\n"
          f"Average Training Time / Epoch: {avg_train_time}s\n"
          f"Average Validation Time / Epoch: {avg_val_time}s\n"
          f"Average Prediction Time: {avg_pred_time}",
          file=benchmarks_file)

    benchmarks_file.close()

    return model_path


def train_model_2(model, opt_fn, loss_fn, folds, batch_size, num_epochs, device, timestamp, path, patience):
    # logging and model
    writer = SummaryWriter(f"{path}/logs/")
    save(model.state_dict(), f"{path}/Models/init_model_{timestamp}")

    # benchmarks
    train_epoch_time = np.array([])
    validation_epoch_time = np.array([])
    pred_times = np.array([])

    # performance
    best_fold_validation_losses = np.array([])
    best_fold_validation_accuracies = np.array([])

    # dataset splitting
    random_index_pairs = generate_random_index_pairs(num_datasets=len(folds), num_pairs=4)

    for i in range(len(random_index_pairs)):
        # preparing the training and testing dataset
        index_pair = random_index_pairs[i]
        validation_folds = folds[index_pair]
        validation_dataset = ConcatDataset(validation_folds)
        training_folds = np.delete(folds, index_pair)
        training_dataset = ConcatDataset(training_folds)
        train_loader = DataLoader(training_dataset, batch_size, drop_last=True)
        validation_loader = DataLoader(validation_dataset, batch_size, drop_last=True)

        # training performance
        train_accuracies = np.array([])
        train_losses = np.array([])
        validation_accuracies = np.array([])
        validation_losses = np.array([])

        # saving the model
        best_validation_accuracy = 0
        model_path = f"{path}/Models/model_fold{i}_{timestamp}"

        fold_patience = patience

        # train the model
        for e in range(num_epochs):
            t_loss, t_acc, train_time, avg_pred_time = train_one_epoch(model, train_loader, opt_fn, loss_fn, e, device)
            v_loss, v_acc, val_time, avg_val_pred = validate_one_epoch(model, validation_loader, loss_fn, e, device)
            write_scalars(writer, t_loss, t_acc, v_loss, v_acc, i, e)

            # conditionally save the model
            if v_acc > best_validation_accuracy:
                print("Saving the model...")
                best_validation_accuracy = v_acc
                save(model.state_dict(), model_path)
                fold_patience = patience
            else:
                fold_patience -= 1

            # benchmarks
            train_epoch_time = np.append(train_epoch_time, train_time)
            validation_epoch_time = np.append(validation_epoch_time, val_time)
            pred_times = np.append(pred_times, [avg_pred_time, avg_val_pred])

            # performance
            train_losses = np.append(train_losses, t_loss)
            train_accuracies = np.append(train_accuracies, t_acc)
            validation_losses = np.append(validation_losses, v_loss)
            validation_accuracies = np.append(validation_accuracies, v_acc)

            if fold_patience == 0:
                break

        # saving the performance metrics
        save_accs_and_losses(train_losses, train_accuracies, validation_losses, validation_accuracies, i, timestamp, path)
        best_fold_validation_accuracies = np.append(best_fold_validation_accuracies, best_validation_accuracy)
        best_validation_loss = np.min(validation_losses)
        best_fold_validation_losses = np.append(best_fold_validation_losses, best_validation_loss)
        print("--------------------------------------------------------------------------------------------\n")

        # resetting the model
        reset_model_weights(model)
        # model.load_state_dict(torch.load(f"{path}/Models/init_model_{timestamp}"))

    # mean performances of each fold
    validation_file = open(f"{path}/PerformanceMetrics/validation_{timestamp}.txt", "a")
    save_validation_performance_to_file(best_fold_validation_accuracies, best_fold_validation_losses, validation_file)
    validation_file.close()
    
    # saving the benchmarks
    benchmarks_file = open(f"{path}/Benchmarks/training_{timestamp}.txt", "a")
    save_benchmarks_to_file(train_epoch_time, validation_epoch_time, len(folds), pred_times, benchmarks_file)
    benchmarks_file.close()

    # printing the performance
    mean_validation_accuracy = np.mean(best_fold_validation_accuracies)
    mean_validation_loss = np.mean(best_fold_validation_losses)
    print(f"All Best Validation Accuracies: {best_fold_validation_accuracies}")
    print(f"Mean Validation Accuracy: {mean_validation_accuracy}")
    print(f"All Best Validation Losses: {best_fold_validation_losses}")
    print(f"Mean Validation Loss: {mean_validation_loss}")

    return f"{path}/Models"

def compute_f1_scores(confusion_matrix):
    sum_horiz = np.sum(confusion_matrix, axis=1)
    sum_vert = np.sum(confusion_matrix, axis=0)
    true_positives = confusion_matrix.diagonal()

    precisions = true_positives / sum_vert
    recalls = true_positives / sum_horiz

    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    return f1_scores


def compute_ROC_curves(output_scores, test_labels, fold, timestamp, path):
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
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=f'{label_map[i]} fold {fold} estimator')
        display.plot()
        plt.savefig(f"{path}/ROCCurves/{label_map[i]}_ROC_curve_{fold}_{timestamp}.png")
        plt.close()


def update_confusion_matrix(confusion_matrix, prediction, labels):
    for i in range(len(prediction)):
        confusion_matrix[labels[i].item(), prediction[i].item()] += 1

    return confusion_matrix


def test_model(model, loss_fn, test_dataset, test_labels, batch_size, device, fold, timestamp, path):
    # remove the last character if it is a slash
    if path[-1] == '/':
        path = path[:-1]

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
    compute_ROC_curves(output_scores, test_labels, fold, timestamp, path)

    save_metrics_to_file(confusion_matrix, f1_scores, output_scores,
                         f"{path}/PerformanceMetrics/ConfusionMatrix_{fold}_{timestamp}.csv",
                         f"{path}/PerformanceMetrics/F1Scores_{fold}_{timestamp}.csv",
                         f"{path}/PerformanceMetrics/OutputScores_{fold}_{timestamp}.csv")

    return avg_loss, accuracy, confusion_matrix, f1_scores


def test_models(model, models_dir, loss_fn, test_folds, batch_size, device, timestamp, path):
    performance_file = open(f"{path}/PerformanceMetrics/testing_results_{timestamp}.txt", "a")
    test_losses = np.array([])
    test_accuracies = np.array([])

    test_dataset = ConcatDataset(test_folds)
    test_labels_0 = test_folds[0].get_labels()
    test_labels_1 = test_folds[1].get_labels()
    test_labels = np.concatenate((test_labels_0, test_labels_1))

    for saved_model in os.listdir(models_dir):
        if "init" in saved_model:
            continue

        model_path = os.path.join(models_dir, saved_model)
        model.load_state_dict(torch.load(model_path))

        saved_model_fold_num = saved_model[saved_model.index('_') + len("fold") + 1]
        if not saved_model_fold_num.isnumeric():
            saved_model_fold_num = saved_model

        test_loss, test_accuracy, confusion_matrix, f1_scores = test_model(model, loss_fn, test_dataset, test_labels, batch_size, device, saved_model_fold_num, timestamp, path)
        test_losses = np.append(test_losses, test_loss)
        test_accuracies = np.append(test_accuracies, test_accuracy)

    save_performance_metrics_to_file(test_losses, test_accuracies, performance_file)
    performance_file.close()

    print(f"test losses: {test_losses}")
    print(f"mean test loss: {np.mean(test_losses)}")
    print(f"test accuracies: {test_accuracies}")
    print(f"mean test accuracies: {np.mean(test_accuracies)}")
        
