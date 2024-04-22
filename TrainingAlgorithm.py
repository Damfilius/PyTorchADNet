import numpy as np
from torch import save
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from Utils import save_metrics_to_file

label_map = {
    0: "CN",
    1: "AD",
    2: "MCI",
}


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


def compute_f1_scores(confusion_matrix):
    sum_horiz = np.sum(confusion_matrix, axis=1)
    sum_vert = np.sum(confusion_matrix, axis=0)
    true_positives = confusion_matrix.diagonal()

    precisions = true_positives / sum_vert
    recalls = true_positives / sum_horiz

    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    return f1_scores


def compute_ROC_curves(output_scores, test_labels):
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
    test_loader = DataLoader(test_dataset, batch_size)
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
                         "PerfoermanceMetrics/ConfusionMatrix.csv",
                         "PerfoermanceMetrics/F1Scores.csv",
                         "PerfoermanceMetrics/ConfusionMatrix.csv")

    return avg_loss, confusion_matrix, f1_scores
