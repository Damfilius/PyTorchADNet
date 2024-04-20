import numpy as np
from torch import save
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
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


def test_model(model, loss_fn, test_dataset, batch_size, device):
    model.train(False)
    test_loader = DataLoader(test_dataset, batch_size)
    running_loss = 0
    num_correct = 0

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs = inputs.usqueeze(1)
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        prediction = outputs.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * num_correct / len(test_loader.dataset)

    print(f"[TEST]: Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy
