import numpy as np
from torch import save
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

kf = StratifiedKFold(n_splits=5, shuffle=True)
writer = SummaryWriter("/home/damfil/Uni/FYP/PyTorchADNet/sample_logs")

def train_one_epoch(model, dataloader, epoch_idx, sum_writer, opt_fn, loss_fn, device):
    model.train(True)

    running_loss = 0.
    num_correct = 0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, labels = data[0].to(device), data[1].to(device)

        # clear the gradients
        opt_fn.zero_grad()

        # generate the output
        input = input.unsqueeze(1)
        output = model(input)

        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        running_loss += loss.item()
        loss.backward()

        # Adjust learning weights
        opt_fn.step()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader.dataset)

    sum_writer.add_scalar('Loss/train', avg_loss, epoch_idx)
    print(f"[TRAIN]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy

def validate_one_epoch(model, loss_fn, dataloader, epoch_idx, sum_writer, device):
    model.eval()

    running_loss = 0.
    num_correct = 0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        input, labels = data[0].to(device), data[1].to(device)

        input = input.unsqueeze(1)
        output = model(input)

        loss = loss_fn(output, labels)
        running_loss += loss.item()

        prediction = output.argmax(dim=1, keepdim=True)
        num_correct += prediction.eq(labels.view_as(prediction)).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * num_correct / len(dataloader.dataset)

    sum_writer.add_scalar('Loss/val', avg_loss, epoch_idx)

    print(f"[VAL]: Epoch [{epoch_idx}] - Avg. loss per batch: [{avg_loss}] - Accuracy: [{accuracy}%]")

    return avg_loss, accuracy

def train_model(model, opt_fn, loss_fn, dataset, train_labels, batch_size, num_epochs, device):
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
            t_loss, t_acc  = train_one_epoch(model, trainloader, i, writer, opt_fn, loss_fn, device)

            # validate the model on the parameters
            v_loss, v_acc = validate_one_epoch(model, loss_fn, valloader, i, writer, device)

            train_losses = np.append(train_losses, t_loss)
            val_losses = np.append(val_losses, v_loss)
            train_accs = np.append(train_accs, t_acc)
            val_accs = np.append(val_accs, v_acc)

            writer.add_scalars('Training vs. Validation Loss', { 'Training' : t_loss, 'Validation' : v_loss }, i)
            writer.flush()

        if v_loss < best_loss:
            best_loss = v_loss
            model_path = 'model_{}_{}'.format(timestamp, fold)
            save(model.state_dict(), model_path)

        print("--------------------------------------------------------------------------------------------\n")
