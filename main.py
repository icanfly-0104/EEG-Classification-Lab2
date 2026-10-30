import torch
import argparse
import dataloader
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import DeepConvNet, EEGNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader, Subset

RESULT_DIR = Path(__file__).resolve().parent / 'result'
PLOT_DIR = RESULT_DIR

class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(train_acc_list, epochs):
    epoch_numbers = range(1, len(train_acc_list) + 1)
    plt.figure()
    plt.plot(epoch_numbers, train_acc_list)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / 'training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_train_loss(train_loss_list, epochs):
    epoch_numbers = range(1, len(train_loss_list) + 1)
    plt.figure()
    plt.plot(epoch_numbers, train_loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / 'training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_validation_acc(validation_acc_list, epochs):
    epoch_numbers = range(1, len(validation_acc_list) + 1)
    plt.figure()
    plt.plot(epoch_numbers, validation_acc_list)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / 'validation_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_validation_loss(validation_loss_list, epochs):
    epoch_numbers = range(1, len(validation_loss_list) + 1)
    plt.figure()
    plt.plot(epoch_numbers, validation_loss_list)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropyLoss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / 'validation_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def split_train_validation(labels, validation_split, seed):
    """Split by class so both train and validation retain each label."""
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1 (exclusive).")
    
    rng = np.random.default_rng(seed)
    train_indices, validation_indices = [], []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_validation = max(1, int(round(len(indices) * validation_split)))
        if n_validation >= len(indices):
            raise ValueError("validation_split leaves no training samples for at least one class.")
        validation_indices.extend(indices[:n_validation])
        train_indices.extend(indices[n_validation:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices

def train(model, loader, validation_loader, criterion, optimizer, args):
    best_wts = None
    best_validation_loss = float('inf')
    best_validation_acc = 0.0
    best_epoch = 0
    avg_acc_list = []
    validation_acc_list = []
    validation_loss_list = []
    avg_loss_list = []

    # Training Loop
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        validation_acc = evaluate_accuracy(model, validation_loader)
        validation_loss = evaluate_loss(model, validation_loader, criterion)
        validation_acc_list.append(validation_acc)
        validation_loss_list.append(validation_loss)

        # Update model when accuracy improve and loss improve.
        if validation_acc >= best_validation_acc:
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_acc = validation_acc
                best_wts = deepcopy(model.state_dict())
                best_epoch = epoch
        print(f'Validation Acc. (%): {validation_acc:3.2f}%')
        print(f'Validation Loss: {validation_loss:.8f}')

    weights_dir = PLOT_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_wts, weights_dir / 'best.pt')
    model.load_state_dict(best_wts)
    return avg_acc_list, avg_loss_list, validation_acc_list, validation_loss_list, best_validation_loss, best_validation_acc, best_epoch

def evaluate_loss(model, loader, criterion):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
    return total_loss / len(loader.dataset)

def evaluate_accuracy(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc

if __name__ == '__main__':
    # Parser and argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.01)
    parser.add_argument("-activation", "--activation", choices=["elu", "relu", "leaky_relu"], default="relu")
    parser.add_argument("-elu_alpha", "--elu-alpha", type=float, default=1.0)
    parser.add_argument("-dropout", "--dropout", type=float, default=0.25)
    parser.add_argument("-model", "--model", choices=["eegnet", "deepconvnet"], default="eegnet")
    parser.add_argument("-f1", "--f1", type=int, default=16, help="EEGNet temporal filter count.")
    parser.add_argument("-d", "--depth-multiplier", type=int, default=2, help="EEGNet depth multiplier.")
    parser.add_argument("-f2", "--f2", type=int, default=None, help="EEGNet separable filter count (default: f1 * d).")
    parser.add_argument("--temporal-kernel", type=int, default=51, help="EEGNet first temporal convolution kernel width.")
    parser.add_argument("--separable-kernel", type=int, default=15, help="EEGNet depthwise separable temporal kernel width.")
    parser.add_argument("--pool1", type=int, default=4, help="EEGNet first average-pooling width and stride.")
    parser.add_argument("--pool2", type=int, default=8, help="EEGNet second average-pooling width and stride.")
    parser.add_argument("-optimizer", "--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("-run_name", "--run-name", default=None)
    parser.add_argument("-seed", "--seed", type=int, default=42)
    parser.add_argument("-validation_split", "--validation-split", type=float, default=0.1,
                        help="Fraction of the original training set used for validation (default: 0.1).")
    args = parser.parse_args()

    # Check if argument makes sense.
    if args.num_epochs < 1:
        parser.error("-num_epochs must be a positive integer.")
    if args.batch_size < 1:
        parser.error("-batch_size must be a positive integer.")
    if args.f1 < 1 or args.depth_multiplier < 1 or (args.f2 is not None and args.f2 < 1):
        parser.error("-f1, -d, and -f2 (when provided) must be positive integers.")
    if args.temporal_kernel < 1 or args.separable_kernel < 1:
        parser.error("Kernel widths must be positive integers.")
    if args.pool1 < 1 or args.pool2 < 1:
        parser.error("Pooling widths must be positive integers.")
    if not 0 < args.validation_split < 1:
        parser.error("-validation_split must be between 0 and 1 (exclusive).")

    # Result dir and folder
    run_name = args.run_name or (
        f"{args.optimizer}_lr{args.lr:g}_bs{args.batch_size}_ep{args.num_epochs}_"
        f"{args.activation}_dropout{args.dropout:g}_val{args.validation_split:g}"
    )
    run_dir = RESULT_DIR / run_name
    PLOT_DIR = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(f"Experiment: {run_name}; optimizer={args.optimizer}; lr={args.lr}; batch_size={args.batch_size}; "
          f"epochs={args.num_epochs}; activation={args.activation}; dropout={args.dropout}; seed={args.seed}")

    # Data loader.
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)

    # Validation set split.
    train_indices, validation_indices = split_train_validation(train_label, args.validation_split, args.seed)
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(train_dataset, validation_indices)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set seed.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set model.
    if args.model == "deepconvnet":
        model = DeepConvNet(activation=args.activation, elu_alpha=args.elu_alpha, dropout=args.dropout)
    else:
        model = EEGNet(activation=args.activation, elu_alpha=args.elu_alpha, dropout=args.dropout,
                       f1=args.f1, d=args.depth_multiplier, f2=args.f2,
                       temporal_kernel=args.temporal_kernel, separable_kernel=args.separable_kernel,
                       pool1=args.pool1, pool2=args.pool2)

    # Set optimizer.
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
        optimizer_description = "Adam"
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
        optimizer_description = "SGD (momentum=0.9)"

    # Training
    model.to(device)
    criterion.to(device)
    (train_acc_list, train_loss_list, validation_acc_list, validation_loss_list,
     best_validation_loss, best_validation_acc, best_epoch) = train(
        model, train_loader, validation_loader, criterion, optimizer, args
    )

    # Final testing.
    final_test_accuracy = evaluate_accuracy(model, test_loader)

    # Save history.
    np.savetxt(run_dir / "history.csv",
               np.column_stack((np.arange(1, args.num_epochs + 1), train_acc_list, train_loss_list,
                                validation_acc_list, validation_loss_list)),
               delimiter=",", header="epoch,train_accuracy,train_loss,validation_accuracy,validation_loss", comments="")

    # Save accuracy and other information.
    (run_dir / "highest_accuracy.txt").write_text(
        f"Best validation loss: {best_validation_loss:.8f}\n"
        f"Validation accuracy at selected checkpoint: {best_validation_acc:.2f}%\nSelected checkpoint epoch: {best_epoch}\n"
        f"Final test accuracy (evaluated once): {final_test_accuracy:.2f}%\n"
        f"Activation: {args.activation}\nELU alpha: {args.elu_alpha}\nDropout: {args.dropout}\n"
        f"Learning rate: {args.lr}\nBatch size: {args.batch_size}\nEpochs: {args.num_epochs}\n"
        f"Validation split: {args.validation_split}\n"
        f"Optimizer: {optimizer_description}\nLoss: CrossEntropyLoss\nWeight decay: 0.001\n"
        f"Seed: {args.seed}\nF1: {args.f1}\nDepth multiplier: {args.depth_multiplier}\n"
        f"F2: {args.f2 if args.f2 is not None else args.f1 * args.depth_multiplier}\n"
        f"Temporal kernel: {args.temporal_kernel}\nSeparable kernel: {args.separable_kernel}\n"
        f"Pool1: {args.pool1}\nPool2: {args.pool2}\n", encoding="utf-8")

    # Plot the curves.
    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_validation_acc(validation_acc_list, args.num_epochs)
    plot_validation_loss(validation_loss_list, args.num_epochs)

    # Print the result.
    print(f"Best Validation Loss: {best_validation_loss:.8f} at epoch {best_epoch}", flush=True)
    print(f"Validation Acc. at selected checkpoint (%): {best_validation_acc:3.2f}%", flush=True)
    print(f"Final Test Acc. (%): {final_test_accuracy:3.2f}%", flush=True)
    print(f"Run artifacts saved to: {run_dir}", flush=True)
