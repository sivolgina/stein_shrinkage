import torch
from train import *
from evaluate import *
from stein_corrected_bn import *
from mean_corrected_bn import *
from lasso_ridge_bn import *
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import os
from resnet import * ### here specify model
from utils import *
import torchattacks
import sys

if len(sys.argv) > 1:
    BN_TYPE = sys.argv[1]
else:
    BN_TYPE = 'stein'

if len(sys.argv) > 2:
    ATTACK_TYPE = sys.argv[2]
else:
    ATTACK_TYPE = 'FGSM'   

#specify model, criterion, dataloader and level of noise
### here we provide example with cifar10 data and stein_corrected_bn
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
])

full_train_dataset = torchvision.datasets.CIFAR10(
    root='/data',
    train=True,
    download=False,
    transform=transform_tensor
)


val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
generator = torch.Generator().manual_seed(6)
train_indices, val_indices = torch.utils.data.random_split(
    range(len(full_train_dataset)), [train_size, val_size], generator=generator
)

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset= Subset(full_train_dataset, val_indices)

test_dataset = torchvision.datasets.CIFAR10(
    root='/data',
    train=False,
    download=False,
    transform=transform_tensor
)

bs=128
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)
#################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_runs = 1
num_epochs = 5
patience = 5
log_path = 'logs/results.txt'
init_log_file(log_path)

for _ in range(num_runs):
    ###specify model, criterion, optimizer and attack###
    model = ResNet9()
    if BN_TYPE == 'stein':
        model = convert_batchnorm_stein(model).to(device)
    elif BN_TYPE == 'lasso':
        model = convert_batchnorm_lasso(model).to(device)
    elif BN_TYPE == 'ridge':
        model = convert_batchnorm_ridge(model).to(device)
    elif BN_TYPE == 'mean':
        model = convert_batchnorm_mean(model).to(device)
    elif BN_TYPE == 'vanilla':
        model = model.to(device)
    else:
        raise ValueError(f"Unknown BN type: {BN_TYPE}")

    if ATTACK_TYPE == "FGSM":
        attack = torchattacks.FGSM(model, eps=0.03)
    elif ATTACK_TYPE == "PGD":
        attack = torchattacks.PGD(model, eps=0.03, alpha=1/255, steps=5)
    else:
        raise ValueError(f"Unknown ATTACK_TYPE: {ATTACK_TYPE}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)  
    #########################################
    best_val_accuracy = 0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_clean_loss, val_clean_acc = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}: TrainAcc={train_acc:.2f}, ValCleanAcc={val_clean_acc:.2f}')
        log_epoch(log_path, epoch+1, [train_loss, train_acc, val_clean_loss, val_clean_acc])

        if val_clean_acc > best_val_accuracy:
            best_val_accuracy = val_clean_acc
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    test_accuracy_attack = validate_with_attack(model, test_loader, attack, device)
    log_test(log_path, test_accuracy_attack)
