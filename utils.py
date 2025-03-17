import numpy as np

import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.functional import F
import torch.nn as nn
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics import Metric
import copy

def precompute_features(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is 
    the rest of the model, it is not necessary to recompute $g(x)$ at 
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and 
    create a new dataset 
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation
    
    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    print("Please, be sure to provide the full model (not just the feature extractor) as this function expects a model with a last layer.")
    g_model = nn.Sequential(*list(model.children())[:-1]).to(device)
    g_dataset = []
    for x, y in dataset:
        with torch.no_grad():
            x = x.to(device)
            # change shape of x so that it is a 4D tensor
            x = x.unsqueeze(0)
            g_x = g_model(x)
            g_x = g_x.view(x.size(0), -1)
            g_x = g_x.squeeze(0)
        g_dataset.append((g_x, y))
    g_dataset = torch.utils.data.TensorDataset(torch.stack([x for x, _ in g_dataset]), torch.tensor([y for _, y in g_dataset]))
    return g_dataset

def train(
    model: nn.Module, train_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int = 10, val_loader: torch.utils.data.DataLoader | None = None, num_classes: int | None = None, print_every: int = 5, original_model: models.ResNet | None = None, augment_fn = None, keep_best: bool = False
) -> nn.Module:
    """
    Train our model.

    Arguments:
    ----------
    model: nn.Module
        our model
    train_loader: torch.utils.data.DataLoader
        Training set / or the precomputed features of the training set
    criterion: nn.Module
        A loss function
    optimizer: torch.optim.Optimizer
        An optimizer
    device: torch.device
        Device : cpu or cuda
    num_epochs: int
        Number of epochs
    val_loader: torch.utils.data.DataLoader | None
        If not None, the validation set / or the precomputed features of the validation set
    num_classes: int | None
        Number of classes for the classification task
    print_every: int
        Print the loss & accuracy every print_every epochs
    original_model: models.ResNet | None
        The original model used to precompute the features (for the evaluation)
    augment_fn: should be a function (x,y)->(x',y') | None
        A function to augment the data (cutmix, mixup, etc.)
    keep_best: bool
        If True, keep the best model based on the validation loss
    
    Returns:
    --------
    nn.Module
        The trained model
    """
    if keep_best and val_loader is not None and num_classes is not None:
        best_acc = 0.0
        best_model = copy.deepcopy(model)
    if augment_fn is not None:
        print("It is recommended to not use data augmentation with precomputed features.")
    if original_model is not None:
        print("Please, be sure that you have provided the orignal test data and not the precomputed one.")
    dict = {}
    dict["loss"] = []
    dict["val_loss"] = []
    if num_classes is not None:
        accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        accuracy = accuracy.to(device)
        dict["accuracy"] = []
        dict["val_accuracy"] = []
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        detach_loss, acc = 0.0, 0.0
        for batch in train_loader:
            x, y = batch
            if augment_fn is not None:
                x, y = augment_fn(x, y)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            detach_loss += loss.cpu().detach().numpy()
            if num_classes is not None:
                acc += accuracy(y_pred, y).cpu().detach().numpy()
            loss.backward()
            optimizer.step()
        detach_loss /= len(train_loader)
        dict["loss"].append(detach_loss)
        if num_classes is not None:
            acc /= len(train_loader)
            dict["accuracy"].append(acc)
        if val_loader is not None and (original_model is None or epoch % print_every == 0):
            if original_model is not None:
                original_model.fc = model
                test_model = original_model
            else:
                test_model = model
            test_model.eval()
            with torch.no_grad():
                val_loss, val_acc = 0.0, 0.0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = test_model(x)
                    val_loss += criterion(y_pred, y)
                    if num_classes is not None:
                        val_acc += accuracy(y_pred, y)
                val_loss /= len(val_loader)
                if original_model is not None:
                    for _ in range(print_every):
                        dict["val_loss"].append(val_loss.cpu().detach().numpy())
                else:
                    dict["val_loss"].append(val_loss.cpu().detach().numpy())
                if num_classes is not None:
                    val_acc /= len(val_loader)
                    if keep_best and val_acc > best_acc:
                        best_acc = val_acc
                        best_model = copy.deepcopy(model)
                    if original_model is not None:
                        for _ in range(print_every):
                            dict["val_accuracy"].append(val_acc.cpu().detach().numpy())
                    else:
                        dict["val_accuracy"].append(val_acc.cpu().detach().numpy())
                    if epoch % print_every == 0:
                        print(f"Epoch {epoch+1}/{num_epochs} with validation loss: {val_loss} and accuracy: {val_acc}")
                else:
                    if epoch % print_every == 0:
                        print(f"Epoch {epoch+1}/{num_epochs} with validation loss: {val_loss}")
    if val_loader is not None:
        if num_classes is not None:
            # training curve with accuracy
            plt.plot(dict["accuracy"], label="accuracy")
            plt.plot(dict["val_accuracy"], label="val_accuracy")
            plt.legend()
            plt.show()
        # training curve with loss
        plt.plot(dict["loss"], label="loss")
        plt.plot(dict["val_loss"], label="val_loss")
        plt.legend()
        plt.show()
    else :
        if num_classes is not None:
            # training curve with accuracy
            plt.plot(dict["accuracy"], label="accuracy")
            plt.legend()
            plt.show()
        # training curve with loss
        plt.plot(dict["loss"], label="loss")
        plt.show()
    if keep_best and val_loader is not None and num_classes is not None:
        print(f"Best model with accuracy: {best_acc}")
        return best_model
    return model

def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, num_classes: int | None = None) -> None:
    """
    Evaluate the model.

    Arguments:
    ----------
    model: nn.Module
        The model to evaluate
    train_loader: torch.utils.data.DataLoader
        The training set / or the precomputed features of the training set
    device: torch.device
        Device : cpu or cuda
    num_classes: int | None
        Number of classes for the classification task
    """
    if num_classes is not None:
        accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        accuracy = accuracy.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        acc = 0.0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            if num_classes is not None:
                acc += accuracy(y_pred, y).cpu().detach().numpy()
        if num_classes is not None:
            acc /= len(data_loader)
            print(f"Accuracy: {acc}")
        else:
            print("No num_classes provided : No evaluation done")


class LastLayer(nn.Module):
    def __init__(self, in_features: int | None = None, out_features: int | None = None):
        super(LastLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if in_features is not None and out_features is not None:
            self.linear = nn.Linear(in_features, out_features)
        else:
            self.linear = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


class FinalModel(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        # <YOUR CODE>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <YOUR CODE>
        raise NotImplementedError("Implement the forward pass of the LastLayer module")