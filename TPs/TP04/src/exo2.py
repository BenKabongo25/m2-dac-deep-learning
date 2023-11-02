# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import datetime
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import RNN, device, SampleMetroDataset


def many_to_one_train(
    model, 
    loss_fn, 
    accuracy_fn, 
    dataloader, 
    optimizer, 
    writer, 
    epoch_id, 
    verbose=True
):
    model.train()
    train_loss = 0
    for X, y in dataloader:
        X, y = X.permute(1, 0, 2).to(device), y.to(device)
        batch = X.size(1)

        h_0 = torch.zeros(batch, model.hidden_dim)
        h_n = model(X, h_0)[-1, :, :]
        y_pred = model.decode(h_n)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        accuracy_fn(y_pred.argmax(1), y)
    
    train_loss /= len(dataloader)
    accuracy = accuracy_fn.compute()
    accuracy_fn.reset()
    writer.add_scalar("Loss/train", train_loss, epoch_id)
    writer.add_scalar("Accuracy/train", accuracy, epoch_id)

    if verbose and epoch_id % 10 == 0:
        print(f"Epoch {epoch_id} : \n\tTrain : acc = {(accuracy):>0.4f}%, loss = {train_loss:>8f}")


def many_to_one_test(
    model, 
    loss_fn, 
    accuracy_fn, 
    dataloader, 
    writer, 
    epoch_id, 
    verbose=True
):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.permute(1, 0, 2).to(device), y.to(device)
            batch = X.size(1)

            h_0 = torch.zeros(batch, model.hidden_dim)
            h_n = model(X, h_0)[-1, :, :]
            y_pred = model.decode(h_n)

            test_loss += loss_fn(y_pred, y).item()
            accuracy_fn(y_pred.argmax(1), y)
            
    test_loss /= len(dataloader)
    accuracy = accuracy_fn.compute()
    accuracy_fn.reset()
    writer.add_scalar("Loss/test", test_loss, epoch_id)
    writer.add_scalar("Accuracy/test", accuracy, epoch_id)

    if verbose and epoch_id % 10 == 0:
        print(f"\tTest : acc = {(accuracy):>0.4f}%, loss: {test_loss:>8f}")

BASE_PATH = "TPS/TP04/"
PATH = BASE_PATH + "data/"

CLASSES = 2 # Nombre de stations utilisé
LENGTH = 20 #Longueur des séquences 
DIM_INPUT = 2 # Dimension de l'entrée (1 (in) ou 2 (in/out))
BATCH_SIZE = 32 #Taille du batch

HIDDEN_DIM = 20
LEARNING_RATE = 1e-2
N_EPOCHS = 30


def sequence_classification(
    train_length=LENGTH,                # longueuer des séquences en train
    test_length=LENGTH,                 # longueur des séquences en test
    input_dim=DIM_INPUT,                # dimension de l'entrée
    hidden_dim=HIDDEN_DIM,              # dimension latente du rnn
    n_classes=CLASSES,                  # nombre de classes
    batch_size=BATCH_SIZE,              # taille du batch
    lr=LEARNING_RATE,                   # learning rate
    n_epochs=N_EPOCHS,                  # nombre d'époques
    hidden_activation=nn.Tanh(),        # activation pour le calcul de l'état caché
    output_activation=nn.Sigmoid(),     # activation pour l'output du rnn
    comment="",                         # commentaire pour le writer
    verbose=True                        # verbose
):

    matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
    ds_train = SampleMetroDataset(matrix_train[:, :, :n_classes, :input_dim], length=train_length)
    ds_test = SampleMetroDataset(matrix_test[:, :, :n_classes, :input_dim], length=test_length, stations_max=ds_train.stations_max)
    data_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    data_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model = RNN(hidden_dim, input_dim, n_classes, hidden_activation, output_activation).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    accuracy_train = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=n_classes
    )
    accuracy_test = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=n_classes
    )

    writer = SummaryWriter(BASE_PATH + "runs/exo2/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"), comment=comment)
    for epoch_id in tqdm(range(n_epochs), "Training"):
        many_to_one_train(model, loss_fn, accuracy_train, data_train, optimizer, writer, epoch_id, verbose=verbose)
        many_to_one_test(model, loss_fn, accuracy_test, data_test, writer, epoch_id, verbose=verbose)

    return model


if __name__ == "__main__":
    
    _ = sequence_classification(
        train_length=LENGTH, 
        test_length=LENGTH, 
        input_dim=DIM_INPUT, 
        hidden_dim=HIDDEN_DIM,
        n_classes=CLASSES,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        hidden_activation=nn.Tanh(),
        output_activation=nn.Sigmoid(),
        comment="",
        verbose=False
    )

    TEST_LENGTH = 30
    print(f"Test length = {TEST_LENGTH}")
    _ = sequence_classification(
        train_length=LENGTH, 
        test_length=TEST_LENGTH, 
        input_dim=DIM_INPUT, 
        hidden_dim=HIDDEN_DIM,
        n_classes=CLASSES,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        hidden_activation=nn.Tanh(),
        output_activation=nn.Sigmoid(),
        comment=f"Test length = {TEST_LENGTH}",
        verbose=False
    )
