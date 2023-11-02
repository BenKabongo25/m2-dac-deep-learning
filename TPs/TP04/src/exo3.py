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
from utils import RNN, device,  ForecastMetroDataset


def many_to_many_train(
    model, 
    loss_fn,
    dataloader, 
    optimizer, 
    writer, 
    epoch_id, 
    verbose=True
):
    model.train()
    train_loss = 0
    for X, Y in dataloader:
        batch, n_classes = X.size(0), X.size(2)
        
        batch_loss = 0
        for c in range(n_classes):
            Xc = X[:, :, c, :].permute(1, 0, 2).to(device)
            Yc = Y[:, :, c, :].permute(1, 0, 2).to(device)

            h_0 = torch.zeros(batch, model.hidden_dim)
            hs = model(Xc, h_0)
            Yc_pred = model.decode(hs)
            loss = loss_fn(Yc_pred, Yc)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss += loss.item()

        batch_loss /= n_classes
        train_loss += batch_loss
    
    train_loss /= len(dataloader)
    writer.add_scalar("Loss/train", train_loss, epoch_id)

    if verbose:
        print(f"Epoch {epoch_id} : \n\tTrain : loss = {train_loss:>8f}")


def many_to_many_test(
    model, 
    loss_fn,
    dataloader, 
    writer, 
    epoch_id, 
    verbose=True
):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, Y in dataloader:
            batch, n_classes = X.size(0), X.size(2)
            
            batch_loss = 0
            for c in range(n_classes):
                Xc = X[:, :, c, :].permute(1, 0, 2).to(device)
                Yc = Y[:, :, c, :].permute(1, 0, 2).to(device)

                h_0 = torch.zeros(batch, model.hidden_dim)
                hs = model(Xc, h_0)
                Yc_pred = model.decode(hs)
                loss = loss_fn(Yc_pred, Yc)
                batch_loss += loss.item()

            batch_loss /= n_classes
            test_loss += batch_loss
            
    test_loss /= len(dataloader)
    writer.add_scalar("Loss/test", test_loss, epoch_id)

    if verbose:
        print(f"\tTest : loss: {test_loss:>8f}")

BASE_PATH = "TPS/TP04/"
PATH = BASE_PATH + "data/"

CLASSES = 3 # Nombre de stations utilisé
LENGTH = 20 #Longueur des séquences 
DIM_INPUT = 2 # Dimension de l'entrée (1 (in) ou 2 (in/out))
BATCH_SIZE = 32 #Taille du batch

HIDDEN_DIM = 20
LEARNING_RATE = 1e-2
N_EPOCHS = 100

STEP = 1


def sequence_prediction(
    train_length=LENGTH,                # longueuer des séquences en train
    test_length=LENGTH,                 # longueur des séquences en test
    input_dim=DIM_INPUT,                # dimension de l'entrée
    hidden_dim=HIDDEN_DIM,              # dimension latente du rnn
    n_classes=CLASSES,                  # nombre de classes
    step=STEP,                          # prédiction au pas t+STEP
    batch_size=BATCH_SIZE,              # taille du batch
    lr=LEARNING_RATE,                   # learning rate
    n_epochs=N_EPOCHS,                  # nombre d'époques
    hidden_activation=nn.Tanh(),        # activation pour le calcul de l'état caché
    output_activation=nn.Sigmoid(),     # activation pour l'output du rnn
    comment="",                         # commentaire pour le writer
    verbose=True                        # verbose
):

    matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
    ds_train = ForecastMetroDataset(
        matrix_train[:, :, :n_classes, :input_dim], length=train_length, step=step)
    ds_test = ForecastMetroDataset(
        matrix_test[:, :, :n_classes, :input_dim], length=test_length, step=step, stations_max=ds_train.stations_max)
    data_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    data_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model = RNN(hidden_dim, input_dim, input_dim, hidden_activation, output_activation).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    writer = SummaryWriter(BASE_PATH + "runs/exo3/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"), comment=comment)
    for epoch_id in tqdm(range(n_epochs), "Training"):
        many_to_many_train(model, loss_fn, data_train, optimizer, writer, epoch_id, verbose=verbose)
        many_to_many_test(model, loss_fn, data_test, writer, epoch_id, verbose=verbose)

    return model


if __name__ == "__main__":

    # t+1
    _ = sequence_prediction(
        train_length=LENGTH, 
        test_length=LENGTH, 
        input_dim=DIM_INPUT, 
        hidden_dim=HIDDEN_DIM,
        n_classes=CLASSES,
        step=STEP,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        hidden_activation=nn.Tanh(),
        output_activation=nn.Sigmoid(),
        comment="t+1",
        verbose=False
    )

    # t+2
    _ = sequence_prediction(
        train_length=LENGTH, 
        test_length=LENGTH, 
        input_dim=DIM_INPUT, 
        hidden_dim=HIDDEN_DIM,
        n_classes=CLASSES,
        step=STEP+1,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        hidden_activation=nn.Tanh(),
        output_activation=nn.Sigmoid(),
        comment="t+2",
        verbose=False
    )

    # t+3
    _ = sequence_prediction(
        train_length=LENGTH, 
        test_length=LENGTH, 
        input_dim=DIM_INPUT, 
        hidden_dim=HIDDEN_DIM,
        n_classes=CLASSES,
        step=STEP+2,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        hidden_activation=nn.Tanh(),
        output_activation=nn.Sigmoid(),
        comment="t+3",
        verbose=False
    )

