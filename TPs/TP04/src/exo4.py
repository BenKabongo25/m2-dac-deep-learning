# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import datetime
import string
import os
import sys
import torch
import torch.nn as nn
import unicodedata
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import RNN, device


## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TrumpDataset(Dataset):
    def __init__(self, text, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
        :param text : texte brut
        :param maxsent : nombre maximum de phrases.
        :param maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long), t])
        return t[:-1],t[1:]


BASE_PATH = "TPS/TP04/"
PATH = BASE_PATH + "data/"

BATCH_SIZE = 8
VOCAB_SIZE = len(lettre2id)
EMBEDDING_DIM = 50
HIDDEN_DIM = 40
MAX_LENGTH = 100

LR = 1e-2
N_EPOCHS = 100
VERBOSE_EVERY = 5

data_trump = DataLoader(
    TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000),
    batch_size=BATCH_SIZE, 
    shuffle=True
)

class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0


writer = SummaryWriter(BASE_PATH + "/runs/exo4/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))


class TrumpModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = RNN(hidden_dim, embedding_dim, vocab_size)

    def forward(self, x, h):
        embeddings = self.embedding(x)  # x.size() = (batch, length), embeddings.size() = (batch, length, embedding_dim)
        if embeddings.ndim == 2: embeddings = embeddings.unsqueeze(1)
        hs = self.rnn(embeddings.transpose(0, 1), h) # hs.size() = (length, batch, hidden_dim)
        output = self.rnn.decode(hs) # output.size() = (length, batch, vocab_size)
        return hs, output


def train(state, dataloader, lr=LR, n_epochs=N_EPOCHS, verbose=True, model_path=None):
    loss_fn = nn.CrossEntropyLoss()

    for epoch_id in tqdm(range(state.epoch, n_epochs), "Training"):
            
        total_loss = 0
        for (source, target) in dataloader:
            source, target = source.to(device), target.to(device)
            total_loss = 0
            h0 = torch.zeros(source.size(0), state.model.hidden_dim).to(device)
            _, output = state.model(source, h0)
            loss = loss_fn(
                output.permute(1, 0, 2).reshape(-1, state.model.vocab_size),
                target.reshape(-1)
            )
            total_loss += loss.item()
            state.optimizer.zero_grad()
            loss.backward()
            state.optimizer.step()

        total_loss /= len(dataloader)
        writer.add_scalar("Loss/train", total_loss, epoch_id)

        if verbose and epoch_id % VERBOSE_EVERY == 0:
            print(f"[Epoch {epoch_id + 1}/{n_epochs}] : loss = {total_loss:>8f}")
            start = list(lettre2id.keys())[:10]
            texts = generate(state.model, start, length=100)
            print()
            for s, text in zip(start, texts):
                print(s + text + "\n")

        state.epoch = epoch_id + 1
        with model_path.open("wb") as fp:
            torch.save(state, fp)


def generate(model, start=[''], length=MAX_LENGTH):
    source = torch.tensor([[lettre2id[c]] for c in start])
    batch_size = source.size(0)
    prediction = source[:, -1].squeeze()
    all_predictions = torch.zeros(batch_size, length)
    with torch.no_grad():
        h = torch.zeros(source.size(0), model.hidden_dim).to(device)
        for i in range(length):
            h, output = model.forward(prediction, h)
            prediction = torch.softmax(output.squeeze(), dim=1).argmax(1).squeeze()
            all_predictions[:, i] = prediction
    texts = [code2string(t) for t in all_predictions]
    return texts


if __name__ == "__main__":
    MODEL_PATH = BASE_PATH + "/models/"
    os.makedirs(MODEL_PATH, exist_ok=True)
    MODEL_PATH += "trump_model.pch"
    load_model = False

    model_path = Path(MODEL_PATH)
    if load_model and model_path.is_file():
        if model_path.is_file():
            with model_path.open("rb") as fp:
                state = torch.load(fp)
    else :
        model = TrumpModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
        model = model.to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        state = State(model, optimizer)
        with model_path.open("wb") as fp:
            torch.save(state, fp)

    train(state, data_trump, LR, N_EPOCHS, verbose=True, model_path=model_path)
    
    start = list(lettre2id.keys())
    texts = generate(state.model, start, MAX_LENGTH)
    print("")
    for s, text in zip(start, texts):
        print(s + text + "\n")
