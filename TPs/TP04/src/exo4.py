# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import datetime
import string
import sys
import torch
import torch.nn as nn
import unicodedata
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

BATCH_SIZE = 16
VOCAB_SIZE = len(lettre2id)
EMBEDDING_DIM = 30
HIDDEN_DIM = 20
MAX_LENGTH = 100

LR = 1e-3
N_EPOCHS = 50

data_trump = DataLoader(
    TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000),
    batch_size=BATCH_SIZE, 
    shuffle=True
)


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
        embeddings = self.embedding(x) # x.size() = (batch, 1), embeddings.size() = (batch, 1, embedding_dim)
        h = self.rnn(embeddings.permute(1, 0, 2), h).squeeze() # h.size() = (batch, hidden_dim)
        output = self.rnn.decode(h) # output.size() = (batch, vocab_size)
        return h, output

    def fit(self, dataloader, lr=LR, n_epochs=N_EPOCHS, verbose=True):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch_id in tqdm(range(n_epochs), "Training"):
            
            total_loss = 0
            for (seq, pred) in dataloader:
                seq, pred = seq.to(device), pred.to(device)
                batch, length = seq.size(0), seq.size(1)
                total_loss = 0
                h = torch.zeros(batch, self.hidden_dim).to(device)
                for i in range(length):
                    h, output = self.forward(seq[:, i].unsqueeze(1), h)
                    loss = loss_fn(output, pred[:, i].squeeze())
                    total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar("Loss/train", total_loss, epoch_id)

            if verbose and epoch_id % 10 == 0:
                print(f"[Epoch {epoch_id + 1}/{n_epochs}] : loss = {total_loss:>8f}")
                start = list(lettre2id.keys())
                texts = model.generate(start, length=50)
                print("===============================================")
                for s, text in zip(start, texts):
                    print(s + text + "\n")

    def generate(self, start=[''], length=MAX_LENGTH):
        seq = torch.tensor([[lettre2id[c]] for c in start])
        prediction = seq[:, -1]
        all_predictions = []
        with torch.no_grad():
            h = torch.zeros(seq.size(0), self.hidden_dim).to(device)
            for _ in range(length):
                h, output = self.forward(prediction.unsqueeze(1), h)
                prediction = torch.softmax(output, dim=1).argmax(1)
                all_predictions.append(code2string(prediction))
        return all_predictions


if __name__ == "__main__":
    model = TrumpModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    model.fit(data_trump, LR, N_EPOCHS, verbose=True)
    
    start = list(lettre2id.keys())
    texts = model.generate(start, MAX_LENGTH)

    print("===============================================")
    for s, text in zip(start, texts):
        print(s + text + "\n")
