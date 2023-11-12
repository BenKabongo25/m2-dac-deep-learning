# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import datetime
import logging
import random
import re
import string
import time
import torch
import unicodedata
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List


def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]


class TradDataset():
    def __init__(self, data, vocOrig, vocDest, adding=True, max_len=10):
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig, dest=map(normalize,s.split("\t")[:2])
            if len(orig) > max_len: continue
            self.sentences.append((
                torch.tensor(
                    [vocOrig.get(o) for o in orig.split(" ")] +
                    [Vocabulary.EOS]
                ),
                torch.tensor(
                    [vocDest.get(o) for o in dest.split(" ")] +
                    [Vocabulary.EOS]
                )
            ))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i): 
        return self.sentences[i]


def collate_fn(batch):
    orig, dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


class Encoder(nn.Module):
    def __init__(self, n_tokens, embedding_dim, hidden_size, padding_idx=0, num_layers=1, dropout=0,
                device=torch.device("cpu")):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim, padding_idx).to(device)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True).to(device)

    def forward(self, X):
        embedded = self.embedding(X)
        _, h_n = self.rnn(embedded)
        #print(f">>> Encoder : X = {X.shape}, h_n = {h_n.shape}")
        return h_n


class Decoder(nn.Module):
    def __init__(self, n_tokens, embedding_dim, hidden_size, padding_idx=0, num_layers=1, dropout=0, 
                sos_idx=Vocabulary.SOS, eos_idx=Vocabulary.EOS, device=torch.device("cpu")):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device
        self.embedding = nn.Embedding(n_tokens, embedding_dim, padding_idx).to(device)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, n_tokens).to(device)

    def forward(self, X, h):
        #print(f">>> Decoder : X = {X.shape}, h = {h.shape} ", end="")
        embedded = self.embedding(X).unsqueeze(1)
        #print(f"embedded = {embedded.shape} ", end="")
        _, h_n = self.rnn(embedded, h)
        #print(f"h_n = {h_n.shape} ", end="")
        output = self.fc(h_n)
        #print(f"output = {output.shape}")
        return h_n, output

    def generate(self, hidden, length=None, constraint_mode=True, target=None, train=False):
        index = 0
        batch_size = hidden.size(1)
        eos_cpt = 0
        logits = []

        input = torch.LongTensor([self.sos_idx]).repeat(batch_size).to(self.device)
        while index != length:
            hidden, output = self.forward(input, hidden)
            logits.append(output)
            if not constraint_mode:
                input = torch.softmax(output, dim=-1).argmax(dim=-1)[0]
                if input.size(0) > 1: input = input.squeeze()
            else:
                input = target[:, index]
            eos_cpt += torch.sum(input == self.eos_idx).item()
            if not train and eos_cpt == batch_size:
                break
            index += 1

        return torch.cat(logits, dim=0).transpose(0, 1)
        

class TraductionModel(nn.Module):
    def __init__(self, source_vocab, dest_vocab, encoder_embedding_dim, decoder_embedding_dim,
                hidden_size, padding_idx=0, device=torch.device("cpu")):
        super().__init__()
        self.source_vocab = source_vocab
        self.dest_vocab = dest_vocab
        self.device = device
        self.encoder = Encoder(len(source_vocab), encoder_embedding_dim, hidden_size, padding_idx, device=device)
        self.decoder = Decoder(len(dest_vocab), decoder_embedding_dim, hidden_size, padding_idx,
                            sos_idx=dest_vocab.SOS, eos_idx=dest_vocab.EOS, device=device)

    def forward(self, source, target, max_length, constraint_mode=True, train=True):
        #print(f">>> Constraint mode = {constraint_mode}, train = {train}")
        source_h_n = self.encoder(source)
        if train: max_length = target.size(1)
        logits = self.decoder.generate(source_h_n, max_length, constraint_mode, target, train)
        return logits

    def predict(self, source_text, max_length=20):
        #print(">>> Prédiction")
        source_text = normalize(source_text)
        source = torch.tensor(
                    [self.source_vocab.get(o) for o in source_text.split(" ")] +
                    [self.source_vocab.EOS]
                ).unsqueeze(0).to(self.device)
        logits = self.forward(source, target=None, max_length=max_length, constraint_mode=False, train=False)
        predicts = logits.argmax(-1).squeeze()
        target_text = self.dest_vocab.getwords(predicts.tolist())
        return ' '.join(target_text)


def train_step(model, optimizer, loss_fn, dataloader, constraint_mode=True, device=torch.device("cpu")):
    model.train()
    total_loss = 0
    for (source, _, target, _) in dataloader:
        source, target = source.T.to(device), target.T.to(device)
        logits = model(source, target, max_length=target.size(1), constraint_mode=constraint_mode, train=True)
        logits, target = logits.flatten(end_dim=1), target.flatten()
        loss = loss_fn(logits, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss


def validate(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (source, _, target, _) in dataloader:
            source, target = source.T.to(device), target.T.to(device)
            logits = model(source, target, max_length=target.size(1), constraint_mode=False, train=True)
            logits, target = logits.flatten(end_dim=1), target.flatten()
            loss = loss_fn(logits, target)
            total_loss += loss.item()
    return total_loss


def train(
        model, 
        optimizer, 
        device,
        loss_fn, 
        train_loader,
        test_loader,
        writer,
        n_epochs=20,
    ):
    for epoch in tqdm(range(1, n_epochs+1), "Training"):
        constraint_mode_probability = (n_epochs-epoch-1) / n_epochs
        constraint_mode = (random.random() <= constraint_mode_probability)
        train_loss = train_step(model, optimizer, loss_fn, train_loader, constraint_mode, device)
        test_loss = validate(model, loss_fn, test_loader, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        if epoch % (n_epochs // 10) == 0:
            logging.info(f"[Epoch {epoch}] train : loss = {train_loss:.2f}, test : loss = {test_loss:.2f}")
            logging.info(f"Traduction 'she loves me' : {model.predict('she loves me.')}")


def main():
    logging.basicConfig(level=logging.INFO)

    BASE_PATH = "TPs/TP06/"
    FILE = BASE_PATH + "data/en-fra.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(FILE) as f:
        lines = f.readlines()

    lines = [lines[x] for x in torch.randperm(len(lines))]
    idxTrain = int(0.8*len(lines))

    vocEng = Vocabulary(True)
    vocFra = Vocabulary(True)
    MAX_LEN=20
    BATCH_SIZE=16

    datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
    datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

    train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

    S_EMBEDDING_DIM=40
    T_EMBEDDING_DIM=40
    HIDDEN_SIZE=30

    model = TraductionModel(
        source_vocab=vocEng,
        dest_vocab=vocFra,
        encoder_embedding_dim=S_EMBEDDING_DIM,
        decoder_embedding_dim=T_EMBEDDING_DIM, 
        hidden_size=HIDDEN_SIZE,
        padding_idx=Vocabulary.PAD,
        device=device
    ).to(device)

    LR=1e-2
    N_EPOCHS=100

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
    writer = SummaryWriter(BASE_PATH + "runs/traduction/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

    logging.info("Training")
    train(model, optimizer, device, loss_fn, train_loader, test_loader, writer, N_EPOCHS)

if __name__ == "__main__":
    main()
