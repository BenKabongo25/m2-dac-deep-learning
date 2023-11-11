# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import datetime
import torch
import torchmetrics
import logging

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List


logging.basicConfig(level=logging.INFO)


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset(Dataset):
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []
        for s in data:
            self.sentences.append(
                (
                    [words.get(token["form"], adding) for token in s], 
                    [tags.get(token["upostag"], adding) for token in s]
                )
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))



def extract_form_upos(file_path):
    sentences = []
    sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('# sent_id'):
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = []
            elif not line or line.startswith('#'):
                continue
            else:
                parts = line.split('\t')
                form = parts[1]
                upos = parts[3]
                sentence.append({"form": form, "upostag": upos})

    if len(sentence) > 0:
        sentences.append(sentence)
    
    return sentences


class TaggingModel(nn.Module):
    def __init__(self, n_words, n_tags, embedding_dim, hidden_size, padding_idx=0, num_layers=1, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(n_words, embedding_dim, padding_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, proj_size=n_tags)

    def forward(self, X):
        embedded = self.embedding(X)
        output, _ = self.rnn(embedded)
        return output


def train_step(model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss = 0
    for (X, y) in dataloader:
        y_pred = model(X)
        y_pred, y = y_pred.view(y.numel(), -1), y.flatten()
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss


def validate(model, loss_fn, accuracy_fn, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (X, y) in dataloader:
            y_pred = model(X)
            y_pred, y = y_pred.view(y.numel(), -1), y.flatten()
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            accuracy_fn(y_pred.argmax(1), y)
    accuracy = accuracy_fn.compute()
    accuracy_fn.reset()
    return total_loss, accuracy


def train(
        model, 
        optimizer, 
        loss_fn, 
        accuracy_fn, 
        train_loader, 
        dev_loader, 
        test_loader,
        writer,
        n_epochs=20,
    ):
    for epoch in tqdm(range(1, n_epochs+1), "Training"):
        train_loss = train_step(model, optimizer, loss_fn, train_loader)
        dev_loss, dev_acc = validate(model, loss_fn, accuracy_fn, dev_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/dev", dev_loss, epoch)
        writer.add_scalar("Accuracy/dev", dev_acc, epoch)
        if epoch % (n_epochs // 10) == 0:
            logging.info(f"[Epoch {epoch}] train : loss = {train_loss:.2f}, dev : loss = {dev_loss:.2f}, acc = {dev_acc:.2f}")
    test_loss, test_acc = validate(model, loss_fn, accuracy_fn, test_loader)
    logging.info(f"Test : loss = {test_loss:.2f}, acc = {test_acc:.2f}")


def main():
    logging.info("Loading datasets...")

    #ds = prepare_dataset('org.universaldependencies.french.gsd')
    BASE_PATH = "TPs/TP06/"
    train_data = extract_form_upos(BASE_PATH + "data/fr_gsd-ud-train.conllu")
    dev_data = extract_form_upos(BASE_PATH + "data/fr_gsd-ud-dev.conllu")
    test_data = extract_form_upos(BASE_PATH + "data/fr_gsd-ud-test.conllu")


    words = Vocabulary(True)
    tags = Vocabulary(False)
    train_dataset = TaggingDataset(train_data, words, tags, True)
    dev_dataset = TaggingDataset(dev_data, words, tags, True)
    test_dataset = TaggingDataset(test_data, words, tags, False)

    N_WORDS = len(words)
    N_TAGS = len(tags)

    logging.info("Vocabulary words size: %d", N_WORDS)
    logging.info("Vocabulary tags size: %d", N_TAGS)

    BATCH_SIZE=100

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)


    EMBEDDING_DIM=512
    HIDDEN_SIZE=256
    NUM_LAYERS=2
    DROPOUT=0.25

    model = TaggingModel(
        n_words=len(words), 
        n_tags=len(tags), 
        embedding_dim=EMBEDDING_DIM, 
        hidden_size=HIDDEN_SIZE, 
        padding_idx=Vocabulary.PAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    LR=1e-2
    N_EPOCHS=100

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
    accuracy_fn = torchmetrics.classification.Accuracy(task="multiclass", num_classes=N_TAGS)
    writer = SummaryWriter(BASE_PATH + "runs/tagging/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

    logging.info("Training")
    train(model, optimizer, loss_fn, accuracy_fn, train_loader, dev_loader, test_loader, writer, N_EPOCHS)


if __name__ == "__main__":
    main()
