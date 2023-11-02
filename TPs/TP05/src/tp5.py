# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *


def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    loss = nn.CrossEntropyLoss()(output, target, reduce=None)
    mask = (target != padcar).float()
    return (mask * loss).sum() / mask.sum()


class RNN(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim, 
                hidden_activation=nn.Tanh(), output_activation=nn.Sigmoid()):
        """
        :param hidden_dim: dimension de l’état caché
        :param input_dim: dimension de l'entrée
        :param output_dim: dimension de la sortie
        :param hidden_activation: activation pour le calcul de l'état caché
        :param output_activation: activation pour le calcul de la sortie
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.Wi = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wd = nn.Linear(hidden_dim, output_dim)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def one_step(self, x, h):
        return self.hidden_activation(self.Wi(x) + self.Wh(h))

    def forward(self, x, h):
        length = x.size(0)
        hiddens = torch.zeros(length, x.size(1), self.hidden_dim)
        for t in range(length):
            h = self.one_step(x[t], h)
            hiddens[t] = h
        return hiddens

    def decode(self, h):
        return self.output_activation(self.Wd(h))


class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.Wf = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.Wi = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.Wc = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.Wd = nn.Linear(hidden_dim, output_dim)

    def one_step(self, x, h, c):
        ft = torch.sigmoid(self.Wf(torch.cat((h, x), dim=0)))
        it = torch.sigmoid(self.Wi(torch.cat((h, x), dim=0)))
        ct = ft * ct + it * torch.tanh(self.Wc(torch.cat((h, x), dim=0)))
        ot = torch.sigmoid(self.Wo(torch.cat((h, x), dim=0)))
        ht = ot * torch.tanh(ct)
        return ht, ct

    def forward(self, x, h, c):
        length = x.size(0)
        hiddens = torch.zeros(length, x.size(1), self.hidden_dim)
        contexts = torch.zeros(length, x.size(1), self.hidden_dim)
        for t in range(length):
            h, c = self.one_step(x[t], h, c)
            hiddens[t] = h
            contexts[t] = c
        return hiddens, contexts

    def decode(self, h):
        return torch.sigmoid(self.Wd(h))    


class GRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.Wz = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.Wt = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.Wd = nn.Linear(hidden_dim, output_dim)

    def one_step(self, x, h):
        zt = torch.sigmoid(self.Wz(torch.cat((h, x), dim=0)))
        rt = torch.sigmoid(self.Wr(torch.cat((h, x), dim=0)))
        ht = (1 - zt) * h + zt * torch.tanh(self.Wt(torch.cat((rt * h, x), dim=0)))
        return ht

    def forward(self, x, h):
        length = x.size(0)
        hiddens = torch.zeros(length, x.size(1), self.hidden_dim)
        for t in range(length):
            h = self.one_step(x[t], h)
            hiddens[t] = h
        return hiddens

    def decode(self, h):
        return torch.sigmoid(self.Wd(h))

