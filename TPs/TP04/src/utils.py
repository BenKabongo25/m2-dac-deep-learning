import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data, length=20, step=1, stations_max=None):
        """
        :param data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
        :param length : longueur des séquences d'exemple
        :param step: pas des prédictions
        :param stations_max : normalisation à appliquer
        """
        self.data, self.length, self.step = data, length, step
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-step,:,:], x[d,t+step:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return (
            self.data[day, timeslot : (timeslot+self.length-self.step)],
            self.data[day, (timeslot+self.step) : (timeslot+self.length)]
        )

