# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        # Renvoyer la valeur de la fonction
        return (yhat - y) ** 2

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        # Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        return grad_output * 2 * (yhat - y), grad_output * -2 * (yhat - y)

# Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE
class Linear(Function):
    """Implementation de la fonction Linear"""
    @staticmethod
    def forward(ctx, W, b, Z):
        ctx.save_for_backward(W, b, Z)
        return Z @ W + b

    @staticmethod
    def backward(ctx, grad_outputs):
        W, _, Z = ctx.saved_tensors
        return Z.T @ grad_outputs, grad_outputs.sum(0), grad_outputs @ W.T


## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

