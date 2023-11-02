# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
X = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
W = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):

    linear_context = Context()
    mse_context = Context()

    ## Calcul du forward (loss)
    yhat = Linear.forward(linear_context, W, b, X)
    loss = MSE.forward(mse_context, yhat, y).mean().item()

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    #print(f"Itérations {n_iter}: loss {loss}")

    ## Calcul du backward (grad_w, grad_b)
    dyhat, dy = MSE.backward(mse_context, torch.randn(50, 3))
    dW, db, _ = Linear.backward(linear_context, dyhat)

    ## Mise à jour des paramètres du modèle
    W -= epsilon * dW
    b -= epsilon * db


