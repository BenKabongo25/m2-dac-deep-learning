# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))



X = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5,2, requires_grad=True, dtype=torch.float64)
b = torch.randn(2, requires_grad=True, dtype=torch.float64)

torch.autograd.gradcheck(linear, (W, b, X))