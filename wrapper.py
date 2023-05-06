# -*- coding: utf-8 -*-

# import packages
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time

import matplotlib.pyplot as plt

# import ODE solvers: current options are euler and rk4
from ODESolve import *

# import NeuralODE
from NeuralODE import NeuralODE

################################################################

data_size = 2000


# generate toy dataset
true_y0 = torch.tensor([[2., 0.]]).cuda()
t = torch.linspace(0., 25., data_size).cuda()
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).cuda()


class Lambda(nn.Module):
  def forward(self, t, y):
    return torch.mm(y**3, true_A)

with torch.no_grad():
  node = NeuralODE(func=Lambda()).cuda()
  true_y = node(y0=true_y0, t=t, solver=rk4)

def plotfig(true_y, figname, pred_y=None):
  fig = plt.figure(figsize=(6, 6), facecolor='white')
  ax = fig.add_subplot(111)
  #ax.set_xlabel('x')
  #ax.set_ylabel('y')
  #ax.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'green', label='true trajectory')
  ax.scatter(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], color='blue', label='Data Samples', s=3)
  if pred_y is not None:
    ax.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'red', label='Model Output')
  ax.set_xlim(-2.5, 2.5)
  ax.set_ylim(-2.5, 2.5)
  plt.legend()
  plt.grid(True)
  plt.savefig('Results/'+figname)
  plt.clf()

plotfig(true_y, 'testimage.png')

batch_time = 10
batch_size = 16

def get_batch():
  s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
  batch_y0 = true_y[s]  # (batch_size, 1, emb)
  batch_t = t[:batch_time]  # (T)
  batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (time, batch_size, 1, emb)
  return batch_y0.cuda(), batch_t.cuda(), batch_y.cuda()


# define dynamic function
class ODEFunc(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(2, 50),
                             nn.Tanh(),
                             nn.Linear(50, 2),
                             nn.Linear(2, 50),
                             nn.Tanh(),
                             nn.Linear(50, 2))
    for m in self.net.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.constant_(m.bias, val=0)

  def forward(self, t, y):
    output = self.net(y)
    return output


## Train
niters = 500

node = NeuralODE(func=ODEFunc()).cuda()
optimizer = optim.RMSprop(node.parameters(), lr=1e-3)

start_time = time.time()

for iter in range(niters + 1):
  optimizer.zero_grad()
  batch_y0, batch_t, batch_y = get_batch()
  pred_y = node(y0=batch_y0, t=batch_t, solver=rk4)
  loss = torch.mean(torch.abs(pred_y - batch_y))
  loss.backward()
  optimizer.step()

  if iter % 50 == 0:
    with torch.no_grad():
      pred_y = node(true_y0, t, solver=rk4)
      loss = torch.mean(torch.abs(pred_y - true_y))
      print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))
      figname = 'phase_' + str(iter) + '.png'
      plotfig(true_y, figname, pred_y)

end_time = time.time() - start_time
print('process time: {} sec'.format(end_time))