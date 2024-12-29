#!/usr/bin/env python
import numpy as np
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import Timing,trange

# nnue -> transformer architecture
# this uses classical cnns

class Model:
  def __init__(self):
    self.c1 = nn.Conv2d(15, 32, kernel_size=3, padding=1)
    self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(64*2*2, 128)
    self.fc2 = nn.Linear(128, 1)

  def __call__(self, x):
    x = self.c1(x).relu().max_pool2d()
    x = self.c2(x).relu().max_pool2d()
    x = x.flatten(1)
    x = self.fc1(x).relu()
    x = self.fc2(x).sigmoid()
    return x

Tensor.manual_seed(1337)

dat = np.load('processed/dataset_100K.npz')
X_train = Tensor(dat['arr_0'], dtype=dtypes.float32)
Y_train = Tensor(dat['arr_1'], dtype=dtypes.float32)

shuf = Tensor.randint(X_train.shape[0], high=X_train.shape[0])

# X_train, Y_train = X_train[shuf], Y_train[shuf]

print(f'X {X_train.shape} Y {Y_train.shape}')

def train():
  print(f'using {Device.DEFAULT}')
  BS = 512
  model = Model()
  opt = nn.optim.Adam(nn.state.get_parameters(model))

  @TinyJit
  def step(x, y):
    with Tensor.train():
      opt.zero_grad()
      loss = (model(x) - y).square().mean().backward()
      opt.step()
      return loss.realize()

  for i in (t := trange(1000)):
    GlobalCounters.reset()
    samps = Tensor.randint(BS, high=X_train.shape[0])
    loss = step(X_train[samps], Y_train[samps])
    t.set_description(f'loss {loss.item():6.2f}')

def test():
  model = Model()
  with Timing('inference in '):
    print(model(X_train[:32]).shape)

if __name__ == '__main__':
  train()
