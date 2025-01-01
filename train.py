#!/usr/bin/env python3.11
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim

print('hello torch')
class SerializedDataset(Dataset):
  def __init__(self):
    dat = np.load('processed/dataset_100K.npz')
    self.X = dat['arr_0']
    self.Y = dat['arr_1']
    print('loaded', self.X.shape, self.Y.shape)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
    self.conv1 = nn.Conv2d(15, 32, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool2d(2, 2) 
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64*2*2, 128)
    self.fc2 = nn.Linear(128, 1)

  def forward(self, x):
    x = self.pool1(torch.relu(self.conv1(x))) 
    x = self.pool2(torch.relu(self.conv2(x))) 
    x = x.view(-1, 64*2*2) 
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return F.sigmoid(x)

def train_torch():
  device = 'cuda'
  chess_dataset = SerializedDataset()

  model = Net()
  model.cuda()

  train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=128, shuffle=True)
  optimizer = optim.Adam(model.parameters())
  lossfn = nn.MSELoss()

  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

  for epoch in range(300):
    all_loss, num_loss = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
      target = target.unsqueeze(-1)
      data, target = data.to(device), target.to(device)
      data = data.float()
      target = target.float()

      optimizer.zero_grad()
      out = model(data)

      loss = lossfn(out, target)
      loss.backward()
      optimizer.step()

      all_loss += loss.item()
      num_loss += 1

    print(f'epoch {epoch}: loss {all_loss/num_loss:.3f} lr {scheduler.get_last_lr()}')
    torch.save(model.state_dict(), 'nets/value.pth')
    scheduler.step()

if __name__ == '__main__':
  train_torch()
