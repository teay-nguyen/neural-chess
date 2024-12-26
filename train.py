#!/usr/bin/env python3.11
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim

if os.environ.get('TORCH') == '1':
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
      self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
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
    if device == 'cuda':
      model.cuda()

    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle=True)
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.MSELoss()

    for epoch in range(100):
      all_loss, num_loss = 0, 0
      for batch_idx, (data, target) in enumerate(train_loader):
        target = target.unsqueeze(-1)
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()

        optimizer.zero_grad()
        out = model(data)

        loss = loss_func(out, target)
        loss.backward()
        optimizer.step()

        all_loss += loss.item()
        num_loss += 1

      print(f'epoch {epoch}: loss {all_loss/num_loss:.3f}')
      torch.save(model.state_dict(), 'nets/value.pth')
elif os.environ.get('TINYGRAD') == '1':
  print('hello tinygrad')
else:
  raise Exception('choose torch or tinygrad')

if __name__ == '__main__':
  pass
