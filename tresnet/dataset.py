import os
import torch
import numpy as np
from torch.utils.data import Dataset

def load_data(data_dir=data_dir):
   """Function to load data from a directory
   """
   pass

class CustomDataset(Dataset):
  """
    Parameters
    ---------- 
    : np.ndarray
      Training data
    
    y: np.ndarray
      Training labels
  """
  def __init__(self, data_dir:str, split:str='train'):

    # Function to load the data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir=data_dir)

    if split not in ['train', 'val', 'test']: raise AttributeError
    
    if split == 'train':
        self.x = torch.unsqueeze(torch.from_numpy(x_train), dim=1)
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    elif split == 'val':
        self.x = torch.unsqueeze(torch.from_numpy(x_val), dim=1)
        self.y = torch.from_numpy(y_val).type(torch.LongTensor)
    elif split == 'test':
        self.x = torch.unsqueeze(torch.from_numpy(x_test), dim=1)
        self.y = torch.from_numpy(y_test).type(torch.LongTensor)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]