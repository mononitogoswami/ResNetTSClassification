import torch

class Dataset(torch.utils.data.Dataset):
  """
    Parameters
    ---------- 
    X: np.ndarray
      Training data
    
    y: np.ndarray
      Training labels
  """
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.y)

  def __getitem__(self, indices):
    return self.X[indices], y[indices]


