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
  def __init__(self, X, y, weigths):
    self.X = X
    self.y = y
    self.weigths = weigths

  def __len__(self):
    return len(self.y)

  def __getitem__(self, indices):
    if self.weigths is None: 
      return self.X[indices], self.y[indices], None
    else:   
      return self.X[indices], self.y[indices], self.weigths[indices]


