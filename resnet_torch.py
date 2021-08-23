from ResNetTSClassification.data import Dataset
import numpy as np
from os.path import join
import time
import torch
from tqdm import tqdm

"""
How to call the model? 

from resnet_torch import ResNet, train

X_train = torch.from_numpy(X_train) # converts X_train from numpy to tensor
model = ResNet(output_directory='', input_shape=X_train.shape[1:], nb_classes=2)
print(model)
train(model, X_train, y_train, device=torch.device('cpu'), nb_epochs=20)

How to run a code on the GPU? 
1. nvidia-smi -- to look at the GPU usage
2. CUDA_VISIBLE_DEVICES=#id *run*.py (where #id is the GPU ID to choose, the one whice is free)

# Can also do it on a jupyter notebook.
"""

class ResNet(torch.nn.Module):
    """
    Parameters
    ---------- 
    output_directory: str
        Output directory to save the model.

    input_shape: tuple
        Shape of the input time series. (f, T) where 
        f is the number of features and T is the number
        of time steps.

    nb_classes: int
        Number of classes. 

    n_feature_maps: int
        Number of feature maps to use. 

    name: str
        Model name. Default: "ResNet"

    verbose: bool
        Controls verbosity while training and loading the models. 

    load_weights: bool
        If true load the weights of a previously saved model. 

    random_seed: int
        Random seed to instantiate the random number generator. 
    """
    def __init__(self, output_directory, input_shape, 
        nb_classes, n_feature_maps=64, name="ResNet", 
        verbose=False, load_weights=False, random_seed=13):
        super().__init__()

        self.n_feature_maps = n_feature_maps
        self.output_directory = output_directory
        self.name = name
        self.input_shape = input_shape # (num_features, num_timesteps)
        self.nb_classes = nb_classes
        self.verbose = verbose

        if load_weights: # Then just load the weights of the model.
            path = join(self.output_directory, f'best_model_{self.name}.hdf5')
            print(f"Loading model at {path}")
            self.model = torch.load(path)

            if self.verbose:
                print(self.model)

        else:
            self.model = self.build_model()

    def build_model(self):
        # BLOCK 1
        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_shape[0], out_channels=self.n_feature_maps, kernel_size=8, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps, kernel_size=5, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps, kernel_size=3, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps)
            )
        
        self.shortcut_block_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_shape[0], out_channels=self.n_feature_maps, kernel_size=1, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps)
            )
        

        # BLOCK 2
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2, kernel_size=8, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=5, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=3, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # expand channels for the sum
        self.shortcut_block_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2, kernel_size=1, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # BLOCK 3
        self.conv_block_3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=8, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=5, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=3, padding='same'),
            torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # no need to expand channels because they are equal
        self.shortcut_block_3 = torch.nn.BatchNorm1d(num_features=self.n_feature_maps * 2)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.n_feature_maps * 2, out_features=self.nb_classes),
            torch.nn.Softmax(dim=1)
            )

    def forward(self, x):
        output_block_1 = self.conv_block_1(x) + self.shortcut_block_1(x)
        output_block_1 = torch.nn.ReLU()(output_block_1)

        output_block_2 = self.conv_block_2(output_block_1) + self.shortcut_block_2(output_block_1)
        output_block_2 = torch.nn.ReLU()(output_block_2)

        output_block_3 = self.conv_block_3(output_block_2) + self.shortcut_block_3(output_block_2)
        output_block_3 = torch.nn.ReLU()(output_block_3).unsqueeze(dim=-1) # Add an extra dimension such that we have N x C x H x W

        gap_layer = torch.nn.AvgPool2d((self.input_shape[1], 1))(output_block_3).squeeze()

        preds = self.output_layer(gap_layer)

        # del output_block_1, output_block_2, output_block_3, gap_layer # release memory
        
        return preds

    def save_model(self):
        torch.save(self.model, join(self.output_directory, f'{self.name}.pth'))


def score(y_true, y_pred):
    """Computes accuracy
    """
    return np.mean(y_true == y_pred)


def train(model, X_train, y_train, sample_weights_train=None,
    X_val=None, y_val=None, sample_weights_val=None,
    device=torch.device('cpu'), batch_size=8, nb_epochs=200, 
    max_lr=1e-3):
    """
    Parameters
    ---------- 
    model: obj
        Pytorch model to train.

    X_train: torch.tensor
        (N, f, T) tensor. Training data with N samples, f features and T timesteps

    y_train: torch.tensor
        (N,) tensor with training targets from N samples

    sample_weights_train: torch.tensor
        (N, 1) tensor with sample weights from N train samples.             

    X_val: torch.tensor
        (M, f, T) tensor. Validation data with data M samples, f features and T timesteps

    y_val: torch.tensor
        (M, 1) tensor with validation targets from M samples

    sample_weights_val: torch.tensor
        (M,) tensor with sample weights from N validation samples. 

    device: torch.device
        Ab object representing the device. By default 'cpu'.    

    max_lr: float
        Maximum learning rate for the LR scheduler.

    batch_size: int
        Batch size to use for training the model. Default: 8. 

    nb_epochs: int
        Number of epochs to train the model. Default: 200. 
    """
    if sample_weights_train is None:
        lossFn = torch.nn.CrossEntropyLoss() 
    else:            
        lossFn = torch.nn.CrossEntropyLoss(reduction=None) # Weigh loss by sample weigths
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)  

    train_data = Dataset(X_train, y_train, sample_weights_train)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if X_val is not None: 
        val_data = Dataset(X_val, y_val, sample_weights_val)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    else:
        val_dataloader = None 

    val_acc, val_loss = [], []
    train_acc, train_loss = [], []
    
    model.to(device) # send model to the device
    for epoch in tqdm(range(nb_epochs)):
        loss_per_batch = []
        acc_per_batch = []
        for batch_data in train_dataloader:
            if sample_weights_train is None: 
                batch_X, batch_y = batch_data[0].to(device), batch_data[1].to(device)
            else:
                batch_X, batch_y, batch_weights = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
    
            optimizer.zero_grad()
            
            y_pred = model(batch_X)
            loss = lossFn(y_pred, batch_y)

            if sample_weights_train is not None: 
                loss = loss * batch_weights
                loss = loss.sum() / batch_weights.sum()
            
            loss.backward()
            optimizer.step()

            acc_per_batch.append(score(batch_y.cpu().detach().numpy(), torch.argmax(y_pred, dim=1).cpu().detach().numpy()))
            loss_per_batch.append(loss.cpu().detach().item())

        batch_acc = np.mean(acc_per_batch)
        batch_loss = np.mean(loss_per_batch)
        print(f'Epoch {epoch} | Accuracy: {batch_acc} | Loss: {batch_loss}')
        train_acc.append(batch_acc)
        train_loss.append(batch_loss)
            
        if val_dataloader is not None: 
            with torch.no_grad():
                loss_per_batch = []
                acc_per_batch = []
                for batch_data in val_dataloader:
                    if sample_weights_val is None: 
                        batch_X, batch_y = batch_data[0].to(device), batch_data[1].to(device)
                    else:
                        batch_X, batch_y, batch_weights = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
                    
                    y_pred = model(batch_X)
                    loss = lossFn(y_pred, batch_y)

                    if sample_weights_val is not None: 
                        loss = loss * batch_weights
                        loss = loss.sum() / batch_weights.sum()

                    acc_per_batch.append(score(batch_y.cpu().detach().numpy(), torch.argmax(y_pred, dim=1).cpu().detach().numpy()))
                    loss_per_batch.append(loss.cpu().detach().item())

                batch_acc = np.mean(acc_per_batch)
                batch_loss = np.mean(loss_per_batch)
                print(f'Epoch {epoch} | Accuracy: {batch_acc} | Loss: {batch_loss}')
                val_acc.append(batch_acc)
                val_loss.append(batch_loss)

    # TODO: Save the best model 
    model.save_model()

    return model