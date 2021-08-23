import argparse
import numpy as np
import os
from os.path import join
from ResNetTSClassification.resnet_torch import ResNet, train
import torch

parser = argparse.ArgumentParser(description='Hyper-parameters to train ResNet for Time series classifaction.')

parser.add_argument('data_path', type=str, 
	default=r'/zfsauton/project/public/Mononito/MLADI_weak_supervision/',
	help='Path where data is stored.', 
	required=True)

parser.add_argument('output_directory', type=str, 
	default='../',
	help='Path where model is saved.')

parser.add_argument('n_feature_maps', type=int, 
	default=64,
	help='Number of feature maps.')

parser.add_argument('name', type=str, 
	default='ResNet',
	help='Name of model.')

parser.add_argument('device', type=str, 
	choices=['cpu', 'cuda']
	default='cuda',
	help='Device to train the model.')

parser.add_argument('nb_epochs', type=int, 
	default=20,
	help='Number of epochs')

parser.add_argument('max_lr', type=int, 
	default=1e-3,
	help='Learning rate for optimization.')

parser.add_argument('batch_size', type=int, 
	default=8,
	help='Batch size for training.')

args = parser.parse_args()

data_path = args['data_path']# r'/zfsauton/project/public/Mononito/MLADI_weak_supervision/'
X_train = torch.load(join(data_path, 'X_RR.pt')).double()
y_train = torch.load(join(data_path, 'y_RR.pt'))

print(f'Shape of X_train: {X_train.shape} and y_train: {y_train.shape}')
# print(f'Shape of X_val: {X_train.shape} and y_val: {y_train.shape}')

# For simulation, since y_train is inaccurate
y_train = torch.from_numpy(np.random.choice(2, len(X_train))).long()
print('Shape of y_train', y_train.shape)

nb_classes = len(np.unique(y_train.numpy()))

model = ResNet(output_directory=args['output_directory'], 
	input_shape=X_train.shape[1:], nb_classes=nb_classes, 
	n_feature_maps=args['n_feature_maps'], 
	name=args['name'], verbose=True, load_weights=False, 
	random_seed=13).double()

print('Model:\n', model)

trained_model = train(model, X_train, y_train, sample_weights=None,
	X_val=None, y_val=None, device=torch.device(args['device']),
	batch_size=args['batch_size'], nb_epochs=args['nb_epochs'], 
	max_lr=args['max_lr'])

print('Finished training model!')




