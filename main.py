import numpy as np
from tresnet.dataset import CustomDataset
from tresnet.resnet import ResNet, train, test, accuracy

def main():
    clf = ResNet(input_shape=(1, 256), n_feature_maps=4, n_classes=2)
    clf.build_model()

    train_data = CustomDataset(features_dir='', split='train')
    val_data = CustomDataset(features_dir='', split='val')
    test_data = CustomDataset(features_dir='', split='test')
        
    train(model=clf, 
          train_data=train_data,
          val_data=val_data,
          batch_size=64,
          n_epochs=100,
          max_learning_rate=1e-3,
          device='cuda:1',
          save_dir='')
    
    y_true, y_preds = test(model=clf, 
                           test_data=test_data,
                           batch_size=64,
                           device='cuda:1')
    
    print(f'Testing Accuracy: {accuracy(y_true, np.argmax(y_preds, axis=1))}')

if __name__ == '__main__':
    main()