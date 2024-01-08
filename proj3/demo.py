"""
@Description :   深度学习网络训练代码
@Author      :   Xubo Luo 
@Time        :   2024/01/08 16:15:26
"""

from core import *
from torch_backend import *
from network import *
import argparse

DATA_DIR = './data'
dataset = cifar10(DATA_DIR)
timer = Timer()
print('Preprocessing training data')
transforms = [
    partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
    partial(transpose, source='NHWC', target='NCHW'), 
]
train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
print(f'Finished in {timer():.2} seconds')
print('Preprocessing test data')
test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
print(f'Finished in {timer():.2} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default=0, type=int, help='Type of the tested network (0: 1x1 conv;  1: 3x3 conv;    2: output;  3: shortcut)')
    parser.add_argument('--is_Train', default=1, type=int, help='Training or not (0: not training; 1: training)')
    parser.add_argument('--dataset_path', default='../../cifar10-python/', type=str, help='dataset')
    parser.add_argument('--save_dir', default='./caches/', type=str, help='save dir')
    args = parser.parse_args()


    net_type = args.net
    is_Train = args.is_Train
    dataset_path = args.dataset_path
    save_dir = args.save_dir

    if net_type == 0:
        net = conv1x1()
    elif net_type == 1:
        net = conv3x3()
    elif net_type == 2:
        net = conv_output()
    elif net_type == 3:
        net = conv_shortcut()
    
    # Prepare the dataset
    DATA_DIR = dataset_path
    dataset = cifar10(DATA_DIR)
    timer = Timer()
    print('Preprocessing training data')
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
    print(f'Finished in {timer():.2} seconds')

    # Training
    if is_Train:
        lr_schedule = PiecewiseLinear([0, 4, 20], [0, 0.4, 0])
        batch_size = 512

        

    