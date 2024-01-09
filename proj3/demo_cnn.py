"""
@Description :   深度学习网络训练代码
@Author      :   Xubo Luo 
@Time        :   2024/01/08 16:15:26
"""

from utils.core import *
from utils.torch_backend import *
from models.network import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default=0, type=int, help='Type of the tested network (0: 1x1 conv;  1: 3x3 conv;    2: output;  3: shortcut)')
    parser.add_argument('--is_Train', default=1, type=int, help='Training or not (0: not training; 1: training)')
    parser.add_argument('--dataset_path', default='../../cifar-10-python/', type=str, help='dataset')
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

    train_set_x = Transform(train_set, [Crop(32, 32), FlipLR(), Cutout(8,8)])
    
    # Training
    if is_Train:
        lr_schedule = PiecewiseLinear([0, 4, 20], [0, 0.4, 0])
        batch_size = 512
        summary = train(net, lr_schedule, train_set_x, test_set, batch_size=batch_size, num_workers=0)
        
        # Save the model
        save_path = save_dir + 'net_' + str(net_type) + '.pth'
        torch.save(net.state_dict(), save_path)
        print('Model saved to ' + save_path)

    # Testing
    else:
        # Load the model
        load_path = save_dir + 'net_' + str(net_type) + '.pth'
        net.load_state_dict(torch.load(load_path))
        print('Model loaded from ' + load_path)

        # Test the model
        corret = 0
        total = 0
        with torch.no_grad():
            for data in test_set:
                images, labels = data
                # image, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                corret += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * corret / total))
    