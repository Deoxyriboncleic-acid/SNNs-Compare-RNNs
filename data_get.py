import torch
from torchvision import datasets
#import spikingjelly.datasets as sjds
from torch.utils.data import DataLoader
from spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask, n_mnist,dvs128_gesture

root_dir = '/home/wqy/SNNs-RNNs/N_MNIST_DATA'

train_set = n_mnist.NMNIST(root_dir, data_type='frame', duration=1000, train=True)
test_set = n_mnist.NMNIST(root_dir, data_type='frame', duration=1000, train=False) 
for i in range(5):
    x, y = train_set[i]
    print(f'x[{i}].shape=[T, C, H, W]={x.shape}')
train_data_loader = DataLoader(train_set, collate_fn=pad_sequence_collate, batch_size=5)
for x, y, x_len in train_data_loader:
    print(f'x.shape=[N, T, C, H, W]={tuple(x.shape)}')
    print(f'x_len={x_len}')
    mask = padded_sequence_mask(x_len)  # mask.shape = [T, N]
    print(f'mask=\n{mask.t().int()}')
    break