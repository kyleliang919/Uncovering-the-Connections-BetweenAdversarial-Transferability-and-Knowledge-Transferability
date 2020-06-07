from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from config import configurations
from backbone.model_resnet import ResNet_18, ResNet_50, ResNet_101, ResNet_152

class HEAD(nn.Module):
    def __init__(self, in_dim = 512, out_dim = 2):
        super(HEAD, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

class MODEL(nn.Module):
    def __init__(self, backbone, head, attr):
        super(MODEL, self).__init__()
        self.backbone = backbone
        self.head = head
        self.attr = attr
    def forward(self, x):
        return self.head(self.backbone(x))

def train(args, model, device, train_loader, optimizer, epoch):
    attr = model.attr
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target[:, attr]
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim = 1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    attr = model.attr
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target[:, attr]
            output = F.log_softmax(model(data), dim = 1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--attr', type = int, default = 0)

    args = parser.parse_args()

    cfg = configurations[1]
    DATA_ROOT = cfg['DATA_ROOT'] 
    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    test_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.CenterCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    dataset_train = datasets.CelebA(DATA_ROOT ,split='train', target_type = 'attr',download=True, transform = train_transform)
    dataset_test  = datasets.CelebA(DATA_ROOT ,split='test', target_type = 'attr',download=False, transform = test_transform)
    dataset_valid = datasets.CelebA(DATA_ROOT ,split='valid', target_type = 'attr',download=False, transform = test_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = args.batch_size, pin_memory = True,
        num_workers = 8, shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size = args.batch_size, pin_memory = True,
        num_workers = 8
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size = args.batch_size, pin_memory = True,
        num_workers = 8
    )

    backbone = ResNet_18(INPUT_SIZE).to(device)
    head = HEAD().to(device)
    model = MODEL(backbone, head, args.attr).to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        path = 'checkpoints/' + str(args.attr) + '/'
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s" % path)
        torch.save(model.head.state_dict(), path + "head.pth")
        torch.save(model.backbone.state_dict(), path + "backbone.pth")
if __name__ == '__main__':
    main()
