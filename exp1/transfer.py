from __future__ import print_function
import argparse
import os
from datetime import datetime
import math
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from subloader import CIFAR10_SubLoader, STL10_SubLoader
import models
from utils import progress_bar
import numpy as np

# pylint: disable=invalid-name,redefined-outer-name,global-statement

model_names = sorted(name for name in models.__dict__ if not name.startswith(
    "__") and callable(models.__dict__[name]))
best_acc = 0 # best test accuracy

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--data', default='cifar10', help = 'choice of dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res_net18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: vgg16)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--load_name', default = '0_percent.pth', type=str, help='name of the checkpoint')
parser.add_argument('--transfer_last', action = 'store_true', default=False, help='train only the last layer')
parser.add_argument('--include_list', nargs='+', type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Data loading code
print('==> Preparing data..')
exclude_list = [i for i in range(10) if i not in args.include_list]
if args.data == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trainset = CIFAR10_SubLoader('../data/cifar10', exclude_list = exclude_list, train=True, transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    testset =  CIFAR10_SubLoader('../data/cifar10', exclude_list = exclude_list, train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=args.workers)
    validateset = testset
    validateloader =  torch.utils.data.DataLoader(validateset,
                                                 batch_size=100,
                                                 shuffle=True,
                                                 num_workers=args.workers)
    num_classes = 4
elif args.data == 'stl10':
    transform_train = transforms.Compose([
          transforms.Resize(32),
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), 
    ])
    transform_test = transforms.Compose([
          transforms.Resize(32),
          transforms.ToTensor(),
          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )
    trainset = STL10_SubLoader('../data/stl10', exclude_list = exclude_list, split='train',transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)

    testset = STL10_SubLoader('../data/stl10', exclude_list = exclude_list, split='test', transform=transform_test,download=True)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=args.workers)
    num_classes = 4
    validateset = testset
    validateloader =  torch.utils.data.DataLoader(validateset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=args.workers)
else:
    raise Exception('no such dataset!')   
# Create Model
print('==> Creating model {}...'.format(args.arch))
model = models.__dict__[args.arch](num_classes = num_classes)
if args.cuda:
    model = torch.nn.DataParallel(model).cuda()

state_dict = torch.load('checkpoints/'+args.load_name)['model']

# Define loss function (criterion), optimizer and learning rate scheduler
criterion = torch.nn.CrossEntropyLoss()
if args.transfer_last:
    new_state_dict = {}
    for k,v in state_dict.items():
        # if it's the last layer re init the weight
        if 'module.linear' in k:
            try:
                stdv = 1. / math.sqrt(v.size(1))
            except:
                stdv = 1.
            if 'weight' in k:
                v.data.uniform_(-stdv, stdv)
        new_state_dict[k] = v
        state_dict = new_state_dict
    opt_param = [v for k,v in model.named_parameters() if 'module.linear' in k]     
else:
    opt_param = model.parameters()

model.load_state_dict(state_dict) 
optimizer = torch.optim.SGD(opt_param,
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0)


def train(epoch):
    ''' Trains the model on the train dataset for one entire iteration '''
    print('\nEpoch: %d' % epoch)
    cudnn.benchmark = True
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return correct/total, train_loss/(batch_idx+1)

def validate(epoch):
    ''' Validates the model's accuracy on validation dataset and saves if better
        accuracy than previously seen. '''
    cudnn.benchmark = False
    global best_acc
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validateloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validateloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        if args.transfer_last: 
            torch.save(state, './checkpoints/'+'transfer_last_'+args.load_name)
        else:
            torch.save(state, './checkpoints/'+'transfer_' + args.load_name)
        best_acc = acc
    return acc, valid_loss/(batch_idx + 1)

def test():
    ''' Final test of the best performing model on the testing dataset. '''
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.transfer_last:
        checkpoint = torch.load('./checkpoint/' + 'transfer_last_' + args.load_name)
    else:
        checkpoint = torch.load('./checkpoint/' + 'transfer_' + args.load_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print('Test best performing model from epoch {} with accuracy {:.3f}%'.format(
        checkpoint['epoch'], checkpoint['acc']))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


start_time = datetime.now()
print('Runnning training and test for {} epochs'.format(args.epochs))

# Run training for specified number of epochs
train_accs = []
train_losses = []
test_accs = []
test_losses = []
for epoch in range(0, args.epochs):
    train_acc, train_loss = train(epoch)
    scheduler.step()
    test_acc, test_loss = validate(epoch)
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    test_accs.append(test_acc)
    test_losses.append(test_loss)

if args.transfer_last:
    prefix = 'transfer_last_'
else:
    prefix = 'transfer_'

# save training stats
np.save('npy/'+prefix+'train_acc_'+args.load_name.split('.')[0] + '.npy', np.array(train_accs))
np.save('npy/'+prefix+'train_loss_'+args.load_name.split('.')[0] + '.npy', np.array(train_losses))
np.save('npy/'+prefix+'test_acc_'+args.load_name.split('.')[0] + '.npy', np.array(test_accs))
np.save('npy/'+prefix+'test_loss_'+args.load_name.split('.')[0] + '.npy', np.array(test_losses))

time_elapsed = datetime.now() - start_time
print('Training time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

# Run final test on never before seen data
test()
