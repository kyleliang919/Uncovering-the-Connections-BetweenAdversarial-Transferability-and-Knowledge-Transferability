import argparse
import os
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from subloader import CIFAR10_SubLoader, STL10_SubLoader
import models
from utils import progress_bar
import numpy as np
from tqdm import tqdm
from attack import attack, attack_feature

# pylint: disable=invalid-name,redefined-outer-name,global-statement

model_names = sorted(name for name in models.__dict__ if not name.startswith(
    "__") and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='attack and transfer')
parser.add_argument('-d', '--data', default='cifar10', help = 'choice of dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='res_net18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: vgg16)')

parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')

parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--eps', default = 0.06, type = float)
parser.add_argument('--attack', default = 'pgd', type = str)
parser.add_argument('--include_list', nargs='+', type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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
                                             shuffle=False,
                                             num_workers=args.workers)
    validateset = testset
    validateloader =  torch.utils.data.DataLoader(validateset,
                                                 batch_size=100,
                                                 shuffle=False,
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
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.workers)
    num_classes = 4
    validateset = testset
    validateloader =  torch.utils.data.DataLoader(validateset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.workers)
else:
    raise Exception('no such dataset!')

mean = np.array([0.0, 0.0, 0.0]).reshape((3, 1, 1))
std = np.array([1.0, 1.0, 1.0]).reshape((3, 1, 1))
ref_model = models.__dict__[args.arch](num_classes = num_classes)
if args.cuda:
    ref_model = ref_model.cuda()

ref_state_dict = torch.load('checkpoints/' + 'target_model.pth')['model']
new_ref_state_dict = {}
for k,v in ref_state_dict.items():
    new_ref_state_dict[k[7:]] = v
ref_model.load_state_dict(new_ref_state_dict)
if args.cuda:
    ref_model = ref_model.cuda()
ref_model.eval()

transfer_models = []
for i in range(5):
    model = models.__dict__[args.arch](num_classes = num_classes)
    if args.cuda:
        model = model.cuda()
    state_dict = torch.load('checkpoints/' + str(i * 25) + '_percent.pth')['model']
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    transfer_models.append(model)
print('finish loading')

eps = args.eps
attack_dict = {
    'fgsm':lambda model,image:attack(model, image, eps = eps, itr = 1),
    'pgd':lambda model,image:attack(model, image, eps = eps, itr = 10),
    'fgsm-l1':lambda model,image: attack_feature(model, image, eps = eps, itr = 1, loss_type = 'l1'),
    'fgsm-l2':lambda model,image:attack_feature(model, image, eps = eps, itr = 1,loss_type = 'l2'),
    'pgd-l1':lambda model, image: attack_feature(model, image, eps = eps, itr = 10, loss_type = 'l1'),
    'pgd-l2':lambda model, image: attack_feature(model, image, eps = eps, itr = 10, loss_type = 'l2')
}

loss = [0 for _ in range(5)]
acc = [0 for _ in range(5)]
total = 0
attack_method = attack_dict[args.attack]
criterion = nn.CrossEntropyLoss()
per_img_results = {}
ref_model.eval()
for batch_idx,(images,labels) in enumerate(tqdm(testloader)):
    images = images.cuda()
    labels = labels.cuda()
    if ref_model(images).max(1, keepdim = True)[1].item()!= labels[0].item():
        continue
    total+= images.shape[0]
    for i in range(5):
        adversarials = attack_method(transfer_models[i], images)
        outputs = ref_model.forward(adversarials)
        l = criterion(outputs, labels).item()
        loss[i]+= l
        try:
            per_img_results[str(i)].append(l)
        except:
            per_img_results[str(i)] = [l]

loss = np.array(loss)/(batch_idx + 1)
np.save('npy/'+ args.attack + '_' + str(eps) + '_attack_transfer_loss.npy', loss)
import pickle
with open('pkl/'+ args.attack + '_' + str(eps) + '_per_img_results.pkl', 'wb') as file:
    pickle.dump(per_img_results, file)
