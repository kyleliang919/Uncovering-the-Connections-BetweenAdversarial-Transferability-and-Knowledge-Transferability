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
from subloader import CIFAR10_SubLoader, STL10_SubLoader, CIFAR100_SubLoader
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
parser.add_argument('--attack', default = 'fgsm', type = str)
parser.add_argument('--include_list', nargs='+', type=int)
parser.add_argument('--superclass', default = "aquatic_mammals", type = str)
parser.add_argument('--load_name', default = '', type=str, help='name of the checkpoint')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print('==> Preparing data..')

if args.data == 'cifar10':
    exclude_list = [i for i in range(10) if i not in args.include_list]
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
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.workers)
    validateset = testset
    validateloader =  torch.utils.data.DataLoader(validateset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.workers)
    num_classes = len(args.include_list)
    
elif args.data == 'stl10':
    exclude_list = [i for i in range(10) if i not in args.include_list]
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
    num_classes = len(args.include_list)
    validateset = testset
    validateloader =  torch.utils.data.DataLoader(validateset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.workers)
elif args.data == 'cifar100':
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
    trainset = CIFAR100_SubLoader('../data/cifar100', superclass = args.superclass, train = True,transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)

    testset = CIFAR100_SubLoader('../data/cifar100', superclass = args.superclass, train = False, transform=transform_test,download=True)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=args.workers)
    num_classes = 5
    validateset = testset
    validateloader =  torch.utils.data.DataLoader(validateset,
                                                 batch_size=100,
                                                 shuffle=True,
                                                 num_workers=args.workers)
else:
    raise Exception('no such dataset!')

mean = np.array([0.0, 0.0, 0.0]).reshape((3, 1, 1))
std = np.array([1.0, 1.0, 1.0]).reshape((3, 1, 1))

ref_model = models.__dict__['res_net18'](num_classes = num_classes)
if args.cuda:
    ref_model = ref_model.cuda()
ref_state_dict = torch.load('checkpoint/' + 'stl10_resnet18.pth')['model']
new_ref_state_dict = {}
for k,v in ref_state_dict.items():
    new_ref_state_dict[k[7:]] = v
ref_model.load_state_dict(new_ref_state_dict)
ref_model.eval()

transfer_model = models.__dict__[args.arch](num_classes = num_classes)
if args.cuda:
    transfer_model = transfer_model.cuda()
transfer_state_dict = torch.load('checkpoint/' + args.load_name)['model']
new_transfer_state_dict = {}
for k,v in transfer_state_dict.items():
    new_transfer_state_dict[k[7:]] = v
transfer_model.load_state_dict(new_transfer_state_dict)
transfer_model.eval()
print('finish loading')


inference_model = models.__dict__[args.arch](num_classes = num_classes)
if args.cuda:
    inference_model = inference_model.cuda()
inf_state_dict = torch.load('checkpoint/transfer_last_'+args.load_name)['model']
new_inf_state_dict = {}
for k,v in inf_state_dict.items():
    new_inf_state_dict[k[7:]] = v
inference_model.load_state_dict(new_inf_state_dict)
inference_model.eval()

eps = args.eps
attack_dict = {
    'fgsm':lambda model,image:attack(model, image, eps = eps, itr = 1),
    'pgd':lambda model,image:attack(model, image, eps = eps, itr = 50)
}

success = 0
base_success = 0
total = 0
alpha = 0
gamma = 0
combined = 0
attack_method = attack_dict[args.attack]
criterion = nn.CrossEntropyLoss()
alphas = []
losses = []

for batch_idx,(images,labels) in enumerate(tqdm(testloader)):
    images = images.cuda()
    labels = labels.cuda()
    
    if ref_model(images).max(1, keepdim = True)[1].item()!= labels[0].item():
        continue
    
    losses.append(criterion(inference_model(images), labels).item())    
    true_outputs = transfer_model(images)
    
    adversarials = attack_method(ref_model, images) 
    outputs = transfer_model(adversarials)
    
    base_adversarials = attack_method(transfer_model, images)
    base_outputs = transfer_model(base_adversarials)
     
    total+= images.shape[0]
    if outputs.max(1, keepdim = True)[1].item()!= labels[0].item():
        success+=1
    if base_outputs.max(1, keepdim = True)[1].item()!= labels[0].item():
        base_success+=1
    
    a = (torch.norm(true_outputs - outputs)/torch.norm(true_outputs - base_outputs)).item()
    alphas.append(a)
    alpha += a
    
    v1 = (ref_model(images) - ref_model(adversarials)).flatten().data.cpu().numpy()
    v1 = v1/np.linalg.norm(v1)
    v2 = (true_outputs - outputs).flatten().data.cpu().numpy()
    v2 = v2/np.linalg.norm(v2)
    g = np.outer(v1, v2)
    gamma += g
    
    c = a * np.outer(v1, v2)
    combined += c

gamma = np.linalg.norm(gamma / total)**2
combined = np.linalg.norm(combined / total)**2
success = success/total
base_success = base_success/total
alpha = alpha /total
print(' alpha:', alpha, 'gamma:', gamma, 'combined:', combined)
print(' alpha:', alpha, 'gamma:', gamma, 'combined:', combined)
np.save('alphas/alpha_'+args.load_name,np.array(alphas))
np.save('alphas/loss_'+args.load_name,np.array(losses))
