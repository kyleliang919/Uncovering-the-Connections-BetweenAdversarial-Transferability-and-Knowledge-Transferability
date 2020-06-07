import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import configurations
from backbone.model_resnet import ResNet_18, ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import pickle
import argparse
from attack import attack, attack_feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attack and transfer')
    parser.add_argument('--eps', default = 0.03, type = float)
    parser.add_argument('--attack', default = 'fgsm', type = str)
    args = parser.parse_args()
    eps = args.eps
    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]
    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = 1
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
    # Construct Dataset for CelebA
    dataset_train = datasets.CelebA(DATA_ROOT ,split='all', target_type = 'identity',download=True, transform = train_transform)

    # create a weighted random sampler to process imbalanced data
    id_list = dataset_train.identity.numpy()

    weights = make_weights_for_balanced_classes([(None,each[0] - 1) for each in id_list], np.unique(id_list).shape[0])
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, pin_memory = PIN_MEMORY,
        shuffle=False,
        num_workers = NUM_WORKERS, 
        drop_last = DROP_LAST
    )


    NUM_CLASS = np.unique(id_list).shape[0]
    print("Number of Training Classes: {}".format(NUM_CLASS))
    
    # load reference model
    reference_backbone = ResNet_18(INPUT_SIZE).to(device)
    reference_head = ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = [1]).to(device)

    LOSS_DICT = {'Focal': FocalLoss(), 
                 'Softmax': nn.CrossEntropyLoss()}
    reference_loss = LOSS_DICT[LOSS_NAME]
    attack_dict = {
        'fgsm':lambda model,image:attack(model, image, eps = eps, itr = 1),
        'pgd':lambda model,image:attack(model, image, eps = eps, itr = 10),
        'fgsm-l1':lambda model,image: attack_feature(model, image, eps = eps, itr = 1, loss_type = 'l1'),
        'fgsm-l2':lambda model,image:attack_feature(model, image, eps = eps, itr = 1,loss_type = 'l2'),
        'pgd-l1':lambda model, image: attack_feature(model, image, eps = eps, itr = 10, loss_type = 'l1'),
        'pgd-l2':lambda model, image: attack_feature(model, image, eps = eps, itr = 10, loss_type = 'l2')
    }
    
    attack_method = attack_dict[args.attack]
    #criterion = loss_dict[args.attack]
    criterion = reference_loss
    backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(reference_backbone) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    _, head_paras_wo_bn = separate_resnet_bn_paras(reference_head)

    reference_backbone.load_state_dict(torch.load('checkpoints/reference_resnet18/Backbone_ResNet_18.pth'))
    reference_head.load_state_dict(torch.load('checkpoints/reference_resnet18/Head_ArcFace.pth'))
    reference_backbone.eval()
    reference_head.eval()

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

    attributes = np.arange(40)
    results = {}
    for attr in attributes:
        backbone = ResNet_18(INPUT_SIZE).to(device)
        head = HEAD().to(device)
        model = MODEL(backbone, head, attr)
        model.eval()
        backbone.load_state_dict(torch.load('checkpoints/' + str(attr) + "/backbone.pth"))
        head.load_state_dict(torch.load('checkpoints/' + str(attr)+ "/head.pth"))
        losses = []
        count = 0
        for img, labels in tqdm(iter(train_loader)):
            img = img.to(device)
            labels = (labels - 1).to(device).long()
            adv_img = attack_method(model,img)
            adv_features = reference_backbone(adv_img)
            adv_outputs = reference_head(adv_features, labels)
            losses.append(criterion(adv_outputs, labels).item())
            count+=1
            if count == 1000: ## change this number to run for more images
                break
        results[str(attr)] = losses
    with open('./pkl/'+ args.attack + '_'+ str(args.eps) +'_results.pkl', 'wb') as file:
        pickle.dump(results,file)
