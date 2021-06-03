import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import arguments, utils

def Cosine(g1, g2):
    return torch.abs(F.cosine_similarity(g1, g2)).mean() + (0.2 * torch.sum(g1**2+g2**2,1)).mean()


class Transfer_Trainer():
    def __init__(self, models, optimizers, schedulers,
                 source_trainloader, source_testloader, target_trainloader, surrogate,
                 writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.optimizer = optimizers
        self.scheduler = schedulers
        self.trainloader = source_trainloader
        self.testloader = source_testloader
        self.targetloader = target_trainloader

        self.surrogate = surrogate

        self.coeff = kwargs['transfer_coeff']
        self.writer = writer
        self.save_root = save_root
        self.depth = kwargs['depth']
        self.criterion = nn.CrossEntropyLoss()

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs + 1)), total=self.epochs, desc='Epoch',
                        leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        loader = utils.DistillationLoader(self.trainloader, self.targetloader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            if (epoch % 10 == 0):
                self.save(epoch)

    def train(self, epoch):


        self.models.train()

        losses = 0
        batch_iter = self.get_batch_iterator()
        for batch_idx, (input, target, si, sl) in enumerate(batch_iter):
            inputs, targets = input.cuda(), target.cuda()
            si, sl = si.cuda(), sl.cuda()

            si.requires_grad = True

            outputs = self.models(inputs)
            benign_loss = self.criterion(outputs, targets)

            s_output = self.models(si)
            loss_s = self.criterion(s_output, sl)
            grad_s = autograd.grad(loss_s, si, create_graph=True)[0]
            grad_s = grad_s.flatten(start_dim=1)

            t_output = self.models(si)
            loss_t = self.criterion(t_output, sl)
            grad_t = autograd.grad(loss_t, si, create_graph=True)[0]
            grad_t = grad_t.flatten(start_dim=1)

            grad_loss = Cosine(grad_s, grad_t).item()


            loss = benign_loss + self.coeff * grad_loss

            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print_message = 'Epoch [%3d] | ' % epoch
        print_message += '{loss:.4f}  '.format(loss=losses / (batch_idx + 1))
        tqdm.write(print_message)

        self.scheduler.step()


        self.writer.add_scalar('train/loss', losses / len(self.trainloader), epoch)

    def test(self, epoch):
        self.models.eval()


        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.models(inputs)
                loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/ensemble_loss', loss / len(self.testloader), epoch)
        self.writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

        print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
            loss=loss / len(self.testloader), acc=correct / total)
        tqdm.write(print_message)

    def save(self, epoch):
        state_dict = self.models.state_dict()
        torch.save(state_dict, os.path.join(self.save_root, 'epoch_%d.pth' % epoch))


def get_args():
    parser = argparse.ArgumentParser(description='Transfer Training', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.transfer_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    save_root = os.path.join('checkpoints', 'transfer', 'seed_{:d}'.format(args.seed),
                             '{:s}{:d}'.format(args.arch, args.depth))

    save_root += "%.2f" % (args.transfer_coeff)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    torch.manual_seed(args.seed)

    # initialize models
    models = utils.get_models(args, train=True, as_ensemble=False, model_file="/sync_transfer/CIFAR/epoch_200.pth",
                              dataset="CIFAR-10")

    # get data loaders
    source_trainloader, source_testloader = utils.get_loaders(args, dataset="CIFAR-10")
    target_trainloader, target_testloader = utils.get_loaders(args, dataset="STL-10")
    # get optimizers and schedulers
    optimizers = utils.get_optimizers(args, models)
    schedulers = utils.get_schedulers(args, optimizers)


    surrogate = utils.get_models(args, train=False, as_ensemble=False, model_file="/sync_transfer/STL/epoch_200.pth",
                                 dataset="STL-10")
    trainer = Transfer_Trainer(models, optimizers, schedulers,
                               source_trainloader, source_testloader, target_trainloader, surrogate, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
