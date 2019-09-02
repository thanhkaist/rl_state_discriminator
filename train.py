import argparse
import os
import shutil
import time
import random
import numpy as np
from numpy.random import choice


import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from model.model import *
from torch.utils import data
import torchvision.transforms as transforms
from ray import tune

parser = argparse.ArgumentParser(description='PyTorch State Discriminator training scripts')
parser.add_argument('arch', metavar='ARCH', default='target', type=str, choices=['target', 'neighbor'],
                    help='Model architecture:target/neighbor')
parser.add_argument('--ngpu', type=int, default=2, metavar='G', help='Number of GPUS to use')
parser.add_argument('--data', '-d', type=str, default='pair_goal.npz', help='Dataset path')
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default 0.9)')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay(default 1e-4)')
parser.add_argument('--echo_freq', '-e', default=2, type=int, help='echo frequency (df:10)')
parser.add_argument('--eval', dest='evaluate', action='store_true', help='Evaluation only')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='Path to latest checkpoint (df: None)')
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate ")
parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DataSet(data.Dataset):
    '''Dataset for load image from npz'''

    def __init__(self, filepath, n, begin=0, transform=None, target_transform=None, arch='target'):
        # initialization
        self.len = n
        self.transform = transform
        self.target_transform = target_transform
        self.begin = begin
        self.arch = arch
        img1s, img2s, labels = self.read_from_npz(filepath, n)
        self.labels = labels
        self.img1s = img1s
        self.img2s = img2s

    def read_from_npz(self, filepath, n):
        data = np.load(filepath)
        if self.arch == 'target':
            imgs = data['pair_goal']
            labels = data['labels']
        elif self.arch == 'neighbor':
            imgs = data['pair_obs_arr']
            labels = data['labels']
        assert self.begin + self.len <= imgs.shape[0]
        img1s = imgs[self.begin:self.begin + n, 0]
        img2s = imgs[self.begin:self.begin + n, 1]
        labels = labels[self.begin:self.begin + n]
        return img1s, img2s, labels

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # generates one sample of data
        img1 = self.img1s[index]
        img2 = self.img2s[index]
        labels = self.labels[index]
        labels = labels.astype(np.int)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.target_transform is not None:
            pass
            # labels = self.target_transform(labels)

        return img1, img2, labels


def test_dataset(filepath, n):
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    train_set = DataSet(filepath, n)
    training_loader = data.DataLoader(train_set, **params)

    for i, (img1, img2, labels) in enumerate(training_loader):
        print(i, img1.shape, img2.shape, labels.shape)


def main():
    args = parser.parse_args()
    print("args", args)
    # test data set
    # test_dataset(args.data,500)
    best_prec1 = 0

    # init ramdom seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create model 
    if args.arch == 'target':
        model = target_model()
    elif args.arch == 'neighbor':
        model = neighbor_model()
    else:
        raise Exception('The the model args.arch is not define')

    # define loss function 
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # use multiple gpu 
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()

    print("model")
    print(model)

    # get number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # Optionally resume from a checkpoint 
    if args.resume:
        if os.path.isfile(args.resume):
            print("=>Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # data loading code 
    param = {
        'batch_size': 64,
        'num_workers': 6,
        'shuffle': True,
        'pin_memory': True
    }
    train_set = DataSet(args.data, 800,begin = 0,
                        transform=transforms.Compose([transforms.ToTensor()]),
                        target_transform=transforms.Compose([transforms.ToTensor()]),
                        arch=args.arch)
    train_loader = data.DataLoader(train_set, **param)

    param = {
        'batch_size': 64,
        'num_workers': 6,
        'shuffle': False,
        'pin_memory': True
    }

    test_set = DataSet(args.data, 200, begin=800,
                       transform=transforms.Compose([transforms.ToTensor()]),
                       target_transform=transforms.Compose([transforms.ToTensor()]),
                       arch=args.arch)
    test_loader = data.DataLoader(test_set, **param)

    if args.evaluate:
        # load best model

        # evaluate
        validate(test_loader, model, criterion, 0, args.echo_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.echo_freq)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion, epoch, args.echo_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch + 1,
                         'arch': args.arch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'optimizer': optimizer.state_dict(),
                         }, is_best, args.prefix)


def save_checkpoint(state, is_best, prefix):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)


def adjust_learning_rate(starting_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 2 epochs"""
    lr = starting_lr * (0.5 ** (epoch // 30))
    print(bcolors.WARNING + 'Leaning rate {}'.format(lr) + bcolors.ENDC)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(logit, labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = logit.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, echo_freq=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # turn model to training mode 
    model.train()
    end = time.time()

    for i, (img1s, img2s, labels) in enumerate(train_loader):
        # measure data loading time 
        data_time.update(time.time() - end)

        labels = labels.cuda(async=True)
        img1s_var = torch.autograd.Variable(img1s)
        img2s_var = torch.autograd.Variable(img2s)
        labels_var = torch.autograd.Variable(labels)

        # compute output
        output = model(img1s_var, img2s_var)
        loss = criterion(output, labels_var)

        # measure accuracy and record loss 
        prec1 = accuracy(output.data, labels)
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1[0].item(), labels.size(0))

        # compute gradient and do SGD step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()

        if i % echo_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.1f} ({top1.avg:.1f})'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time, data_time=data_time,
                                                                  loss=losses, top1=top1)

                  )


def validate(test_loader, model, criterion, epoch, echo_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # turn model to training mode 
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (img1s, img2s, labels) in enumerate(test_loader):
            # measure data loading time
            labels = labels.cuda(async=True)
            img1s_var = torch.autograd.Variable(img1s)
            img2s_var = torch.autograd.Variable(img2s)
            labels_var = torch.autograd.Variable(labels)

            # compute output
            output = model(img1s_var, img2s_var)
            loss = criterion(output, labels_var)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, labels)
            losses.update(loss.item(), labels.size(0))
            top1.update(prec1[0].item(), labels.size(0))

            # compute gradient and do SGD step

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % echo_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.1f} ({top1.avg:.1f})'.format(epoch, i, len(test_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses, top1=top1)

                      )
        print(
            bcolors.OKGREEN + '***Epoch [{epoch}] Prec@1 {top1.avg:.3f}'.format(epoch=epoch, top1=top1) + bcolors.ENDC)
    return top1.avg


if __name__ == '__main__':
    main()
