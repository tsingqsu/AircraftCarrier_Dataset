from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms as T

from utils import data_manager
from utils.dataset_loader import ImageDataset
from utils import transforms as selfT
import models
from utils.losses import CrossEntropyLabelSmooth
from utils.utils import AverageMeter, Logger, save_checkpoint, accuracy
from utils.optimizers import init_optim

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')

# Datasets
parser.add_argument('--root', type=str, default='/home/deep/kk/data/FineGrained/Aircraft_Carrier/', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='air', choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=8, type=int, help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224, help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=224, help="width of an image (default: 224)")
parser.add_argument('--split-id', type=int, default=0, help="split index")

# Optimization options
parser.add_argument('--optim', type=str, default='sgd', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int, help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int, help="train batch size")
parser.add_argument('--test-batch', default=64, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int, help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='squeezenetv2', choices=models.get_names())

# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=2020, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=1, help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='3,2,1,0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform_train = T.Compose([
        selfT.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id)

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,    )

    testloader = DataLoader(
        ImageDataset(dataset.test, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_cls)
    print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = CrossEntropyLabelSmooth(num_classes=dataset.num_cls, use_gpu=use_gpu)

    self_parameters = [{'params': model.base.parameters(), 'lr': 0.001},
                       {'params': model.cls_head.parameters(), 'lr': 0.01}]
    optimizer = init_optim(args.optim, self_parameters, args.lr, args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, [70, 100], gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=range(4))

    if args.evaluate:
        print("Evaluate only")
        test(model, testloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        if args.stepsize > 0:
            scheduler.step()
        
        if (epoch+1) > args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, testloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best or epoch+1 == 120:
                if is_best:
                    best_rank1 = rank1
                    best_epoch = epoch + 1

                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir,
                                     'checkpoint_ep' + str(epoch+1) + '_' + str(rank1) + '.pth.tar'))

    print("==> Best Rank-1 {:.2f}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    prec_list = [AverageMeter() for _ in range(5)]

    model.train()

    end = time.time()
    for batch_idx, (img_path, imgs, pids) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        
        outputs = model(imgs)

        if isinstance(outputs, list):
            loss = 0
            prec1_list, prec5_list = [], []
            for i in range(len(outputs)):
                loss += criterion(outputs[i], pids)
                prec1, prec5 = accuracy(outputs[i], pids, topk=(1, 5))
                prec1_list.append(prec1)
                prec5_list.append(prec5)
        else:
            loss = criterion(outputs, pids)
            prec1_list, prec5_list = [], []
            prec1, prec5 = accuracy(outputs, pids, topk=(1, 5))
            prec1_list.append(prec1)
            prec5_list.append(prec5)

        losses.update(loss.item(), pids.size(0))
        for i in range(len(prec1_list)):
            prec_list[i].update(prec1_list[i].item(), pids.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % args.print_freq == 0:
            if isinstance(outputs, list):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec1 {prec1.val:.4f} ({prec1.avg:.4f})  '
                      'Prec2 {prec2.val:.4f} ({prec2.avg:.4f})  '
                      'Prec3 {prec3.val:.4f} ({prec3.avg:.4f})  '
                      'Prec4 {prec4.val:.4f} ({prec4.avg:.4f})  '.format(
                    epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time, loss=losses,
                    prec1=prec_list[0], prec2=prec_list[1],
                    prec3=prec_list[2], prec4=prec_list[3]))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec1 {prec1.val:.4f} ({prec1.avg:.4f})  '.format(
                    epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time, loss=losses,
                    prec1=prec_list[0]))


def test(model, testloader, use_gpu):
    model.eval()

    out1_all = []
    out2_all = []
    out3_all = []
    out4_all = []
    pid_all = []
    # softmax = nn.Softmax(dim=1)
    # end = time.time()
    with torch.no_grad():
        for batch_idx, (img_path, imgs, pids) in enumerate(testloader):
            if use_gpu:
                imgs, pids = imgs.cuda(), pids.cuda()

            outputs = model(imgs)

            if isinstance(outputs, list):
                out = outputs
                out1_all.append(out[0])
                out2_all.append(out[1])
                out3_all.append(out[2])
                out4_all.append(out[3])
            else:
                out = outputs
                out1_all.append(out)

            pid_all.append(pids)

    prec1, prec5 = accuracy(torch.cat(out1_all, dim=0),
                            torch.cat(pid_all, dim=0),
                            topk=(1, 5))  # this is metric on trainset
    print('Out1: Prec1 {}, Prec5 {}'.format(prec1[0], prec5[0]))

    return np.array(prec1[0].cpu().data).item()


if __name__ == '__main__':
    main()
