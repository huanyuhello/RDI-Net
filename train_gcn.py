import argparse
import os
import shutil
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from load_model import get_model
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.loss import InfoNCELoss
from utils.utils import cp_projects
from models.base import UniformSample


def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def parse():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--model', type=str, default='R110_C10', help='resnets')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=160, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR',
                        help='Initial learning rate.  '
                             'Will be scaled by '
                             '<global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  '
                             'A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--clip', default=0.05, type=int)

    parser.add_argument('--log_path', type=str, default='', help='')
    parser.add_argument('--note', type=str, default='', help='note of the experiment')
    parser.add_argument('--uniform_sample', action='store_true')
    parser.add_argument('--freeze_gate', action='store_true')
    parser.add_argument('--freeze_net', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None, help='')
    parser.add_argument('--weight', type=float, default=0.1, help='')
    parser.add_argument('--option', type=float, default=3)
    args = parser.parse_args()
    return args


def main():
    global best_prec, best_cost, args

    args = parse()
    # print("opt_level = {}".format(args.opt_level))
    # print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    # print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec = 0.0
    best_cost = 0.0

    args.distributed = False

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_%s' % (args.model)

    save_path = os.path.join(args.log_path, now + hyper_param_str + '_' + args.note)

    logger = SummaryWriter(save_path)

    config_txt = os.path.join(save_path, 'args')

    with open(config_txt, 'w') as fp:
        fp.write(str(args))
        cp_projects(save_path)

    # create model
    print("=> creating model '{}'".format(args.model))
    model, nblocks = get_model(model='R50_ImgNet', freeze_gate=args.freeze_gate, uniform_sample=args.uniform_sample,
                               freeze_net=args.freeze_net, resume_path=args.resume_path)

    sampler = UniformSample(nblocks, options=args.option)

    model = torch.nn.DataParallel(model).cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size) / 256.

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cre_crite = nn.CrossEntropyLoss().cuda()

    if args.resume:

        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                source = checkpoint['state_dict']
                model.load_state_dict(source)

                # optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # Data loading code
    traindir = os.path.join('/home/huanyu/dataset/ImageNet/', 'train')
    valdir = os.path.join('/home/huanyu/dataset/ImageNet/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, cre_crite, None, optimizer, epoch, logger, sampler, args.uniform_sample)

        # evaluate on validation set
        prec1, flops = validate(val_loader, model, epoch, logger, sampler, args.uniform_sample)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec
        if is_best:
            best_cost = flops
            best_prec = prec1
        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path)
        logger.add_scalar('best/top1', best_prec, global_step=epoch)
        logger.add_scalar('best/cost', best_cost, global_step=epoch)


def train(train_loader, model, criterion, nce_crite, optimizer, epoch, logger, sampler=None, uniform_sample=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        input, target = input.cuda(), target.cuda()
        output, prob = model(input)
        loss_ce = criterion(output, target)
        loss = loss_ce

        # compute gradient and do SGD step
        optimizer.zero_grad()

        optimizer.step()

        if i % args.print_freq == 0:
            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = loss.data

            losses.update(reduced_loss, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            # print('Epoch:[{0}][{1}/{2}] Loss:{loss.val:.3f}({loss.avg:.3f}) '
            #       'Top@1:{top1.val:.2f}%({top1.avg:.2f}%) '
            #       'Top@5:{top5.val:.2f}%({top5.avg:.2f}%)'.format(epoch, i, len(train_loader),
            #                                                       loss=losses, top1=top1, top5=top5))

            logger.add_scalar('train/top1', top1.avg, global_step=len(train_loader) * epoch + i)
            logger.add_scalar('train/top5', top5.avg, global_step=len(train_loader) * epoch + i)
            logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'],
                              global_step=len(train_loader) * epoch + i)
            logger.add_scalar('train/loss', losses.avg, global_step=len(train_loader) * epoch + i)

        # input, target = prefetcher.next()


def validate(val_loader, model, epoch, logger, sampler=None, uniform_sample=False):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cost = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()

    # prefetcher = data_prefetcher(val_loader)
    # input, target = prefetcher.next()
    # i = 0
    # while input is not None:
    for i, (input, target) in enumerate(tqdm(val_loader)):
        input, target = input.cuda(), target.cuda()
        # compute output
        with torch.no_grad():
            if uniform_sample:
                route = sampler.sample(input.size(0)).cuda()
                output, flops = model(input, route)
            else:
                output, flops = model(input)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        # cost.update(to_python_float(flops), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Test:[{0}/{1}] '
        #           'Top@1:{top1.val:.2f}({top1.avg:.2f}) '
        #           'Top@5:{top5.val:.2f}({top5.avg:.2f})'.format(
        #         i, len(val_loader), top1=top1, top5=top5))

        # input, target = prefetcher.next()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    logger.add_scalar('valid/top1', top1.avg, global_step=epoch)
    logger.add_scalar('valid/top5', top5.avg, global_step=epoch)

    return top1.avg, cost.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def save_model(state, is_best, save_path):
    if is_best:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        model_path = os.path.join(save_path, 'ckpt.pth')
        torch.save(state, model_path)


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


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    # if epoch >= 80:
    #     factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    # if epoch < 5:
    #     lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
