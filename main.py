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
from utils.loss import InfoNCELoss
from utils.utils import cp_projects
from models.base import UniformSample

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


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
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--model', type=str, default='R110_C10', help='resnets')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
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
    parser.add_argument('--lrfact', default=1, type=float,
                        help='learning rate factor')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--clip', default=0.05, type=int)
    parser.add_argument('--lrdecay', default=30, type=int,
                        help='epochs to decay lr')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--log_path', type=str, default='', help='')
    parser.add_argument('--note', type=str, default='', help='note of the experiment')
    parser.add_argument('--uniform_sample', action='store_true')
    parser.add_argument('--freeze_gate', action='store_true')
    parser.add_argument('--freeze_net', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None, help='')
    parser.add_argument('--weight', type=float, default=0.1, help='')
    parser.add_argument('--option', type=float, default=3)
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    global best_prec, best_cost, args

    args = parse()
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec = 0.0
    best_cost = 0.0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    if args.local_rank == 0:
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        hyper_param_str = '_%s' % (args.model)

        save_path = os.path.join(args.log_path, now + hyper_param_str + '_' + args.note)

        logger = SummaryWriter(save_path)

        config_txt = os.path.join(save_path, 'args')

        with open(config_txt, 'w') as fp:
            fp.write(str(args))
            cp_projects(save_path)
    else:
        logger = None
        save_path = None

    # create model
    print("=> creating model '{}'".format(args.model))
    model, nblocks = get_model(model='R50_ImgNet', freeze_gate=args.freeze_gate, uniform_sample=args.uniform_sample,
                               freeze_net=args.freeze_net, resume_path=args.resume_path)

    sampler = UniformSample(nblocks, options=args.option)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to(memory_format=memory_format)

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.SGD([{'params': [param for name, param in model.named_parameters() if 'router' in name],
                            'lr': args.lrfact * args.lr, 'weight_decay': args.weight_decay},
                           {'params': [param for name, param in model.named_parameters() if 'router' not in name],
                            'lr': args.lr, 'weight_decay': args.weight_decay}
                           ], momentum=args.momentum)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    else:
        model = torch.nn.DataParallel(model).cuda()

    cre_crite = nn.CrossEntropyLoss().cuda()
    nce_crite = InfoNCELoss(num_block=nblocks).cuda()

    if args.resume:

        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                source = checkpoint['state_dict']
                target = model.state_dict()
                for k, v in source.items():
                    if 'router' not in k:
                        target[k] = v
                model.load_state_dict(target)

                # optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # Data loading code
    traindir = os.path.join('/home/huanyu/dataset/ImageNet/', 'train')
    valdir = os.path.join('/home/huanyu/dataset/ImageNet/', 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, cre_crite, nce_crite, optimizer, epoch, logger, sampler,
              args.uniform_sample, args.weight)

        # evaluate on validation set
        prec1, flops = validate(val_loader, model, epoch, logger, sampler, args.uniform_sample)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
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


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def train(train_loader, model, criterion, nce_crite, optimizer, epoch, logger, sampler=None, uniform_sample=False,
          weight=0.0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1
        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")

        if uniform_sample:
            route = sampler.sample(input.size(0)).cuda()
            output, prob = model(input, route)
            if args.prof >= 0: torch.cuda.nvtx.range_pop()
            loss = criterion(output, target)
        else:
            output, prob = model(input)
            if args.prof >= 0: torch.cuda.nvtx.range_pop()
            loss_ce = criterion(output, target)
            # loss_nce = nce_crite(prob)
            # loss = loss_ce + loss_nce * weight
            loss = loss_ce

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())
        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i % args.print_freq == 0:
            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print(
                    'Epoch:[{0}][{1}/{2}] Time:{batch_time.val:.1f}({batch_time.avg:.1f}) '
                    'Speed:{3:.1f}({4:.1f}) Loss:{loss.val:.3f}({loss.avg:.3f}) '
                    'Top@1:{top1.val:.2f}%({top1.avg:.2f}%) '
                    'Top@5:{top5.val:.2f}%({top5.avg:.2f}%)'.format(
                        epoch, i, len(train_loader),
                        args.world_size * args.batch_size / batch_time.val,
                        args.world_size * args.batch_size / batch_time.avg,
                        batch_time=batch_time, loss=losses, top1=top1, top5=top5))

                logger.add_scalar('train/top1', top1.avg, global_step=len(train_loader) * epoch + i)
                logger.add_scalar('train/top5', top5.avg, global_step=len(train_loader) * epoch + i)
                logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'],
                                  global_step=len(train_loader) * epoch + i)
                logger.add_scalar('train/loss', losses.avg, global_step=len(train_loader) * epoch + i)

        if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
        input, target = prefetcher.next()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()


def validate(val_loader, model, epoch, logger, sampler=None, uniform_sample=False):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cost = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    rates_list = [0] * 16
    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            if uniform_sample:
                route = sampler.sample(input.size(0)).cuda()
                output, flops = model(input, route)
            else:
                output, flops = model(input)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        # cost.update(to_python_float(flops), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test:[{0}/{1}] '
                  'Time:{batch_time.val:.1f}({batch_time.avg:.1f}) '
                  'Speed:{2:.1f}({3:.1f}) '
                  'Top@1:{top1.val:.2f}({top1.avg:.2f}) '
                  'Top@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                i, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time, top1=top1, top5=top5))

        input, target = prefetcher.next()

    if args.local_rank == 0:
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        logger.add_scalar('valid/top1', top1.avg, global_step=epoch)
        logger.add_scalar('valid/top5', top5.avg, global_step=epoch)
        # logger.add_scalar('valid/cost', cost.avg, global_step=epoch)
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
    # factor = epoch // 30
    lr = args.lr * (0.1 ** (epoch // args.lrdecay))
    factor = args.lrfact

    """Warmup"""
    # if epoch < 5:
    #     lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    optimizer.param_groups[0]['lr'] = factor * lr
    optimizer.param_groups[1]['lr'] = lr


    # if epoch >= 80:
    #     factor = factor + 1

    # lr = args.lr * (0.1 ** factor)

    """Warmup"""
    # if epoch < 5:
    #     lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


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
