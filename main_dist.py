'''Train CIFAR10 with PyTorch.'''

import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.utils.data
from datasets.dataloader import get_data
import os
from utils.argument import get_args
from utils.metric import MultiLabelAcc, AverageMetric
from utils.loss import CostLoss
from utils.loss import InfoNCELoss
from utils.metric import accuracy
from utils.utils import parse_system
from utils.dist_utils import dist_print, dist_tqdm, is_main_process
from utils.dist_utils import synchronize, to_python_float
from utils.dist_utils import dist_mean_reduce_tensor, dist_sum_reduce_tensor
from models.route_net import *
from apex.parallel import DistributedDataParallel as DDP
from load_model import get_model
from apex import amp

def get_variables(inputs, labels):
    if 'aug' in args.dataset:
        assert len(inputs.shape) == 5
        assert len(labels.shape) == 2
        inputs = inputs.view(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3],
                             inputs.shape[4]).cuda()
        labels = labels.view(-1).cuda()
    else:
        inputs, labels = inputs.cuda(), labels.cuda()
    return inputs, labels


def train(train_loader, logger, epoch, sample=False, weight=0.0):
    net.train()
    for batch_idx, (inputs, labels) in enumerate(dist_tqdm(train_loader)):
        global_step = epoch * len(train_loader) + batch_idx
        inputs, labels = get_variables(inputs, labels)
        if sample:
            route = sampler.sample(inputs.size(0)).cuda()
            result, prob = net(inputs, route)
        else:
            result, prob = net(inputs)
            loss_nce = nce_crite(prob)
        # prob: block * batch * 2 * 1 * 1
        loss_ce = cre_crite(result, labels)

        if sample:
            loss = loss_ce
        else:
            loss = loss_ce + loss_nce * weight
        # calc gradient and do SGD step
        optimizer.zero_grad()

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()
        

        # measure accuracy and record loss
        prec, = accuracy(result, labels, topk=(1,))
        if not args.DEBUG:
            logger.add_scalar('train/precision', dist_mean_reduce_tensor(prec), global_step=global_step)
            logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            logger.add_scalar('train/loss_ce', dist_mean_reduce_tensor(loss_ce), global_step=global_step)
            # logger.add_scalar('train/loss_nce', dist_mean_reduce_tensor(loss_nce), global_step=global_step)


def valid(test_loader, logger, epoch, sample=False):
    PrecMetric = MultiLabelAcc()
    FlopMetric = AverageMetric()
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dist_tqdm(test_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            if sample:
                route = sampler.sample(inputs.size(0)).cuda()
                result, flops = net(inputs, route)
            else:
                result, flops = net(inputs)

            PrecMetric.update(result, targets)
            FlopMetric.update(flops, inputs.size(0))

    total_sample = dist_sum_reduce_tensor(torch.tensor([PrecMetric.total, ], dtype=torch.long).cuda())
    flops_sample = dist_sum_reduce_tensor(torch.tensor([FlopMetric.avg, ], dtype=torch.long).cuda())
    correct_sample = dist_sum_reduce_tensor(torch.tensor([PrecMetric.correct, ], dtype=torch.long).cuda())


    if not args.DEBUG:
        logger.add_scalar('valid/precision', to_python_float(correct_sample) * 100.0 / to_python_float(total_sample),
                          global_step=epoch)
        logger.add_scalar('valid/flops', to_python_float(flops_sample), global_step=epoch)

    return to_python_float(correct_sample) * 100.0 / to_python_float(total_sample), flops_sample


def save_model(top1, flops, best_acc, best_flops, epoch, save_path):
    if top1 > best_acc:
        dist_print('Saving best model..')
        state = {'net': net.state_dict(), 'acc': top1, 'epoch': epoch, }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        model_path = os.path.join(save_path, 'ckpt.pth')
        torch.save(state, model_path)
        best_acc = top1
        best_flops = flops
    return best_acc, best_flops


if __name__ == "__main__":

    args = get_args().parse_args()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        assert int(os.environ[
                       'WORLD_SIZE']) == torch.cuda.device_count(), 'It should be the same number of devices and processes'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(args)

    best_acc = 0
    best_flops = 0# best test accuracy

    train_loader, test_loader, classes = get_data(args.train_bs, args.test_bs, dataset=args.dataset,
                                                  data_root=args.data_root, distributed=distributed)

    cudnn.benchmark = True
    # Model
    dist_print('==> Building model..')

    net, nblock = get_model(model=args.model, freeze_gate=args.freeze_gate,
                            uniform_sample=False, freeze_net=args.freeze_net,
                            resume_path=args.resume_path)
    sampler = UniformSample(nblock, options=args.option)

    if distributed:
        net = net.cuda()
        net = DDP(net, delay_allreduce=True)  # TODO test no delay
    else:
        net = nn.DataParallel(net.cuda())

    logger, save_path = parse_system(args)

    cre_crite = nn.CrossEntropyLoss().cuda()
    nce_crite = InfoNCELoss().cuda()


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 300], gamma=0.1)

    for epoch in range(args.epochs):
        dist_print('\nEpoch: %d' % epoch)

        scheduler.step(epoch)

        train(train_loader, logger, epoch, False, args.weight)

        top1, flops = valid(test_loader, logger, epoch, False)

        synchronize()
        if is_main_process():
            best_acc, best_flops = save_model(top1, flops, best_acc, best_flops, epoch, save_path)
        synchronize()

        logger.add_scalar('best/acc', best_acc, global_step=epoch)
        logger.add_scalar('best/flops', best_flops, global_step=epoch)

    dist_print('\n HARD GUMBEL_SOFTMAX RESNET: acc %f' % (best_acc))

    torch.set_printoptions(profile="full")

    logger.close()
