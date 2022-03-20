'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.utils.data
from models.shallownet import *
from utils.dataloader import get_data
import os
from utils.argument import get_args
from utils.metric import MultiLabelAcc
from utils.metric import AverageMetric
from utils.loss import *
from utils.metric import accuracy
from utils.utils import parse_system, adjust_learning_rate, tensor_path2nums
from utils.dist_utils import dist_print, dist_tqdm, is_main_process
from utils.dist_utils import synchronize, to_python_float
from utils.dist_utils import dist_mean_reduce_tensor, dist_sum_reduce_tensor, dist_cat_reduce_tensor

from apex.parallel import DistributedDataParallel as DDP


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


def train(net, train_loader, logger, epoch):
    net.train()
    for batch_idx, (inputs, labels) in enumerate(dist_tqdm(train_loader)):
        global_step = epoch * len(train_loader) + batch_idx
        inputs, labels = get_variables(inputs, labels)

        result, prob = net(inputs)
        # prob: block * batch * 2 * 1 * 1

        loss_CE = criterion_CE(result, labels)
        loss = loss_CE

        if args.loss_lda is not None:
            loss_lda_inter, loss_lda_intra = criterion_LDA(prob)
            loss += args.loss_lda * (loss_lda_inter + loss_lda_intra)

        if args.loss_w is not None:
            loss_FL = criterion_FL(prob)
            loss += loss_FL * args.loss_w

        if args.loss_d is not None:
            loss_DL = criterion_DL(prob)
            loss += args.loss_d

        # measure accuracy and record loss
        prec, = accuracy(result, labels, topk=(1,))

        # calc gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not args.DEBUG:
            logger.add_scalar('block/train', dist_mean_reduce_tensor(prob[:, :, 0].sum() / prob.shape[0]),
                              global_step=global_step)
            logger.add_scalar('metric/train_prec', dist_mean_reduce_tensor(prec), global_step=global_step)
            logger.add_scalar('meta/loss_single', loss, global_step=global_step)
            # logger.add_scalar('loss/prob', dist_mean_reduce_tensor(loss_FL), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            logger.add_scalar('loss/cls', dist_mean_reduce_tensor(loss_CE), global_step=global_step)
            logger.add_scalar('loss/all', dist_mean_reduce_tensor(loss), global_step=global_step)
            # logger.add_scalar('loss/lda', dist_mean_reduce_tensor(loss_lda_inter + loss_lda_intra),
            #                   global_step=global_step)
            # logger.add_scalar('loss/lda_inter', dist_mean_reduce_tensor(loss_lda_inter), global_step=global_step)
            # logger.add_scalar('loss/lda_intra', dist_mean_reduce_tensor(loss_lda_intra), global_step=global_step)


def valid(net, test_loader, logger, epoch):
    counter = MultiLabelAcc()
    probable = AverageMetric()
    path_nums = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dist_tqdm(test_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            result, prob = net(inputs)
            counter.update(result, targets)
            probable.update(prob[:, :, 0].sum() / prob.shape[0], inputs.size(0))
            path_nums += tensor_path2nums(prob)

    this_machine_total = torch.tensor([counter.total, ], dtype=torch.long).cuda()
    this_machine_correct = torch.tensor([counter.correct, ], dtype=torch.long).cuda()
    all_total = dist_sum_reduce_tensor(this_machine_total)
    all_correct = dist_sum_reduce_tensor(this_machine_correct)
    this_machine_prob = torch.tensor([probable.avg], dtype=torch.float).cuda()
    all_prob = dist_mean_reduce_tensor(this_machine_prob)
    this_machine_paths = torch.tensor(path_nums, dtype=torch.long).cuda()
    all_paths = dist_cat_reduce_tensor(this_machine_paths)

    if not args.DEBUG:
        logger.add_scalar('metric/val_prec', to_python_float(all_correct) * 100.0 / to_python_float(all_total),
                          global_step=epoch)
        logger.add_scalar('block/val', all_prob, global_step=epoch)
        logger.add_scalar('path_num', len(torch.unique(all_paths)), global_step=epoch)
        logger.add_scalar('debug/precise_single', counter.correct * 100.0 / counter.total, global_step=epoch)

    return to_python_float(all_correct) * 100.0 / to_python_float(all_total)


def save_model(top1, best_acc, epoch, save_path):
    if top1 > best_acc:
        dist_print('Saving best model..')
        state = {'net': net.state_dict(), 'acc': top1, 'epoch': epoch, }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        model_path = os.path.join(save_path, 'ckpt.pth')
        torch.save(state, model_path)
        best_acc = top1
    return best_acc


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
    best_acc = 0  # best test accuracy
    train_loader, test_loader, classes = get_data(args.train_bs, args.test_bs, dataset=args.dataset,
                                                  data_root=args.data_root, distributed=distributed,
                                                  aug_repeat=args.aug_repeat)

    cudnn.benchmark = True
    # Model
    dist_print('==> Building model..')

    net = ResNetSkip(args.backbone, args.block_type, classes, args.beta, args.finetune).cuda()

    if distributed:
        net = net.cuda()
        net = DDP(net, delay_allreduce=True)  # TODO test no delay
    # else:
    #     net = nn.DataParallel(net)

    if args.finetune:
        dist_print('==> Load finetune model...')
        # net.load_state_dict(load_model(
        #     'outputs/imagenet/20200528_195337_resnet50_imagenet_beta_25_GateBlockI_GateI_w_0.00000_l_0.000_num_16_resnet50_ImageNet/ckpt.pth'))
    elif not args.pretrain:
        dist_print('==> Load pretrain model...')
    else:
        dist_print('pretrain model')

    logger, save_path = parse_system(args)

    criterion_CE = nn.CrossEntropyLoss().cuda()
    criterion_FL = FLOPSL1Loss(target=args.num_target).cuda()
    criterion_LDA = LDALoss(args.aug_repeat, args.lda_intra_margin, args.lda_inter_margin).cuda()
    criterion_DL = DiversityLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        dist_print('\nEpoch: %d' % epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train(net, train_loader, logger, epoch)

        top1 = valid(net, test_loader, logger, epoch)

        synchronize()
        if is_main_process():
            best_acc = save_model(top1, best_acc, epoch, save_path)
        synchronize()

        logger.add_scalar('best_acc', best_acc, global_step=epoch)
    dist_print('\nRESNET: acc %f' % (best_acc))

    torch.set_printoptions(profile="full")

    logger.close()
