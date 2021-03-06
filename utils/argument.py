import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150, help='Number of epoch')
    parser.add_argument('--train_bs', type=int, default=16, help='Batch size')
    parser.add_argument('--test_bs', type=int, default=1, help='Batch size')
    parser.add_argument('--gpus', type=str, default='0', help='gpus')
    parser.add_argument('--model', type=str, default='R110_C10', help='resnets')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--freeze_gate', action='store_true')
    parser.add_argument('--freeze_net', action='store_true')
    parser.add_argument('--uniform_sample', action='store_true')
    parser.add_argument('--use_gcn', action='store_true')
    parser.add_argument('--use_lc', action='store_true')
    parser.add_argument('--resume_path', type=str, default='', help='')
    parser.add_argument('--log_path', type=str, default='', help='')
    parser.add_argument('--note', type=str, default='', help='note of the experiment')
    parser.add_argument('--network', type=str, default='', help='network choice')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='multi')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--weight', type=float, default=0.0)
    parser.add_argument('--step_ratio', default=0.1, type=float, help='ratio for learning rate deduction')
    parser.add_argument('--warm_up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 iterations')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--data_root', type=str,default = '/home/huanyu/dataset')
    parser.add_argument('--local_rank',type=int, default=0)
    parser.add_argument('--option',type=float, default=3)
    return parser
