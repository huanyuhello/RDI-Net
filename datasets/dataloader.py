import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
import torch.utils.data
import os
from datasets.mydatasets import MYCIFAR10_AUG, MYCIFAR100_AUG


def get_data(train_bs, test_bs, dataset, data_root='/home/huanyu/dataset', distributed=False, aug_repeat=None):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                               transform=transform_test)
        num_class = 10
    elif dataset == 'cifar10_aug':
        trainset = MYCIFAR10_AUG(root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                               transform=transform_test)
        num_class = 10
    elif dataset == 'cifar100':

        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True,
                                                transform=transform_test)

        num_class = 100
    elif dataset == 'cifar100_aug':
        trainset = MYCIFAR100_AUG(root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True,
                                                transform=transform_test)
        num_class = 100

    elif dataset == 'imagenet':
        print(dataset)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder('/home/huanyu/dataset/ImageNet/train',
                                        transforms.Compose([transforms.RandomResizedCrop(224),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            normalize, ]))
        testset = datasets.ImageFolder('/home/huanyu/dataset/ImageNet/val',
                                       transforms.Compose([transforms.Scale(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           normalize,
                                                           ]))

        num_class = 1000
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    else:
        sampler = torch.utils.data.RandomSampler(trainset)
        test_sampler = torch.utils.data.SequentialSampler(testset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, sampler=sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, sampler=test_sampler, num_workers=4)

    return train_loader, test_loader, num_class
