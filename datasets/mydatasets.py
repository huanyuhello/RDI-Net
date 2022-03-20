import torchvision
import torch
import cv2


class MYCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        super(MYCIFAR10, self).__init__(**kwargs)

    def __getitem__(self, index):
        img, target = super(MYCIFAR10, self).__getitem__(index)

        hist = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)

        return img, target, hist


class MYCIFAR10_AUG(torchvision.datasets.CIFAR10):
    def __init__(self, repeat=2, **kwargs):
        super(MYCIFAR10_AUG, self).__init__(**kwargs)
        self.repeat = repeat
        assert self.train == True

    def __getitem__(self, index):
        images = []
        targets = []
        for i in range(self.repeat):
            img, target = super(MYCIFAR10_AUG, self).__getitem__(index)
            images.append(img)
            targets.append(target)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)
        return images, targets


class MYCIFAR100_AUG(torchvision.datasets.CIFAR100):
    def __init__(self, repeat=2, **kwargs):
        super(MYCIFAR100_AUG, self).__init__(**kwargs)
        self.repeat = repeat
        assert self.train == True

    def __getitem__(self, index):
        images = []
        targets = []
        for i in range(self.repeat):
            img, target = super(MYCIFAR100_AUG, self).__getitem__(index)
            images.append(img)
            targets.append(target)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)
        return images, targets


def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    # import pdb; pdb.set_trace()
    if isinstance(elem, torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, tuple):
        return ((my_collate(samples) for samples in zip(*batch)))
    else:
        raise NotImplementedError


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = MYCIFAR10_AUG(repeat=10, root='/home/zequn/datatset/', train=True, download=True,
                             transform=transform_train)
    sampler = torch.utils.data.RandomSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=10, sampler=sampler,
                                               num_workers=2)  # , collate_fn = my_collate)
    for (data, labels) in train_loader:
        print(labels)
        print(data.shape)
        break
