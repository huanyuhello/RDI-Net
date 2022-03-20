import numpy as np
import torch
from utils.flops import flops_conv2d, flops_linear

def converter(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()


class MultiLabelAcc():

    def __init__(self):
        self.total = 0
        self.correct = 0

    def reset(self):
        self.total = 0
        self.correct = 0

    def update(self, output, target):
        _, predict = output.max(1)
        predict, target = converter(predict), converter(target)
        self.total += len(predict)
        self.correct += np.sum(predict == target)

    def get_acc(self):
        return self.correct * 1.0 / self.total


class AverageMetric(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, flops, n):
        # import pdb
        # pdb.set_trace()
        self.sum += flops.sum()
        self.count += n
        self.avg = self.sum / self.count


class MultiAddMetric(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_block):
        self.num_block = num_block * 3
        self.reset()

    def reset(self):
        self.avg = [0] * self.num_block
        self.sum = [0] * self.num_block
        self.count = 0

    def multi_add_110(self):
        # （2*Kh*Kw*Ci - 1）* Co * Ho*Wo
        layer0 = (3 * 3 * 3) * 16 * 32 * 32
        layer1 = (1 * 1 * 16) * 16 * 32 * 32 + (3 * 3 * 16) * 16 * 32 * 32 + (3 * 3 * 16) * 16 * 32 * 32
        layer1_1 = (3 * 3 * 16) * 16 * 32 * 32 + (3 * 3 * 16) * 16 * 32 * 32
        layer2 = (1 * 1 * 16) * 32 * 16 * 16 + (3 * 3 * 16) * 32 * 16 * 16 + (3 * 3 * 32) * 32 * 16 * 16
        layer2_1 = (3 * 3 * 16) * 32 * 16 * 16 + (3 * 3 * 32) * 32 * 16 * 16
        layer3 = (1 * 1 * 32) * 64 * 8 * 8 + (3 * 3 * 32) * 64 * 8 * 8 + (3 * 3 * 64) * 64 * 8 * 8
        layer3_1 = (3 * 3 * 32) * 64 * 8 * 8 + (3 * 3 * 64) * 64 * 8 * 8

        total = layer0
        for i in range(self.num_block):
            if i == 0:
                total += layer1 * self.avg[i]
            elif i > 0 and i < 18:
                total += layer1_1 * self.avg[i]
            elif i == 18:
                total += layer2 * self.avg[i]
            elif i > 18 and i < 36:
                total += layer2_1 * self.avg[i]
            elif i == 36:
                total += layer3 * self.avg[i]
            elif i > 36:
                total += layer3_1 * self.avg[i]
        return total

    def update(self, val, n=1):

        self.count += n
        for i in range(self.num_block):
            self.sum[i] += val[i]
            self.avg[i] = self.sum[i] / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    metric = MultiAddMetric(5)
    a = [5, 4, 6, 7, 3]
    metric.update(a, n=1)
    b = [4, 3, 5, 6, 2]
    metric.update(b, n=1)
    print(metric.avg)
