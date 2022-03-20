import torch


# from torch.distributions.uniform import Uniform
# m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
# route = m.sample(sample_shape=(5,)).floor()
# print(route.dtype)
# route = torch.zeros(route.size(0), 5, dtype=route.dtype, device=route.device).scatter_(1, route.long(), 1)
# print(route)
#
# import torch as t
# import numpy as np
#
# batch_size = 8
# class_num = 10
# label = np.random.randint(0,class_num,size=(batch_size,1))
# label = t.LongTensor(label)
# print(label.dtype)
# print(label.shape)
# y_one_hot = t.zeros(batch_size,class_num).scatter_(1,label,1)
# print(y_one_hot)


class B():
    def __init__(self):
        self.sum = 100

    def forward(self, a, b):
        b.append(self.sum)
        return a



class A():
    def __init__(self):
        self.sum = []
        self.b = B()

    def forward(self):
        self.b.forward(10, self.sum)
        return self.sum

# print(torch.FloatTensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]]))
