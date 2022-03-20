import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features=256, dropout=0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.linear0 = nn.Linear(in_features, out_features, bias=True)
        self.linear1 = nn.Linear(out_features, in_features, bias=True)

    def forward(self, input):
        support = self.linear0(input)
        adj = torch.mm(input, input.t())
        # support = torch.bmm(input, self.weight)
        output = torch.mm(adj, support)
        output = self.linear1(output)

        output = F.dropout(output, self.dropout, training=self.training)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LayerGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features=512):
        super(LayerGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear0 = nn.Linear(in_features, out_features, bias=True)
        self.linear1 = nn.Linear(out_features, in_features, bias=True)

    def forward(self, input, prev, pprev):
        if prev is None:
            prev, pprev = input, input
        elif pprev is None:
            pprev = prev

        fea0 = self.linear0(pprev)
        fea1 = self.linear0(prev)
        fea2 = self.linear0(input)

        support = torch.stack([fea0, fea1, fea2], dim=1)
        # choice 1
        adj = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=support.dtype, device=support.device)

        output = torch.einsum('mn,bni->bmi', adj, support)

        output = self.linear1(output[:, 2, :])

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LayerGraph(nn.Module):
    def __init__(self, in_features, out_features):
        super(LayerGraph, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        self.cfgs = Parameter(torch.FloatTensor([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]]), requires_grad=False)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, prev, pprev):
        if prev is None:
            prev, pprev = input, input
        elif pprev is None:
            pprev = prev

        support = torch.stack([pprev, prev, input], dim=1)

        # support = torch.stack([pprev, prev, input], dim=2)
        # output = self.gmp(support).squeeze()
        # return output

        support = torch.einsum('bni,ij->bnj', support, self.weight)
        output = torch.einsum('mn,bnj->bmj', self.cfgs, support)
        return output[:, 2, :]

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
