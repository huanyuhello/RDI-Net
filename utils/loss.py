import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F


class CostLoss(nn.Module):
    def __init__(self, blocks, target):
        super(CostLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = torch.ones([blocks], dtype=torch.float).cuda() * target

    def forward(self, logits):
        logits = logits.sum(dim=1)
        probs = (logits - self.target).pow(2)
        return self.criterion(probs, torch.zeros(probs.size()).cuda())


class SimilarLoss(nn.Module):
    def __init__(self):
        super(SimilarLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, logits, paths):
        logits = F.softmax(logits, dim=1)

        entropy = -(logits * torch.log2(logits)).sum(dim=1, keepdim=True)
        similar0 = torch.mm(entropy, entropy.t())
        similar0 = F.normalize(similar0, p=2, dim=0)

        paths = paths.view(paths.size(0), -1)
        similar1 = torch.mm(paths, paths.t())
        similar1 = F.normalize(similar1, p=2, dim=0)
        return self.criterion(similar0, similar1)


class InfoNCELoss(nn.Module):
    def __init__(self, temp=1.0, K=65536, num_block=54, option=3):
        super(InfoNCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.T = temp
        self.K = K
        self.register_buffer("queue", torch.randn(K, num_block, option))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if self.K % batch_size == 0:  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[ptr:ptr + batch_size, :, :] = keys
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[0] = ptr
        else:
            return

    def forward(self, logits):
        # logits [2n, block, 3]
        logits = logits.view(-1, 2, logits.shape[1], logits.shape[2])
        sample0 = logits[:, 0, :, :]
        sample0 = nn.functional.normalize(sample0, dim=1)
        sample1 = logits[:, 1, :, :]
        sample1 = nn.functional.normalize(sample1, dim=1)

        self._dequeue_and_enqueue(sample1)
        # Einstein sum is more intuitive
        # positive logits: Nx1
        pos = torch.einsum('nci,nci->n', sample0, sample1).unsqueeze(-1)
        # negative logits: NxN
        neg = torch.einsum('nci,kci->nk', sample0, self.queue)

        # logits: Nx(1+N)
        logits = torch.cat([pos, neg], dim=1)

        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return self.criterion(logits, labels)


class LDALoss(nn.Module):
    def __init__(self, num_per_path, margin_intra, margin_inter, norm=2):
        super(LDALoss, self).__init__()
        self.norm = norm
        self.num_per_path = num_per_path
        self.margin_intra = margin_intra
        self.margin_inter = margin_inter

    def _get_center(self, path_fea):
        batch_num, block_num, _, _, _ = path_fea.shape
        path_fea = path_fea.permute(1, 0, 2, 3, 4)
        path_fea = path_fea.view(block_num, batch_num // self.num_per_path, self.num_per_path, 2)
        center = torch.mean(path_fea, dim=2)
        return center.view(block_num, -1, 2).permute(1, 0, 2).contiguous().view(-1, block_num * 2)

    def _get_inter_loss(self, center):
        # center: num_sample, block * 2
        inter_term = 0
        num_sample = center.shape[0]
        for i in range(num_sample):
            for j in range(i + 1, num_sample):
                diff = torch.norm(center[i, :] - center[j, :], self.norm)
                inter_term += torch.clamp(self.margin_inter - diff, min=0.0) ** 2
        inter_term /= num_sample * (num_sample - 1) / 2
        return inter_term

    def _get_intra_loss(self, center, path_fea):
        # center : num_sample ,block * 2
        # path_fea : batch * block *  * 2 * 1 * 1
        num_sample = center.shape[0]
        # num_block = path_fea.shape[0]
        intra_term = 0
        for i in range(num_sample):
            center_i = center[i]
            # block * 2
            for j in range(i * self.num_per_path, (i + 1) * self.num_per_path):
                cur_fea = path_fea[j, :, :, :, :].contiguous().view(-1)
                diff = torch.norm(cur_fea - center_i, self.norm)
                intra_term += torch.clamp(diff - self.margin_intra, min=0.0) ** 2
        intra_term /= num_sample * self.num_per_path
        return intra_term

    def forward(self, path_fea):
        # fea :   batch* block * 2 * 1 * 1
        # print(path_fea.shape)
        assert path_fea.shape[0] % self.num_per_path == 0
        center = self._get_center(path_fea)
        inter_loss = self._get_inter_loss(center)
        intra_loss = self._get_intra_loss(center, path_fea)
        return inter_loss, intra_loss


if __name__ == "__main__":
    logits = torch.stack([c, b], dim=0)
    entropy = -(logits * torch.log2(logits)).sum(dim=1, keepdim=True)
    similar = torch.mm(entropy, entropy.t())
    print(similar)
    similar = F.normalize(similar, p=2, dim=0)
    print(similar)

    similar = torch.mm(logits, logits.t())
    similar = F.normalize(similar, p=2, dim=1)
    print(similar)
