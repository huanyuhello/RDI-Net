from datasets.dataloader import get_data
import torch
import os

# def get_histogram(images):
#     hist = torch.histc(images, bins=200, min=0, max=0)
#     # # normalize histogram to sum to 1
#     hist = hist.div(hist.sum())
#     return hist

# train_loader, test_loader, classes = get_data(1, 1, dataset='cifar10')

# hists = []

# for batch_idx, (inputs, labels) in enumerate(train_loader):
#     if batch_idx % 100 == 0:
#         print(batch_idx)
#     # print(inputs.shape)
#     hist = get_histogram(inputs[0])
#     hists.append(hist)

# # torch.save(hists, './hist_dic.pth')

# # hists = torch.load('./hist_dic.pth')
# t_hists = torch.stack(hists, dim=0)
# print(t_hists.shape)

# ssim = torch.mm(t_hists, t_hists.t())
# torch.save(ssim, './ssim_dic.pth')

case = 0
total = 0

ssim = torch.load('./ssim_dic.pth')
print(ssim.shape)
for i in range(ssim.size(0)):
    sorted, indices = torch.sort(ssim[i], descending=True)
    for j in range(ssim.size(1)-1):
        if sorted[j] == sorted[j+1]:
            case += 1
        total += 1
    print(case, total)

print('total')
