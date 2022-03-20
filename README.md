# RDI-Net

This contains the PyTorch implementation of the RDI-Net papers,

RDI-Net: Relational Dynamic Inference Networks, ICCV 2021.

By Huanyu Wang, Songyuan Li, Shihao Su, Zequn Qin, Xi Li.

## Abstract 
Dynamic inference networks, aimed at promoting computational efficiency, go along an adaptive executing path for a given sample. Prevalent methods typically assign a router for each convolutional block and sequentially make block-by-block executing decisions, without considering the relations during the dynamic inference. In this paper, we model the relations for dynamic inference from two aspects: the routers and the samples. We design a novel type of router called the relational router to model the relations among routers for a given sample. In principle, the cur- rent relational router aggregates the contextual features of preceding routers by graph convolution and propagates its router features to subsequent ones, making the executing decision for the current block in a long-range manner. Fur- thermore, we model the relation between samples by intro- ducing a Sample Relation Module (SRM), encouraging cor- related samples to go along correlated executing paths. As a whole, we call our method the Relational Dynamic Inference Network (RDI-Net). Extensive experiments on CIFAR- 10/100 and ImageNet show that RDI-Net achieves state-of- the-art performance and computational cost reduction. 

[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_RDI-Net_Relational_Dynamic_Inference_Networks_ICCV_2021_paper.html)

## Usage
  python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --model R110_C10 --weight 0.0 --log_path ./outputs/ --note REBUTTAL --resume_path ./outputs/CIFAR10/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
### Disclamer

We based our code on [Convnet-AIG](https://github.com/andreasveit/convnet-aig), CoDiNet(https://github.com/huanyuhello/codinet), please go show some support!
