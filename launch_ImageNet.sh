#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4

#python main_dist.py --dataset imagenet --model R50_ImgNet --train_bs 32 --test_bs 32 --lr 0.1 --uniform_sample --epochs 90 --log_path ./outputs/Imagenet --note resnet50_ImageNet_STAGE1_sampling
#python main.py -a resnet50 --b 224 --workers 4 --opt-level O1 --loss-scale 128.0
#python main.py -a resnet50 --b 224 --workers 4 --opt-level O3
#python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --uniform_sample --model R50_ImgNet --loss-scale 128.0 --b 48 --workers 4 --opt-level O0 --epochs 90 --log_path ./outputs/Imagenet --note resnet50_ImageNet_STAGE1_sampling
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --model R50_ImgNet --b 48 --workers 4 --opt-level O0 --epochs 100 --lr 0.1 --log_path ./outputs/Imagenet --note resnet50_ImageNet_STAGE2_120 --resume ./outputs/Imagenet/20200908_132324_R50_ImgNet_resnet50_ImageNet_STAGE1_sampling/ckpt.pth
#python main.py --uniform_sample --model R50_ImgNet --loss-scale 128.0 --b 80 --workers 4 --opt-level O2 --epochs 90 --log_path ./outputs/Imagenet --note resnet50_ImageNet_STAGE1_sampling


#python train_gcn.py --model R50_ImgNet --b 160 --lr 0.1 --log_path ./outputs/Imagenet --note resnet50_ImageNet_STAGE2_Router --resume_path ./outputs/Imagenet/20200926_140304_R50_ImgNet_resnet50_ImageNet_STAGE2/ckpt.pth

