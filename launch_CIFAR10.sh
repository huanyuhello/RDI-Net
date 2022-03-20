#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3
export NGPUS=2

#python -m torch.distributed.launch --nproc_per_node=$NGPUS main_dist.py --dataset cifar10 --backbone resnet110 --train_bs 128 --test_bs 100 --epochs 150 --lr 0.1 --log_path ./outputs/ --note baseline

#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --uniform_sample --log_path ./outputs/ --note Uniform_STAGE1
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --freeze_net --log_path ./outputs/ --note Train_STAGE2 --resume_path ./outputs/20200820_211252__cifar10_baseline_STAGE1/ckpt.pth
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --freeze_gate --log_path ./outputs/ --note Train_STAGE3 --resume_path ./outputs/20200821_105338_R110_C10_cifar10_Freeze_STAGE2/ckpt.pth

#python main_dist.py --dataset cifar10 --backbone resnet110 --train_bs 256 --test_bs 100 --epochs 220 --lr 0.1 --log_path ./outputs/ --note baseline_ReLU_2LINEAR
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --freeze_gate --log_path ./outputs/ --note Train_STAGE3 --resume_path ./outputs/20200821_110419_R110_C10_cifar10_Train_STAGE2/ckpt.pth

#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --log_path ./outputs/ --uniform_sample --note baseline_LC_STAGE1_NO_C
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --log_path ./outputs/ --note baseline_LC_STAGE2 --resume_path ./outputs/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --log_path ./outputs/ --freeze_gate --note baseline_LC_STAGE3 --resume_path ./outputs/20200825_165518_R110_C10_cifar10_baseline_LC_STAGE2/ckpt.pth

#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --weight 0.0 --log_path ./outputs/ --note baseline_LC_STAGE2_FLOPS --resume_path ./outputs/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --log_path ./outputs/ --freeze_gate --note baseline_LC_STAGE3 --resume_path ./outputs/20200825_165518_R110_C10_cifar10_baseline_LC_STAGE2/ckpt.pth

#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --weight 0.0 --log_path ./outputs/ --note baseline_LC_STAGE2_FLOPS_GCN_SAMPLE256_DROPOUT --resume_path ./outputs/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --w1 0.05 --log_path ./outputs/ --note baseline_LC_STAGE2_FLOPS_GCN_2LAYER256_DROPOUT_SM --resume_path ./outputs/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --w1 0.1 --log_path ./outputs/ --note baseline_LC_STAGE2_FLOPS_GCN_2LAYER256_DROPOUT_SM --resume_path ./outputs/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
#python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --w1 0.5 --log_path ./outputs/ --note baseline_LC_STAGE2_FLOPS_GCN_2LAYER256_DROPOUT_SM --resume_path ./outputs/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth

#python main_dist.py --dataset cifar10_aug --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --weight 0.8 --log_path ./outputs/ --note baseline_STAGE2_FLOPS_GCN_2LAYER256_DROPOUT0.4_NCE0.8 --resume_path ./outputs/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
python main_dist.py --dataset cifar10 --train_bs 256 --test_bs 100 --epochs 320 --lr 0.1 --model R110_C10 --weight 0.0 --log_path ./outputs/ --note REBUTTAL --resume_path ./outputs/CIFAR10/20200825_090114_R110_C10_cifar10_baseline_LC_STAGE1_NO_C/ckpt.pth
