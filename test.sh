#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4


python -m torch.distributed.launch --nproc_per_node=$NGPUS main_drn.py --dataset cifar10 --backbone resnet110 --gate_type Gumbel_Gate --block_type GateBlockI --lda_inter_margin 0.5 --lda_intra_margin 0.25 --loss_lda 0.0 --train_bs 32 --aug_repeat 1 --test_bs 100 --log_path ./outputs/pipeline/resnet50 --note baseline
