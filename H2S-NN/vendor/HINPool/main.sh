#!/usr/bin/bash

#### No need to modify ####
model=HGPool
backbone=Simple  # Simple
###########################

dataset=MUTAG  # MUTAG / NCI1 / NCI109 / PROTEINS / ENZYMES
num_types=0  # 0 / 10

# Optimizer
batch=20  # 20 / 150 / 200
lr=1e-3  # 1e-3 / 5e-4 / 2e-4 / 1e-4 / 5e-5 / 2e-5 / 1e-5
wd=1e-4

# HGNN
head=8
hidden=128

# HINPool Architecture
fusion=mean  # concat / mean / max / att_shared / att_sep
skip=concat  # concat / mean / max / att
mixed=0
mlp=1
mul=1
norm=0

# Ablation Study
do_cross_layer=0
do_TA_readout=0
pool_by=type

out=$dataset

echo "Searching $model using $backbone for $dataset"

python -W ignore::UserWarning: main.py \
    --dataset $dataset --model $model --backbone $backbone \
    --hetero-mode 0  --num-atom-types $num_types \
    --seed 46 --out-dir $out \
    --train-ratio 0.8 --test-ratio 0.1 \
    --epochs 1500 --lr $lr --weight-decay $wd --warmup-epoch 60 \
    --num-heads $head --hidden-dim $hidden \
    --batch-size $batch --patience 100 \
    --repeat 5  --cuda 1 \
    --pool-ratio 0.9 0.9 0.9 --hop 1 \
    --pool mean \
    --fusion $fusion --skip-aggr $skip --norm $norm \
    --pool-by $pool_by \
    --use-mixed $mixed --mlp-score $mlp --mul-score $mul \
    --do-TA-readout $do_TA_readout --do-cross-layer $do_cross_layer
    