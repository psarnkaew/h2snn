#!/usr/bin/bash

# MUTAG
echo "Generating MUTAG pkl file: All Type"
python generate.py --dataset MUTAG --num-ntype 0 --hetero-mode 0 --seed 42 \
                    --train-ratio 0.8 --test-ratio 0.1
echo ""

# PROTEINS
echo "Generating PROTEINS pkl file: All Type"
python generate.py --dataset PROTEINS --num-ntype 0 --hetero-mode 0 --seed 42 \
                    --train-ratio 0.8 --test-ratio 0.1
echo ""

# ENZYMES
echo "Generating ENZYMES pkl file: All Type"
python generate.py --dataset ENZYMES --num-ntype 0 --hetero-mode 0 --seed 42 \
                    --train-ratio 0.8 --test-ratio 0.1
echo ""

# NCI1
# total=15
for num in 10
do
    echo "Generating NCI1 pkl file: $num"
    python generate.py --dataset NCI1 --num-ntype $num --hetero-mode 0 --seed 42 \
                        --train-ratio 0.8 --test-ratio 0.1
    echo ""
done
# echo "Generating NCI1 pkl file: All Type"
# python generate.py --dataset NCI1 --num-ntype 0 --hetero-mode 0 --seed 42 \
#                     --train-ratio 0.8 --test-ratio 0.1
# echo ""

# NCI109
for num in 10
do
    echo "Generating NCI109 pkl file: $num"
    python generate.py --dataset NCI109 --num-ntype $num --hetero-mode 0 --seed 42 \
                        --train-ratio 0.8 --test-ratio 0.1
    echo ""
done
# echo "Generating NCI109 pkl file: All Type"
# python generate.py --dataset NCI109 --num-ntype 0 --hetero-mode 0 --seed 42 \
#                     --train-ratio 0.8 --test-ratio 0.1
# echo ""
