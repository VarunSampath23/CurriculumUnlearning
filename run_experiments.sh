#!/bin/bash
set -e  # stop on any error

echo "=================================================================="
echo "Training the base model (5 epochs)"
echo "=================================================================="
python train.py \
    --epochs 5 \
    --lr 5e-4 \
    --seed 42

echo ""
echo "=================================================================="
echo "2. Running unlearning experiments with different orderings..."
echo "=================================================================="

for mode in random curriculum_low_first curriculum_high_first; do
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "Running unlearning with mode: $mode"
    echo "──────────────────────────────────────────────────────────────"
    python main.py \
        --mode "$mode" \
        --seed 42 \
        --unlearn_epochs 5 \
        --unlearn_lr 5e-6 \
        --batch_size 32
done

echo ""
echo "All experiments finished!"
echo "Checkpoints → outputs/checkpoints/"
echo "Results are printed above."