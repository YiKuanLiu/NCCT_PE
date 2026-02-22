#!/bin/bash
# choose the model and input 
for f in {0..9}; do
    echo "Running with f=$f"
    python3 ./main_DDP_10folds_MLP_inhale.py --N_rounds "$f"
    python3 ./main_DDP_10folds_MLP_IE.py --N_rounds "$f"
    python3 ./main_DDP_10folds_SwinUNETR_inhale.py --N_rounds "$f"
    python3 ./main_DDP_10folds_SwinUNETR_IE.py --N_rounds "$f"
done