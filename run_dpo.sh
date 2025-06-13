#!/bin/bash

# define max epoch
max_epoch=20

# loop to train
for ((i=0; i<max_epoch; i++))
do
    echo "==============================================="
    echo "start to train $i/$max_epoch epoch"
    echo "==============================================="
    
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=8 main.py \
        --actual_epoch $i \
        --mode caption_rlhf \
        --version proclean_itc_alm \
        --use_wandb True \
        --max_epochs 1 \
        --dpo_total_epochs $max_epoch
    
    echo "train $i epoch done"
    
    # wait for a while to ensure resource is released
    sleep 60
done

echo "all epochs done!"

#   chmod +x run_dpo.sh
#   ./run_dpo.sh
