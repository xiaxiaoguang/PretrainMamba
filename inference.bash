#!/bin/bash

# Array of lengths
lengths=(96 192 336 720)
dataset_name='ETTh1'
# Loop through each length
for len in "${lengths[@]}"; do
  # Execute each command with the current length
  echo "Running with pred_len=${len} in ${dataset_name}"
  python main.py --mod inference --device 1 --dataset_name "$dataset_name" --pred_len "$len"
  #python main.py --mod inference --device 1 --pre_layer 0 --frozen_type none --dataset_name "$dataset_name" --pred_len "$len" --model_path '...'
  python main.py --mod inference --device 1 --pre_layer 0 --frozen_type none --dataset_name "$dataset_name" --pred_len "$len"
  python main.py --mod inference --device 1 --pre_layer 0 --frozen_type select --dataset_name "$dataset_name" --pred_len "$len"
  python main.py --mod inference --device 1 --pre_layer 0 --frozen_type selectAD --dataset_name "$dataset_name" --pred_len "$len"
done

echo "All tasks completed."