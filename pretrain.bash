python main.py --mod pretrain --device 0 --dataset_name ETTh1 &
python main.py --mod pretrain --device 1 --dataset_name ETTh2 &
python main.py --mod pretrain --device 2 --dataset_name ETTm1 &
python main.py --mod pretrain --device 3 --dataset_name ETTm2 &
wait  # Ensures the script waits for all background processes to complete
