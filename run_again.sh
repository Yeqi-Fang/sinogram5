while true; do
    python main.py --data_dir /mnt/d/fyq/sinogram/2e9div_smooth --mode train --batch_size 32 --num_epochs 30 --models_dir checkpoints --attention 1 --lr 1e-5 --light 1
    sleep 1  # 可选，防止CPU占用过高
done