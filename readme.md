# Command to run script

```
python train.py --problem double_integrator_single  --epochs 1000 --lr_decay 400

python train.py --problem double_integrator_multi   --epochs 5000 --lr_decay 800 --hidden_dim 128

python train.py --problem single_integrator_swarm   --epochs 3500 --lr_decay 800 --hidden_dim 192 --n_blocks 4

python train.py --problem quadcopter_multi   --epochs 2800 --lr_decay 600 --hidden_dim 128

python train.py --problem quadcopter_swarm  --epochs 4500 --lr_decay 800 --hidden_dim 128 --n_blocks 4

```