cd ../
# Reach 88% test accurachy
python train.py neighbor --prefix neighbor --epochs 100 --lr 0.0003 --data pair_obs1.npz --weight_decay 0.001
