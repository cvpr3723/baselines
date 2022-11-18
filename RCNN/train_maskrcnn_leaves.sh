#block(name=[maskrcnn_leaves], threads=10, memory=48000, subtasks=1, gpus=1, hours=100)

python train_maskrcnn_leaf.py -c configs/maskrcnn_leaves.yaml
