#block(name=[fastrcnn_leaves], threads=10, memory=48000, subtasks=1, gpus=1, hours=100)

python train_fasterrcnn_leaf.py -c configs/fasterrcnn_leaves.yaml
