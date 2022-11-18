#block(name=[maskrcnn], threads=10, memory=48000, subtasks=1, gpus=1, hours=100)

python train_maskrcnn.py -c configs/maskrcnn.yaml
