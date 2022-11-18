# HAPT

This folder contains the instructions to run the code of the paper *Hierarchical Approach for Joint Semantic, Plant Instance, and Leaf Instance Segmentation in the Agricultural Domain* ([arXiv](https://arxiv.org/pdf/2210.07879.pdf)). 

The source code can be found [here](https://github.com/PRBonn/HAPT).

## How To run

1. Clone the linked repository
2. Install the requirements 
3. In file `datasets/datasets.py` add your custom dataset class and data loader
4. Modify the `config/config.yaml`
5. `python train_hapt.py`

To load pre-trained weights execute 5 as `python train_hapt.py --weights pre_trained_weights.ckpt`.
