# Semantic Segmentation

In this codebase we provide methods to perform semantic segmentation with ERFNet, UNet, and DeepLabV3+.

## Prerequisites
Create a virtual environment with conda and install dependencies:

```bash
./setup/setup_env.sh
```

## Setup 

Before training, validation, testing, or prediction set the desired configuration in the corresponding configuration file.
You can find the config files at: ```./config/pdc/*.yaml```

## Training

To start the training procedure run the following command:
```bash
python train.py --config ... --export_dir ..
```

## Validation

To start the training procedure run the following command:
```bash
python val.py --config ... --export_dir ... --ckpt_path ...
```

## Testing

To start the training procedure run the following command:
```bash
python test.py --config ... --export_dir ... --ckpt_path ...
```

## Predicting
```bash
python predict.py --config ... --export_dir ... --ckpt_path ...
```