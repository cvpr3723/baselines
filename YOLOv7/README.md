# Run YOLOv7 on our data 

## Setup instructions 
1. clone YOLOv7 repo to the 'src' dir:
```sh 
$ git clone https://github.com/WongKinYiu/yolov7 to ./src
```

2. in the Makefile:
+ set data_dir to the parent_dir of the PDC dataset
+ set the log_dir to the output parent dir desired
+ change the cfg file path 
+ in data/pdc.yaml, change train val test paths to the correct place 
+ in src/yolov7/test.py change the anno_json path to the correct one

## Running the code
### For plant detection

for training:
```sh
make train_pdc
```   
for testing:
```
make test_pdc
```

### For leaves detection 

for training:
```sh
make train_leaves
```   
for testing:
```
make test_leaves
```


