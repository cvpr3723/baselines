data_dir="/plant/data/challenge/dir/path"
log_dir="/output/dir/path"
baseline_name=yolov7

SHELL = /bin/sh

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

build:
	docker build -t $(baseline_name)_docker -f Dockerfile --build-arg USER_ID=$(CURRENT_UID) --build-arg GROUP_ID=$(CURRENT_GID) .
train_pdc:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}:/data \
        -v $(log_dir):/logs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/train.py --workers 8 --batch-size 16 --data /src/data/pdc.yaml --img 1024 1024 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
test_pdc:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}:/data \
        -v $(log_dir):/logs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/test.py --data /src/data/pdc.yaml --img 1024 --batch 16 --conf-thres 0.0 --iou-thres 0.65 --weights runs/train/yolov7/weights/best.pt --name yolov7_pdc_val --task test --save-txt --save-conf --save-json
infer_pdc:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}:/data \
        -v $(log_dir):/logs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/detect.py --img-size 1024 --conf-thres 0.25 --iou-thres 0.65 --weights runs/train/yolov7/weights/best.pt --save-txt --save-conf --source /data/yolo_annos/test/images
test_leaves:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}:/data \
        -v $(log_dir):/logs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/test.py --data /src/data/pdc_leaves.yaml --img 1024 --batch 16 --conf-thres 0.0 --iou-thres 0.65 --weights runs/train/yolov7/weights/best.pt --name yolov7_pdc_leaves_test --task test --save-txt --save-conf --save-json --verbose 
train_leaves:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}:/data \
        -v $(log_dir):/logs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/train.py --workers 8 --batch-size 16 --data /src/data/pdc_leaves.yaml --img 1024 1024 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
val_leaves:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}:/data \
        -v $(log_dir):/logs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/test.py --data /src/data/pdc_leaves.yaml --img 1024 --batch 16 --conf-thres 0.0 --iou-thres 0.65 --weights runs/train/yolov7/weights/best.pt --name yolov7_pdc_leaves_val --task val --save-txt --save-conf --save-json --verbose 
val_pdc:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}:/data \
        -v $(log_dir):/logs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/test.py --data /src/data/pdc.yaml --img 1024 --batch 16 --conf-thres 0.0 --iou-thres 0.65 --weights runs/train/yolov7/weights/best.pt --name yolov7_pdc_val --task val --save-txt --save-conf --save-json


