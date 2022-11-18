#!/usr/bin/env python3
"""Script to convert PDC annotations to YOLO format annotations.
"""

import os
import numpy as np
import cv2

def one_anno_leaf(id_path, out_dir):
    out_file_path = os.path.join(out_dir, os.path.basename(id_path).split('.')[0]+".txt")
    print(f"Write to file {out_file_path}")
    out_file = open(out_file_path, "x")

    id_mask = cv2.imread(id_path, cv2.IMREAD_UNCHANGED)
    img_width_x = id_mask.shape[1]
    img_height_y = id_mask.shape[0]

    plant_ids = np.unique(id_mask)

    for plant_id in plant_ids:
        if plant_id == 0:
            continue # skip because this is just bg
        # create an instance 
        if not np.any(id_mask==plant_id):
            print("no semantics found.")
            print(f"{sem_path} {plant_id}")
            continue
        # assert np.all(plant_class_mask == plant_class_mask[0]), f"{sem_path}: Error on plant id {plant_id} each plant id must be associated with only one class (crop or weed)"
        plant_class = 0

        plant_indices = np.where(id_mask==plant_id)

        plant_max_x = plant_indices[1].max()
        plant_min_x = plant_indices[1].min()

        plant_max_y = plant_indices[0].max()
        plant_min_y = plant_indices[0].min()

        plant_center_x = int((plant_max_x + plant_min_x)/2)
        plant_center_y = int((plant_max_y + plant_min_y)/2)

        plant_width_x = plant_max_x - plant_min_x
        plant_height_y = plant_max_y - plant_min_y

        out_file.write(f"{plant_class} {plant_center_x/img_width_x} {plant_center_y/img_height_y} {plant_width_x/img_width_x} {plant_height_y/img_height_y}\n")


    out_file.close()


if __name__ == '__main__':
    id_dir = "/plant/dataset/challenge/path/val/leaf_instances/"
    output_dir = "/plant/dataset/challenge/path/yolo_annos/leaves/val/labels"

    for img_name in os.listdir(id_dir):
        id_path = os.path.join(id_dir, img_name)
        one_anno_leaf(id_path, output_dir)
