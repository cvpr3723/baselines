#!/usr/bin/env python3
"""Script to convert PDC annotations to YOLO format annotations.
"""

import os
import numpy as np
import cv2

def one_anno(sem_path, id_path, out_dir):
    out_file_path = os.path.join(out_dir, os.path.basename(sem_path).split('.')[0]+".txt")
    print(f"Write to file {out_file_path}")
    out_file = open(out_file_path, "x")

    id_mask = cv2.imread(id_path, cv2.IMREAD_UNCHANGED)
    img_width_x = id_mask.shape[1]
    img_height_y = id_mask.shape[0]

    sem_mask = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
    # merge partial classes
    sem_mask[sem_mask == 3] = 1 # partial crop --> crop
    sem_mask[sem_mask == 4] = 2 # partial weed --> weed

    plant_ids = np.unique(id_mask)

    soil_mask = sem_mask == 0
    id_mask[soil_mask] = 0

    for plant_id in plant_ids:
        if plant_id == 0:
            continue # skip because this is just bg
        # create an instance 
        plant_class_mask = sem_mask[id_mask==plant_id]
        if not np.any(id_mask==plant_id):
            print("no semantics found.")
            print(f"{sem_path} {plant_id}")
            continue
        plant_class = plant_class_mask[0]

        plant_indices = np.where(id_mask==plant_id)

        plant_max_x = plant_indices[1].max()
        plant_min_x = plant_indices[1].min()

        plant_max_y = plant_indices[0].max()
        plant_min_y = plant_indices[0].min()

        plant_center_x = int((plant_max_x + plant_min_x)/2)
        plant_center_y = int((plant_max_y + plant_min_y)/2)

        plant_width_x = plant_max_x - plant_min_x
        plant_height_y = plant_max_y - plant_min_y

        # import pdb; pdb.set_trace();
        out_file.write(f"{plant_class} {plant_center_x/img_width_x} {plant_center_y/img_height_y} {plant_width_x/img_width_x} {plant_height_y/img_height_y}\n")


    out_file.close()


if __name__ == '__main__':
    sem_dir = "/plant/dataset/challenge/path/test/semantics/"
    id_dir = "/plant/dataset/challenge/path/test/plant_instances/"
    output_dir = "/plant/dataset/challenge/path/yolo_annos/test/labels"

    for img_name in os.listdir(sem_dir):
        sem_path = os.path.join(sem_dir, img_name)
        id_path = os.path.join(id_dir, img_name)
        one_anno(sem_path, id_path, output_dir)
