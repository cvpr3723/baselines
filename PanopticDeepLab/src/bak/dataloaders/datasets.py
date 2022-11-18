import torch
import os
from PIL import Image, ImageFile
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset 
import cv2
import copy


def collate_pdc(items):
    batch = {}
    images = []
    targets = []
    for i in range(len(items)):
        images.append(items[i]['image'])
        targets.append(items[i]['targets'])
    batch['image'] = list(images)
    batch['targets'] = list(targets)
    return batch


class Plants(Dataset):
    def __init__(self, datapath, overfit=False):
        super().__init__()
        
        self.datapath = datapath
        self.overfit = overfit 
        self.annotations_path = os.path.join(self.datapath, 'semantics')

        self.annotations_list =[os.path.join(self.annotations_path,x) for x in os.listdir(self.annotations_path) if ".png" in x]
        self.images_list =[x.replace('/semantics', '') for x in self.annotations_list]

        self.annotations_list.sort()
        self.images_list.sort()
            
        self.len = len(self.images_list)
      

    def __getitem__(self, index):
        
        image_path = self.images_list[index]
        label_path = self.annotations_list[index]

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        png_annotation = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)[:, :, :2].astype(np.uint32)

        raw_instance = png_annotation[:, :, 1]
        raw_semantic = png_annotation[:, :, 0]

        crop_instances = copy.deepcopy(raw_instance)
        crop_instances[raw_semantic != 1] = 0
        raw_instance[raw_semantic == 2] += crop_instances.max()

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        instance_ids = np.unique(raw_instance)[1:] # no background
        instances_successive =  np.zeros_like(raw_instance)
        
        for idx, id_ in enumerate(instance_ids):
            instance_mask = raw_instance == id_
            instances_successive[instance_mask] = idx + 1
        instances = instances_successive

        assert np.max(instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'

        # instances = Image.fromarray(np.uint8(instances))
        # if(self.transform is not None):
        #     sample = self.transform(sample)

        instances_torch = torch.from_numpy(np.int64(instances))
        masks = F.one_hot(instances_torch).permute(2,0,1)
        
        class_vector = torch.zeros_like(torch.unique(torch.from_numpy(np.int64(raw_instance))))
        class_vector[1:torch.unique(torch.from_numpy(np.int64(crop_instances))).max()+1] = 1
        class_vector[torch.unique(torch.from_numpy(np.int64(crop_instances))).max()+1:] = 2
        boxes = self.masks_to_boxes(masks)
        
        image = torch.from_numpy(image/255).permute(2,0,1).float()

        maskrcnn_input = {}
        maskrcnn_input['image'] = image.cuda()
        maskrcnn_input['targets'] = {}
        maskrcnn_input['targets']['masks'] = masks.to(torch.uint8).cuda()
        maskrcnn_input['targets']['labels'] = class_vector.cuda()
        maskrcnn_input['targets']['boxes'] = boxes.cuda()

        return maskrcnn_input
       
    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]

        bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            y, x = torch.where(mask != 0)

            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

        return bounding_boxes

    def __len__(self):
        if self.overfit: 
            return self.overfit
        return self.len

