import torchvision
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
import torch
import os
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
import torchvision.ops as tops

class FasterRCNN(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.n_classes = cfg['train']['n_classes']
        self.epochs = cfg['train']['max_epoch'] 
        self.batch_size = cfg['train']['batch_size']

        self.ap = MeanAveragePrecision(box_format='xyxy', num_classes=self.n_classes, reduction='none')
        self.network = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=None, progress=True, num_classes = self.n_classes)
        self.network = self.network.float().cuda()

        self.prob_th = cfg['val']['prob_th']
        self.overlapping_th = cfg['val']['nms_th']

        self.ckpt_dir, self.tboard_dir = self.set_up_logging_directories(cfg)
        self.writer = SummaryWriter(log_dir=self.tboard_dir)
            
    def forward(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'], batch['targets'])
        return out 

    def getLoss(self, out):
        loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_objectness'] + out['loss_rpn_box_reg']
        return loss

    def training_step(self, batch):
        out = self.forward(batch)
        loss = self.getLoss(out)
        return loss

    def validation_step(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'])

        # here start the postprocessing 
        b = len(batch['targets'])

        for b_idx in range(b):

            scores = out[b_idx]['scores']
            boxes = out[b_idx]['boxes']
            labels = out[b_idx]['labels']

            # non maximum suppression
            refined = tops.nms(boxes, scores, self.overlapping_th)
            refined_boxes = boxes[refined]
            refined_scores = scores[refined]
            refined_labels = labels[refined]

            # keeping only high scores
            high_scores = refined_scores > self.prob_th

            # if any scores are above self.prob_th we can compute metrics
            if high_scores.sum():
                surviving_boxes = refined_boxes[high_scores]
                surviving_scores = refined_scores[high_scores]
                surviving_labels = refined_labels[high_scores]
                
                surviving_dict = {}
                surviving_dict['boxes'] = surviving_boxes.cuda()
                surviving_dict['labels'] = surviving_labels.cuda()
                surviving_dict['scores'] = surviving_scores.cuda()
            
            # if not populate prediction dict with empty tensor to get 0 for ap and ap_ins
            # define zero sem and ins masks for iou metric
            else:
                surviving_dict = {}
                surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
                surviving_dict['labels'] = torch.empty(0).cuda()
                surviving_dict['scores'] = torch.empty(0).cuda()

            self.ap.update([surviving_dict], [batch['targets'][b_idx]])

    def test_step(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'])

        # here start the postprocessing 
        b = len(batch['targets'])
        predictions_dictionaries = []

        for b_idx in range(b):

            scores = out[b_idx]['scores']
            boxes = out[b_idx]['boxes']
            labels = out[b_idx]['labels']

            # non maximum suppression
            refined = tops.nms(boxes, scores, self.overlapping_th)
            refined_boxes = boxes[refined]
            refined_scores = scores[refined]
            refined_labels = labels[refined]

            # keeping only high scores
            high_scores = refined_scores > self.prob_th

            # if any scores are above self.prob_th we can compute metrics
            if high_scores.sum():
                surviving_boxes = refined_boxes[high_scores]
                surviving_scores = refined_scores[high_scores]
                surviving_labels = refined_labels[high_scores]
                
                surviving_dict = {}
                surviving_dict['boxes'] = surviving_boxes.cuda()
                surviving_dict['labels'] = surviving_labels.cuda()
                surviving_dict['scores'] = surviving_scores.cuda()
            
            # if not populate prediction dict with empty tensor to get 0 for ap and ap_ins
            # define zero sem and ins masks for iou metric
            else:
                surviving_dict = {}
                surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
                surviving_dict['labels'] = torch.empty(0).cuda()
                surviving_dict['scores'] = torch.empty(0).cuda()

            predictions_dictionaries.append(surviving_dict)
        return predictions_dictionaries

    def compute_metrics(self):
        return self.ap.compute()

    @staticmethod
    def set_up_logging_directories(cfg):
        os.makedirs(cfg['checkpoint'], exist_ok = True) 
        os.makedirs(cfg['tensorboard'], exist_ok = True) 

        versions = os.listdir(cfg['checkpoint'])
        versions.sort()

        if len(versions) == 0:
            current_version = 0

        else:
            max_v = 0
            for fname in versions:
                if os.path.isdir(os.path.join(cfg['checkpoint'],fname)):
                    tmp_v = int(fname.split('_')[1])
                    if tmp_v > max_v:
                        max_v = tmp_v

            current_version = max_v  + 1

        new_dir = 'version_{}'.format(current_version)
        ckpt = os.path.join(cfg['checkpoint'],new_dir)
        tboard = os.path.join(cfg['tensorboard'],new_dir)
        os.makedirs(ckpt, exist_ok = True) 
        os.makedirs(tboard, exist_ok = True) 
        return ckpt, tboard