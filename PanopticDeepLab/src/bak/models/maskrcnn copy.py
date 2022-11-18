import torchvision
import torch
import os
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

class MaskRCNN(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.n_classes = cfg['train']['n_classes']
        self.epochs = cfg['train']['max_epoch'] 
        self.batch_size = cfg['train']['batch_size']

        self.iou = JaccardIndex(num_classes=self.n_classes, reduction='none', ignore_index=0)
        self.ap = MeanAveragePrecision(box_format='xyxy', num_classes=self.n_classes, reduction='none')
        self.ap_ins = MeanAveragePrecision(box_format='xyxy', num_classes=self.n_classes, reduction='none', iou_type='segm')

        self.network = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, progress=True, num_classes = self.n_classes)
        self.network = self.network.float().cuda()

        self.ckpt_dir, self.tboard_dir = self.set_up_logging_directories(cfg)
        self.writer = SummaryWriter(log_dir=self.tboard_dir)
            
    def forward(self, batch):
        out = self.network(batch['image'], batch['targets'])
        return out 

    def getLoss(self, out):
        loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_mask'] + out['loss_objectness'] + out['loss_rpn_box_reg']
        return loss

    def training_step(self, batch):
        out = self.forward(batch)
        loss = self.getLoss(out)
        return loss

    def validation_step(self, batch):
        out = self.network(batch['image'])
        batch, out = self.to_cpu(batch, out)

        self.ap.update(out, batch['targets'])
        self.ap_ins.update(out, batch['targets'])

        img = batch['image'][0]
        _, h, w = img.shape
        b = len(batch['targets'])
        semantic_labels = torch.zeros((b,h,w))
        semantic_predictions = torch.zeros((b,h,w))

        for b_idx in range(b):
            sem_gt = batch['targets'][b_idx]['labels'].unsqueeze(dim=1).unsqueeze(dim=1)*batch['targets'][b_idx]['masks']
            sem_gt,_ = sem_gt.max(dim=0)

            sem_out = out[b_idx]['labels'].unsqueeze(dim=1).unsqueeze(dim=1)*out[b_idx]['masks']
            sem_out,_ = sem_out.max(dim=0)

            ins_out = torch.arange(out[b_idx]['masks'].shape[0]).unsqueeze(dim=1).unsqueeze(dim=1)*out[b_idx]['masks']
            ins_out,_ = ins_out.max(dim=0)

            semantic_labels[b_idx,:,:] = sem_gt
            semantic_predictions[b_idx,:,:] = sem_out
            
        self.iou.update(semantic_predictions.long(), semantic_labels.long())

        out_processed = {}
        out_processed['instances'] = ins_out
        out_processed['semantic'] = sem_out

        return out_processed
        
    def compute_metrics(self):
        return self.ap.compute(), self.ap_ins.compute(), self.iou.compute()

    @staticmethod
    def to_cpu(input, output):
        for x in output:
            x['boxes'] = x['boxes'].cpu()
            x['scores'] = x['scores'].cpu()
            x['labels'] = x['labels'].cpu()
            x['masks'] = x['masks'].to(torch.uint8).squeeze().cpu()

        for x in input['image']:
            x = x.cpu()

        for x in input['targets']:
            x['boxes'] = x['boxes'].cpu()
            x['labels'] = x['labels'].cpu()
            x['masks'] = x['masks'].cpu() 

        return input, output

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