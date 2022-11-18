import torchvision
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class FasterRCNN(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.n_classes = cfg['train']['n_classes']
        self.epochs = cfg['train']['max_epoch'] 
        self.batch_size = cfg['train']['batch_size']

        self.iou = JaccardIndex(num_classes=self.n_classes, reduction='none').cuda()
        self.ap = MeanAveragePrecision(box_format='xyxy', num_classes=self.n_classes, reduction='none').cuda()
        self.network = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, progress=True, num_classes = self.n_classes)
        self.network = self.network.float()
            
    def forward(self, input):
        out = self.network(input['image'], input['targets'])
        return out 

    def getLoss(self, out):
        loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_objectness'] + out['loss_rpn_box_reg']
        return loss

    def training_step(self, batch):
        out = self.forward(batch)
        loss = self.getLoss(out)
        return loss

    def validation_step(self, batch):
        out = self.forward(batch)
        return out
