from cProfile import label
import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Leaves, collate_pdc
import models
import yaml


def save_model(model, epoch, optim, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, name)

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/cfg.yaml'))
@click.option('--ckpt_file',
              '-w',
              type=str,
              help='path to trained weights (.pt)',
              default=join(dirname(abspath(__file__)),'checkpoints/best.pt'))
@click.option('--out',
              '-o',
              type=str,
              help='output directory',
              default=join(dirname(abspath(__file__)),'results/'))
def main(config, ckpt_file, out):
    cfg = yaml.safe_load(open(config))

    val_dataset = Leaves(datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    weights = torch.load(ckpt_file)["model_state_dict"]
    model.load_state_dict(weights)
    
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import copy

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()
        for idx, item in enumerate(iter(val_loader)):
            with torch.no_grad():
                size = item['image'][0].shape[1]
                semantic, instance, predictions = model.test_step(item)

                res_names = item['name']
                for i in range(len(res_names)):
                    fname_ins = os.path.join(out,'predictions/leaf_instances/',res_names[i])
                    fname_sem = os.path.join(out,'predictions/semantics/',res_names[i])
                    fname_box = os.path.join(out,'leaf_bboxes',res_names[i].replace('png','txt'))

                    cv2.imwrite(fname_sem, semantic[i].cpu().long().numpy())
                    cv2.imwrite(fname_ins,instance[i].cpu().long().numpy())

                    size = item['image'][i].shape[1]
                    scores = predictions[i]['scores'].cpu().numpy()
                    labels = predictions[i]['labels'].cpu().numpy()
                    # converting boxes to center, width, height format
                    boxes_ = predictions[i]['boxes'].cpu().numpy()
                    num_pred = len(boxes_)
                    cx = (boxes_[:,2] + boxes_[:,0])/2
                    cy = (boxes_[:,3] + boxes_[:,1])/2
                    bw = boxes_[:,2] - boxes_[:,0]
                    bh = boxes_[:,3] - boxes_[:,1]

                    # ready to be saveds
                    pred_cls_box_score = np.hstack((labels.reshape(num_pred,1), 
                                         cx.reshape(num_pred,1)/size,
                                         cy.reshape(num_pred,1)/size,
                                         bw.reshape(num_pred,1)/size,
                                         bh.reshape(num_pred,1)/size,
                                         scores.reshape(num_pred,1)
                                        ))
                    np.savetxt(fname_box, pred_cls_box_score, fmt='%f')


                if False:
                    imgs = item['image']
                    
                    for b_idx in range(len(predictions)):

                        img = (imgs[b_idx]*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
                        imb = copy.deepcopy(img)
                        sem = semantic[b_idx]
                        ins = instance[b_idx]

                        dic = predictions[b_idx]
                        scores = dic['scores']
                        labels = dic['labels']
                        boxes = dic['boxes']

                        for i in range(boxes.shape[0]):
                            if labels[i] == 0:
                                continue
                            elif labels[i] == 1:
                                color = (0,int(255*scores[i].item()),0)
                            else:
                                color = (int(255*scores[i].item()),0,0)

                            imb = cv2.rectangle(imb, (int(boxes[i][0].item()), int(boxes[i][1].item())), (int(boxes[i][2].item()), int(boxes[i][3].item())), color, 3)

                        fig, axs = plt.subplots(1,4)
                        axs[0].imshow(img)
                        axs[1].imshow(sem)
                        axs[2].imshow(ins)
                        axs[3].imshow(imb)
                        plt.show()

    #     ap_detection, ap_instance, iou = model.compute_metrics()

    # print(ap_detection)

if __name__ == "__main__":
    main()
