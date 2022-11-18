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
def main(config):
    cfg = yaml.safe_load(open(config))

    train_dataset = Leaves(datapath=cfg['data']['train'], overfit=cfg['train']['overfit'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False)
    if cfg['train']['overfit']:
        val_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False)
    else:
        val_dataset = Leaves(datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
        val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False)

    model = models.get_model(cfg)
    optim = torch.optim.AdamW(model.network.parameters(), lr=cfg['train']['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

    best_map_det = 0
    best_map_ins = 0
    best_iou = 0
    best_pq = 0

    with torch.autograd.set_detect_anomaly(True):
        n_iter = 0  # used for tensorboard
        for e in range(cfg['train']['max_epoch']):
            model.network.train()
            for idx, item in enumerate(iter(train_loader)):
                # import matplotlib.pyplot as plt
                # import cv2
                # masks = item['targets'][0]['masks']
                # labels = item['targets'][0]['labels']
                # bbox = item['targets'][0]['boxes'].long()
                # img = item['image'][0].cpu().permute(1, 2, 0).numpy()
                # import ipdb;ipdb.set_trace()
                # for i in range(masks.shape[0]):
                #     img = cv2.rectangle(
                #         img, (bbox[i][0].item(), bbox[i][1].item()), (bbox[i][2].item(), bbox[i][3].item()), (255, 0, 0), 3)
                #     plt.imshow(img)
                #     plt.show()
                optim.zero_grad()
                loss = model.training_step(item)
                loss.backward()
                optim.step()
                # TODO: Scheduler

                print('Epoch: {}/{} -- Step: {}/{} -- Loss: {} -- Lr: {}'.format(
                    e, cfg['train']['max_epoch'], idx*cfg['train']['batch_size'], len(
                        train_dataset), loss.item(),  scheduler.get_lr()[0]
                ))
                model.writer.add_scalar('Loss/Train/', loss.item(), n_iter)
                n_iter += 1
            
            scheduler.step()
            name = os.path.join(cfg['checkpoint'],'last.pt')
            save_model(model, e, optim, name)
            
            if e <= 498: continue
            model.network.eval()
            for idx, item in enumerate(iter(val_loader)):
                with torch.no_grad():
                    model.validation_step(item)

            ap_detection = model.compute_metrics()
            model.writer.add_scalar('Metric/Val/mAP_detection', ap_detection['map'].item(), n_iter)

            # checking improvements on validation set
            if ap_detection['map'].item() >= best_map_det:
                name = os.path.join(cfg['checkpoint'],'best_detection_map.pt')
                save_model(model, e, optim, name)
        
if __name__ == "__main__":
    main()
