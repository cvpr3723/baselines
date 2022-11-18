import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Plants, collate_pdc
import models
import yaml
import torchvision


def save_model(model, epoch, optim, name):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        },
        name,
    )


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "configs/cfg.yaml"),
)
def main(config):
    cfg = yaml.safe_load(open(config))

    # TODO: transformation, especially cropping
    train_dataset = Plants(
        datapath=cfg["data"]["train"], overfit=cfg["train"]["overfit"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        collate_fn=collate_pdc,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = Plants(
        datapath=cfg["data"]["val"], overfit=cfg["train"]["overfit"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        collate_fn=collate_pdc,
        shuffle=True,
        drop_last=True,
    )

    model = models.get_model(cfg)
    optim = torch.optim.AdamW(model.network.parameters(), lr=1e-5)

    best_map_det = 0
    best_map_ins = 0
    best_iou = 0
    best_pq = 0

    n_iter = 0  # used for tensorboard
    for e in range(cfg["train"]["max_epoch"]):
        model.network.train()
        for idx, item in enumerate(iter(train_loader)):
            optim.zero_grad()
            loss = model.training_step(item)
            loss.backward()
            optim.step()
            # TODO: Scheduler

            print(
                "Epoch: {}/{} -- Step: {}/{} -- Loss: {}".format(
                    e,
                    cfg["train"]["max_epoch"],
                    idx * cfg["train"]["batch_size"],
                    len(train_dataset),
                    loss.item(),
                )
            )
            model.writer.add_scalar("Loss/Train/", loss.item(), n_iter)
            n_iter += 1

        model.network.eval()
        for idx, item in enumerate(iter(val_loader)):
            with torch.no_grad():
                pred = model.validation_step(item)

        ap_detection, ap_instance, iou = model.compute_metrics()
        model.writer.add_scalar(
            "Metric/Val/mAP_detection", ap_detection["map"].item(), n_iter
        )
        model.writer.add_scalar(
            "Metric/Val/mAP_instance", ap_instance["map"].item(), n_iter
        )
        model.writer.add_scalar("Metric/Val/mIoU", iou.item(), n_iter)

        sem_images = torchvision.utils.make_grid(pred["semantic"])
        ins_images = torchvision.utils.make_grid(pred["instances"])
        model.writer.add_image("Images/Instance", ins_images, n_iter)
        model.writer.add_image("Images/Semantic", sem_images, n_iter)

        # checking improvements on validation set
        if ap_detection["map"].item() >= best_map_det:
            name = os.path.join(cfg["checkpoint"], "best_detection_map.pt")
            save_model(model, e, optim, name)
        if ap_instance["map"].item() >= best_map_ins:
            name = os.path.join(cfg["checkpoint"], "best_instance_map.pt")
            save_model(model, e, optim, name)
        if iou.item() >= best_iou:
            name = os.path.join(cfg["checkpoint"], "best_miou.pt")
            save_model(model, e, optim, name)


if __name__ == "__main__":
    main()
