"""Main training script for DroneNet-Pico."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import yaml
from tqdm import tqdm

from models import DroneNetPico, DetectionLoss
from dataset import DroneDataset, get_train_transforms, get_val_transforms
from dataset.drone_dataset import collate_fn
from utils.ema import ModelEMA


def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if cfg["output"].get("tensorboard") and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    # Model
    model = DroneNetPico(
        num_classes=cfg["model"]["num_classes"],
        input_size=cfg["model"]["input_size"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"DroneNet-Pico: {param_count:,} parameters")

    # Dataset
    train_transforms = get_train_transforms()
    train_dataset = DroneDataset(
        cfg["data"]["train_images"],
        cfg["data"]["train_labels"],
        input_size=cfg["model"]["input_size"],
        transforms=train_transforms,
        mosaic_prob=cfg["augmentation"]["mosaic_prob"],
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dataset = DroneDataset(
        cfg["data"]["val_images"],
        cfg["data"]["val_labels"],
        input_size=cfg["model"]["input_size"],
        transforms=get_val_transforms(),
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["base_lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"]["min_lr"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["fp16"] and device.type == "cuda")
    criterion = DetectionLoss(
        num_classes=cfg["model"]["num_classes"],
        obj_weight=cfg["loss"]["obj_weight"],
        cls_weight=cfg["loss"]["cls_weight"],
        reg_weight=cfg["loss"]["reg_weight"],
    )
    ema = ModelEMA(model, decay=cfg["training"]["ema_decay"])

    # Warmup
    warmup_epochs = cfg["training"]["warmup_epochs"]

    best_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['training']['epochs']}")

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Warmup LR
            if epoch < warmup_epochs:
                warmup_factor = (epoch * len(train_loader) + n_batches) / (warmup_epochs * len(train_loader))
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg["training"]["base_lr"] * warmup_factor

            with torch.amp.autocast("cuda", enabled=cfg["training"]["fp16"] and device.type == "cuda"):
                predictions = model(images)
                loss, loss_dict = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "obj": f"{loss_dict['obj_loss']:.3f}",
                "cls": f"{loss_dict['cls_loss']:.3f}",
                "reg": f"{loss_dict['reg_loss']:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        if writer:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_interval"] == 0 or avg_loss < best_loss:
            checkpoint = {
                "epoch": epoch,
                "model": ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }

            torch.save(checkpoint, save_dir / "last.pt")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, save_dir / "best.pt")
                print(f"  New best: {best_loss:.4f}")

    if writer:
        writer.close()

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Models saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)
