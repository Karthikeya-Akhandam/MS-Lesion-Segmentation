import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm
from .model import get_model
from .loss import MSLesionLoss
from .data_loader import get_dataloader

def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        images, labels = batch["image"].to(device), batch["label"].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate_epoch(model, loader, dice_metric, hd_metric, device, roi_size=(96, 96, 96)):
    model.eval()
    dice_metric.reset()
    hd_metric.reset()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            # Sliding window inference for full volume
            val_outputs = sliding_window_inference(images, roi_size, 4, model)
            val_outputs = (val_outputs.sigmoid() > 0.5).float()
            
            dice_metric(y_pred=val_outputs, y=labels)
            hd_metric(y_pred=val_outputs, y=labels)
            
    return dice_metric.aggregate().item(), hd_metric.aggregate().item()

def run_training(train_files, val_files, epochs=100, out_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Initialize Model, Loss, Optimizer
    model = get_model(in_channels=1, out_channels=1).to(device)
    loss_fn = MSLesionLoss(alpha=1.0, weight=10.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    # 2. Data Loaders
    train_loader = get_dataloader(train_files, batch_size=1, patch_size=(96, 96, 96), num_samples=4)
    val_loader = get_dataloader(val_files, batch_size=1, patch_size=(96, 96, 96), num_samples=1)
    
    # 3. Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    
    best_dice = 0
    for epoch in range(epochs):
        print(f"
Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_dice, val_hd = validate_epoch(model, val_loader, dice_metric, hd_metric, device)
        scheduler.step()
        
        print(f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val HD95: {val_hd:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            print("Model saved (best Dice score)")
            
if __name__ == "__main__":
    pass
