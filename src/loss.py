import torch
import torch.nn as nn
from monai.losses import DiceLoss

class MSLesionLoss(nn.Module):
    def __init__(self, alpha=1.0, weight=10.0):
        super(MSLesionLoss, self).__init__()
        self.alpha = alpha
        # Stabilized DiceLoss with squared_pred and sigmoid
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True)
        # Weighted Binary Cross Entropy
        # pos_weight=10.0 penalizes false negatives more heavily (Limitation 3)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

    def forward(self, input, target):
        dice = self.dice_loss(input, target)
        bce = self.bce_loss(input, target)
        
        return dice + self.alpha * bce

if __name__ == "__main__":
    # Test loss computation
    loss_fn = MSLesionLoss()
    pred = torch.randn(1, 1, 64, 64, 64)
    target = torch.randint(0, 2, (1, 1, 64, 64, 64)).float()
    loss = loss_fn(pred, target)
    print(f"Computed MS Lesion Loss: {loss.item():.4f}")
