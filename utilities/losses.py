import torch
import torch.nn as nn
import torch.nn.functional as F

#Feature Similarity Loss
class feature_sim(nn.Module):
	'''
	Input similarity-loss between outputs of same minibatch
	'''
	def __init__(self):
		super(feature_sim, self).__init__()

	def forward(self, f_1, f_2):
		f_1 = f_1.view(f_1.size(0), -1)
		G_1  = torch.mm(f_1, f_1.t())
		norm_G_1 = F.normalize(G_1, p=2, dim=1)

		f_2 = f_2.view(f_2.size(0), -1)
		G_2  = torch.mm(f_2, f_2.t())
		norm_G_2 = F.normalize(G_2, p=2, dim=1)

		loss = F.mse_loss(norm_G_1, norm_G_2)

		return loss

class FeatureSimilarityLoss(nn.Module):
    """
    Feature Similarity Loss between outputs of the same minibatch.

    This loss encourages similarity between two feature representations by comparing their
    normalized Gram matrices using Mean Squared Error (MSE). It can be used to enforce
    consistency between features from different branches or augmentations of the same input.

    Args:
        None (no additional parameters required).

    Inputs:
        f_1 (torch.Tensor): First feature tensor, shape [B, C, H, W] or [B, D].
        f_2 (torch.Tensor): Second feature tensor, shape [B, C, H, W] or [B, D].

    Returns:
        torch.Tensor: Scalar MSE loss between normalized Gram matrices of f_1 and f_2.

    Shape:
        - Input: Both f_1 and f_2 can be [B, C, H, W] (convolutional features) or [B, D] (flattened).
        - Output: Scalar loss value.
    """
    def __init__(self):
        super(FeatureSimilarityLoss, self).__init__()

    def forward(self, f_1: torch.Tensor, f_2: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions if present (e.g., [B, C, H, W] -> [B, C*H*W])
        f_1 = f_1.reshape(f_1.size(0), -1)  # Use reshape instead of view
        f_2 = f_2.reshape(f_2.size(0), -1)  # Use reshape instead of view
        
        # Compute Gram matrices (batch-wise feature correlations)
        gram_1 = torch.mm(f_1, f_1.t())  # [B, B]
        gram_2 = torch.mm(f_2, f_2.t())  # [B, B]
        
        # Normalize Gram matrices along rows (L2 norm)
        norm_gram_1 = F.normalize(gram_1, p=2, dim=1)
        norm_gram_2 = F.normalize(gram_2, p=2, dim=1)
        
        # Compute MSE loss between normalized Gram matrices
        loss = F.mse_loss(norm_gram_1, norm_gram_2)
        
        return loss

    # Example usage
# if __name__ == "__main__":
#     # Dummy feature tensors
#     batch_size, channels, height, width = 4, 16, 32, 32
#     f_1 = torch.randn(batch_size, channels, height, width)
#     f_2 = torch.randn(batch_size, channels, height, width)
    
#     # Initialize loss
#     criterion = FeatureSimilarityLoss()
    
#     # Compute loss
#     loss = criterion(f_1, f_2)
#     print(f"Feature Similarity Loss: {loss.item()}")
    
# #Combined Dice and Cross-Entropy Loss
class DiceCELoss(nn.Module):
    """
    Combined Dice and Cross-Entropy Loss for multi-class segmentation.

    This loss combines Dice Loss (overlap-based) and Cross-Entropy Loss (probability-based)
    to optimize segmentation models. The Dice component focuses on spatial overlap, while
    CE enforces class probability alignment. Suitable for imbalanced datasets.

    Args:
        num_classes (int): Number of classes (e.g., 4 for your task).
        dice_weight (float, optional): Weight for Dice Loss (default: 0.5).
        ce_weight (float, optional): Weight for CE Loss (default: 0.5).
        smooth (float, optional): Smoothing factor for Dice Loss (default: 1e-5).
        ignore_background (bool, optional): If True, excludes class 0 from Dice Loss (default: False).

    Inputs:
        logits (torch.Tensor): Model predictions, shape [B, C, H, W], raw logits.
        targets (torch.Tensor): Ground truth, shape [B, H, W], class indices [0, C-1].

    Returns:
        torch.Tensor: Combined loss (scalar).
    """
    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5, smooth=1e-5, ignore_background=True):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, logits, targets):
        """Compute Dice Loss."""
        # Convert logits to probabilities
        probas = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Match data types
        targets_one_hot = targets_one_hot.type(logits.type())
        
        # Define dimensions to sum over (batch, height, width)
        dims = (0, 2, 3)
        
        # Optionally exclude background (class 0)
        start_idx = 1 if self.ignore_background else 0
        probas = probas[:, start_idx:]
        targets_one_hot = targets_one_hot[:, start_idx:]
        
        # Compute Dice
        intersection = torch.sum(probas * targets_one_hot, dims)
        cardinality = torch.sum(probas + targets_one_hot, dims)
        dice_coeff = 2. * intersection / (cardinality + self.smooth)
        
        return 1 - dice_coeff.mean()

    def forward(self, logits, targets):
        """Compute combined Dice and CE Loss."""
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


import torch
import torch.nn as nn

class BoundaryAwareBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        """
        Args:
            pos_weight: Weight for boundary (positive) pixels.
                        Background pixels get a weight of 1.
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCELoss(reduction='none')  # expects probabilities

    def forward(self, pred, target):
        """
        pred: (B, 1, W, H) - sigmoid output (probabilities)
        target: (B, W, H) - binary ground truth
        """
        # Ensure target shape matches pred
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, 1, W, H)
        # Compute element-wise BCE loss
        loss = self.bce(pred, target.float())
        # Create a weight map: pos_weight for boundary, 1 for background
        weight = torch.ones_like(target)
        weight[target == 1] = self.pos_weight
        # Apply weights
        weighted_loss = loss * weight
        return weighted_loss.mean()
