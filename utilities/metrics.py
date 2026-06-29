
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

#MIOU

def mIoU(pred_mask, mask, num_classes=7, smooth=1e-5, include_background=False):
    """
    Compute mean IoU score across classes for multi-class segmentation.
    
    Args:
        pred_mask (torch.Tensor): Predicted logits/probabilities, shape (N, C, H, W)
        mask (torch.Tensor): Ground truth, shape (N, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        n_classes (int): Number of classes (e.g., 4)
        include_background (bool): Whether to include class 0 in the metric
    
    Returns:
        float: Mean IoU score across classes (NaN values ignored)
    """
    with torch.no_grad():
        # Convert logits to class indices
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)  # Shape: (N, H, W)
        
        # Flatten tensors
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        
        # Set starting class based on background inclusion
        start_class = 0 if include_background else 1
        
        iou_per_class = []
        for clas in range(start_class, num_classes):
            true_class = (pred_mask == clas).float()  # Binary mask for predicted class
            true_label = (mask == clas).float()       # Binary mask for ground truth
            
            if true_label.sum() == 0:  # No instances of this class in ground truth
                iou_per_class.append(np.nan)
            else:
                intersect = (true_class * true_label).sum().item()
                union = (true_class + true_label - true_class * true_label).sum().item()
                # Note: union = sum(pred) + sum(true) - intersect
                
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        
        return np.nanmean(iou_per_class)

# DICE SCORE

def mDice(pred_mask, mask, num_classes=7, smooth=1e-5, include_background=False):
    """
    Compute mean Dice score across classes for multi-class segmentation.
    
    Args:
        pred_mask (torch.Tensor): Predicted logits/probabilities, shape (N, C, H, W)
        mask (torch.Tensor): Ground truth, shape (N, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        n_classes (int): Number of classes (e.g., 4)
        include_background (bool): Whether to include class 0 in the metric
    
    Returns:
        float: Mean Dice score across classes (NaN values ignored)
    """
    with torch.no_grad():
        # Convert logits to class indices
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)  # Shape: (N, H, W)
        
        # Flatten tensors
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        
        # Set starting class based on background inclusion
        start_class = 0 if include_background else 1
        
        dice_per_class = []
        for clas in range(start_class, num_classes):
            true_class = (pred_mask == clas).float()  # Binary mask for predicted class
            true_label = (mask == clas).float()       # Binary mask for ground truth
            
            if true_label.sum() == 0:  # No instances of this class in ground truth
                dice_per_class.append(np.nan)
            else:
                intersect = (true_class * true_label).sum().item()
                sum_pred = true_class.sum().item()
                sum_true = true_label.sum().item()
                
                dice = 2 * (intersect + smooth) / (sum_pred + sum_true + smooth)
                dice_per_class.append(dice)
        
        return np.nanmean(dice_per_class)

