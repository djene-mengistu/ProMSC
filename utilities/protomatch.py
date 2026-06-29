import torch
import torch.nn.functional as F
from typing import Optional, Literal

##consistency loss between weak and strong augmentations

def consistency_loss(weak_output, strong_output, temperature=1.0, loss_type='mse', is_binary=False):
    """
    Enhanced consistency loss with multiple distance metrics.
    
    Args:
        weak_output (torch.Tensor):   Weak augmentation logits
        strong_output (torch.Tensor): Strong augmentation logits  
        temperature (float):          Temperature for sharpening
        is_binary (bool):             Whether binary segmentation
        loss_type (str):             'mse', 'kl', or 'cosine'
    
    Returns:
        torch.Tensor: Consistency loss
    """
    # Apply temperature scaling
    if is_binary:
        weak_probs = torch.sigmoid(weak_output / temperature)
        strong_probs = torch.sigmoid(strong_output / temperature)
    else:
        weak_probs = F.softmax(weak_output / temperature, dim=1)
        strong_probs = F.softmax(strong_output / temperature, dim=1)
    
    # Detach weak predictions (teacher)
    weak_probs = weak_probs.detach()
    
    # Select loss type
    if loss_type == 'mse':
        loss = F.mse_loss(strong_probs, weak_probs, reduction='mean')
    
    elif loss_type == 'kl':
        # KL divergence: D_KL(strong || weak)
        loss = F.kl_div(
            torch.log(strong_probs + 1e-8), 
            weak_probs, 
            reduction='batchmean'
        )
    
    elif loss_type == 'cosine':
        # Flatten spatial dimensions for cosine similarity
        B, C, H, W = strong_probs.shape
        strong_flat = strong_probs.view(B, C, -1)
        weak_flat = weak_probs.view(B, C, -1)
        
        # Compute cosine similarity per class and average
        cos_sim = F.cosine_similarity(strong_flat, weak_flat, dim=2)
        loss = (1 - cos_sim.mean())
    
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    return loss

#Computing prototypes and prototype loss
def compute_prototypes(features: torch.Tensor, 
                      labels: torch.Tensor, 
                      num_classes: int, 
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute class prototypes from feature maps and labels.
    
    Prototypes are computed as the mean feature vector for each class across
    all spatial locations and batch samples where the class is present.
    
    Args:
        features: Input features of shape [B, C, H, W]
        labels: Ground truth labels of shape [B, H, W] with values in [0, num_classes-1]
        num_classes: Number of classes
        mask: Optional binary mask of shape [B, H, W] to exclude certain regions
              (e.g., uncertain areas, padding). 1 = include, 0 = exclude.
    
    Returns:
        prototypes: Tensor of shape [num_classes, C] containing mean feature vectors
                    for each class. Zero vector for classes with no samples.
    
    Example:
        >>> features = torch.randn(2, 512, 32, 32)
        >>> labels = torch.randint(0, 2, (2, 32, 32))
        >>> prototypes = compute_prototypes(features, labels, 2)
        >>> prototypes.shape
        torch.Size([2, 512])
    """
    B, C, H, W = features.shape
    prototypes = []
    
    # Flatten spatial dimensions for efficient computation
    features_flat = features.view(B, C, -1)  # [B, C, H*W]
    labels_flat = labels.view(B, -1)         # [B, H*W]
    
    if mask is not None:
        mask_flat = mask.view(B, -1)         # [B, H*W]
    else:
        mask_flat = torch.ones_like(labels_flat, dtype=torch.bool)
    
    for c in range(num_classes):
        # Create class mask with optional additional mask
        class_mask = (labels_flat == c) & mask_flat
        
        if class_mask.sum() > 0:
            # Use advanced indexing for efficiency
            valid_features = features_flat.permute(1, 0, 2)[:, class_mask]  # [C, num_valid_pixels]
            proto = valid_features.mean(dim=1)
            prototypes.append(proto)
        else:
            prototypes.append(torch.zeros(C, device=features.device))
    
    return torch.stack(prototypes, dim=0)  # [num_classes, C]


def prototype_loss(prototypes_l: torch.Tensor, 
                  prototypes_u: torch.Tensor, 
                  loss_type: Literal['l2', 'l2_normalized', 'cosine'] = 'l2',
                  eps: float = 1e-8) -> torch.Tensor:
    """
    Compute consistency loss between labeled and unlabeled prototypes.
    
    This loss encourages feature consistency between prototypes computed from
    labeled and unlabeled data in semi-supervised learning.
    
    Args:
        prototypes_l: Prototypes from labeled data of shape [num_classes, C]
        prototypes_u: Prototypes from unlabeled data of shape [num_classes, C]
        loss_type: Type of distance metric to use:
            - 'l2': Euclidean distance (L2 norm)
            - 'l2_normalized': L2 distance between normalized prototypes
            - 'cosine': Cosine distance (1 - cosine similarity)
        eps: Small value for numerical stability
    
    Returns:
        loss: Scalar tensor representing the mean consistency loss across valid classes
    
    Example:
        >>> prototypes_l = torch.randn(2, 512)
        >>> prototypes_u = torch.randn(2, 512)
        >>> loss = prototype_loss(prototypes_l, prototypes_u, 'cosine')
        >>> loss.shape
        torch.Size([])
    """
    # Create validity mask
    valid_mask = (
        torch.isfinite(prototypes_l).all(dim=1) & 
        torch.isfinite(prototypes_u).all(dim=1) &
        (prototypes_l.norm(dim=1) > eps) & 
        (prototypes_u.norm(dim=1) > eps)
    )
    
    if not valid_mask.any():
        return torch.tensor(0.0, device=prototypes_l.device)
    
    valid_proto_l = prototypes_l[valid_mask]
    valid_proto_u = prototypes_u[valid_mask]
    
    if loss_type == 'l2':
        distances = torch.norm(valid_proto_l - valid_proto_u, p=2, dim=1)
        loss = distances.mean()
    elif loss_type == 'l2_normalized':
        proto_l_norm = F.normalize(valid_proto_l, p=2, dim=1)
        proto_u_norm = F.normalize(valid_proto_u, p=2, dim=1)
        distances = torch.norm(proto_l_norm - proto_u_norm, p=2, dim=1)
        loss = distances.mean()
    elif loss_type == 'cosine':
        cos_sim = F.cosine_similarity(valid_proto_l, valid_proto_u, dim=1)
        loss = (1 - cos_sim).mean()
    else:
        raise ValueError(f"Unsupported loss_type {loss_type}")
    
    return loss


