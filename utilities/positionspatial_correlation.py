import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureConsistencyLoss(nn.Module):
    def __init__(self, pos_weight=1.0, spatial_weight=1.0, use_l1=False, 
                 max_spatial_size=14, temperature=0.1, use_attention=True):
        super().__init__()
        self.pos_weight = pos_weight
        self.spatial_weight = spatial_weight
        self.use_l1 = use_l1
        self.max_spatial_size = max_spatial_size
        self.temperature = temperature
        self.use_attention = use_attention
        
    def positional_correlation(self, weak_feats, strong_feats):
        """
        Compute L1 or L2 loss between corresponding feature positions.
        NO NORMALIZATION - absolute values matter for positional consistency.
        """
        if self.use_l1:
            return F.l1_loss(weak_feats, strong_feats)
        else:
            return F.mse_loss(weak_feats, strong_feats)
    
    def compute_cross_correlation(self, weak_feats, strong_feats):
        """
        Compute cross correlation matrix for attention-based logit enhancement.
        NORMALIZATION IS NEEDED HERE for stable correlation calculations.
        """
        b, c, h, w = weak_feats.shape
        
        # Reduce spatial size if too large
        if h > self.max_spatial_size or w > self.max_spatial_size:
            weak_feats = F.adaptive_avg_pool2d(weak_feats, (self.max_spatial_size, self.max_spatial_size))
            strong_feats = F.adaptive_avg_pool2d(strong_feats, (self.max_spatial_size, self.max_spatial_size))
            h = w = self.max_spatial_size
        
        # Flatten and normalize (NEEDED for correlation)
        weak_flat = weak_feats.view(b, c, -1)
        strong_flat = strong_feats.view(b, c, -1)
        
        weak_flat = F.normalize(weak_flat, p=2, dim=1)
        strong_flat = F.normalize(strong_flat, p=2, dim=1)
        
        # Compute cross correlation
        cross_corr = torch.bmm(weak_flat.transpose(1, 2), strong_flat)
        
        # Convert to attention matrix
        if self.use_attention:
            attention_matrix = F.softmax(cross_corr / self.temperature, dim=-1)
            return attention_matrix
        else:
            return cross_corr
    
    def spatial_correlation(self, weak_feats, strong_feats):
        """
        Compute cosine similarity between spatial correlation matrices.
        NORMALIZATION IS NEEDED HERE for meaningful cosine similarity.
        """
        b, c, h, w = weak_feats.shape
        
        if h > self.max_spatial_size or w > self.max_spatial_size:
            weak_feats = F.adaptive_avg_pool2d(weak_feats, (self.max_spatial_size, self.max_spatial_size))
            strong_feats = F.adaptive_avg_pool2d(strong_feats, (self.max_spatial_size, self.max_spatial_size))
            h = w = self.max_spatial_size
        
        weak_flat = weak_feats.view(b, c, -1)
        strong_flat = strong_feats.view(b, c, -1)
        
        # Normalize (NEEDED for correlation consistency)
        weak_flat = F.normalize(weak_flat, p=2, dim=1)
        strong_flat = F.normalize(strong_flat, p=2, dim=1)
        
        weak_corr = torch.bmm(weak_flat.transpose(1, 2), weak_flat)
        strong_corr = torch.bmm(strong_flat.transpose(1, 2), strong_flat)
        
        spatial_loss = 1 - F.cosine_similarity(
            weak_corr.view(b, -1), 
            strong_corr.view(b, -1), 
            dim=1
        ).mean()
        
        attention_matrix = self.compute_cross_correlation(weak_feats, strong_feats)
        
        return spatial_loss, attention_matrix
    
    def forward(self, weak_features, strong_features):
        """
        Compute combined consistency loss for multi-scale features.
        """
        assert len(weak_features) == len(strong_features)
        
        total_pos_loss = 0
        total_spatial_loss = 0
        attention_matrices = []
        
        for weak, strong in zip(weak_features, strong_features):
            weak = weak.detach() #CHECK
            
            # POSITIONAL LOSS: No normalization needed
            pos_loss = self.positional_correlation(weak, strong)
            
            # SPATIAL LOSS: Normalization happens inside spatial_correlation
            spatial_loss, attention_matrix = self.spatial_correlation(weak, strong)
            
            total_spatial_loss += spatial_loss
            total_pos_loss += pos_loss
            attention_matrices.append(attention_matrix)
        
        num_scales = len(weak_features)
        total_pos_loss /= num_scales
        total_spatial_loss /= num_scales
        
        comb_loss = self.pos_weight * total_pos_loss + self.spatial_weight * total_spatial_loss
        
        return comb_loss, attention_matrices