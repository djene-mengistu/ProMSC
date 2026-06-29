import torch

class Model:
    def __init__(self, models, ensemble_type='average', device=None):
        """
        Initialize the ensemble model
        Args:
            models: List of individual models
            ensemble_type: 'average' for soft voting or 'maximum' for pixel-level maximum
            device: torch.device object (defaults to CPU if None)
        """
        self.models = models
        self.ensemble_type = ensemble_type
        # Set device to CPU if not provided, otherwise use provided device
        self.device = device if device is not None else torch.device('cpu')
    
    def __call__(self, x):
        """
        Forward pass through the ensemble
        Args:
            x: Input tensor
        Returns:
            pred: Ensemble prediction
        """
        res = []
        feat_map = []
        x = x.to(self.device)  # Move input to the specified device
        with torch.no_grad():
            for m in self.models:
                m.to(self.device)  # Move model to the specified device
                m.eval()
                feat, _, output = m(x)
                res.append(output)
                feat_map.append(feat)
                
        # Stack predictions for easier manipulation
        res = torch.stack(res)
        
        if self.ensemble_type == 'average':
            # Average ensemble (soft voting)
            pred = torch.mean(res, dim=0)
        elif self.ensemble_type == 'maximum':
            # Pixel-level maximum ensemble
            pred = torch.max(res, dim=0)[0]
        else:
            raise ValueError("ensemble_type must be 'average' or 'maximum'")
            
        return feat_map, pred      
           
