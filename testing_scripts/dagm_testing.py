import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F
import albumentations as A
from tqdm import tqdm
from tabulate import tabulate
import random
from datetime import datetime

# Import your custom models
from models.segmodels_multitasking import SEGFORMER
from models.segmodels import CONVNEXTMODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentationEvaluator:
    def __init__(self, model1_path, model2_path, use_model1=True):
        self.device = device
        self.setup_models(model1_path, model2_path, use_model1)
        
    def setup_models(self, model1_path, model2_path, use_model1):
        """Initialize and load models"""
        self.model1 = SEGFORMER('MiT-B0', 7).to(self.device)
        self.model2 = CONVNEXTMODEL('ConvNeXt-T', 2).to(self.device)
        
        checkpoint_1 = torch.load(model1_path)
        checkpoint_2 = torch.load(model2_path)
        
        print(f"Model 1 epoch: {checkpoint_1['epoch']}")
        print(f"Model 2 epoch: {checkpoint_2['epoch']}")
        
        self.model1.load_state_dict(checkpoint_1['state_dict'])
        self.model2.load_state_dict(checkpoint_2['state_dict'])
        
        self.model = self.model1 if use_model1 else self.model2
        self.model_name = "SegFormer" if use_model1 else "ConvNeXt"
        print(f"Using {self.model_name}")

    @staticmethod
    def pixel_accuracy(output, mask):
        with torch.no_grad():
            output = torch.argmax(F.softmax(output, dim=1), dim=1)
            correct = torch.eq(output, mask).int()
            return float(correct.sum()) / float(correct.numel())

    @staticmethod
    def mIoU(pred_mask, mask, smooth=1e-10, n_classes=7):
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1).view(-1)
            mask = mask.view(-1)
            
            iou_per_class = []
            for clas in range(1, n_classes):
                true_class = (pred_mask == clas)
                true_label = (mask == clas)
                
                if true_label.sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().float().item()
                    union = torch.logical_or(true_class, true_label).sum().float().item()
                    iou = (intersect + smooth) / (union + smooth)
                    iou_per_class.append(iou)
            
            result = np.nanmean(iou_per_class)
            return result if not np.isnan(result) else 0.0

    @staticmethod
    def mDice(pred_mask, mask, smooth=1e-10, n_classes=7):
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1).contiguous().view(-1)
            mask = mask.contiguous().view(-1)
            
            dice_per_class = []
            for clas in range(1, n_classes):
                true_class = pred_mask == clas
                true_label = mask == clas
                
                if true_label.sum().item() == 0:
                    dice_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().float().item()
                    dice = (2 * intersect + smooth) / (true_class.sum().item() + true_label.sum().item() + smooth)
                    dice_per_class.append(dice)
            
            return np.nanmean(dice_per_class)

    def predict(self, image, mask, metric='iou'):
        """Predict mask and calculate metric"""
        self.model.eval()
        
        t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_t = t(image).unsqueeze(0).to(self.device)
        mask_t = mask.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, _, output = self.model(image_t)
            
            if metric.lower() == 'iou':
                score = self.mIoU(output, mask_t)
            elif metric.lower() == 'dice':
                score = self.mDice(output, mask_t)
            elif metric.lower() == 'pixel':
                score = self.pixel_accuracy(output, mask_t)
            else:
                raise ValueError("Metric must be 'iou', 'dice', or 'pixel'")
            
            pred_mask = torch.argmax(output, dim=1).cpu().squeeze(0)
            
        return pred_mask, score

    def evaluate_dataset(self, test_set, metrics=['iou']):
        """Evaluate model on entire dataset"""
        results = {metric: [] for metric in metrics}
        
        for i in tqdm(range(len(test_set)), desc="Evaluating"):
            img, mask, id = test_set[i]
            
            for metric in metrics:
                _, score = self.predict(img, mask, metric)
                results[metric].append(score)
        
        return results

class NEUTestDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.file_ids = self._get_file_ids()
        
    def _get_file_ids(self):
        """Get all file IDs without extensions"""
        file_ids = []
        for filename in os.listdir(self.img_path):
            if filename.endswith('.jpg'):
                file_ids.append(filename.split('.')[0])
        return file_ids
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # Load image
        img = cv2.imread(os.path.join(self.img_path, file_id + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(os.path.join(self.mask_path, file_id + '.png'), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        else:
            img = Image.fromarray(img)
        
        mask = torch.from_numpy(mask).long()
        return img, mask, file_id

def decode_segmap(mask, n_classes=7):
    """Convert mask tensor to RGB image"""
    label_colors = {
        0: [0, 0, 0],       # Background
        1: [255, 0, 0],     # Class 1
        2: [0, 255, 0],     # Class 2
        3: [0, 0, 255],     # Class 3
        4: [255, 255, 0],   # Class 4
        5: [0, 255, 255],   # Class 5
        6: [255, 0, 255]  # Class 6
    }
    
    mask_np = mask.numpy() if torch.is_tensor(mask) else mask
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    
    for label, color in label_colors.items():
        if label < n_classes:
            rgb[mask_np == label] = color
    
    return rgb

def save_visualization(image, true_mask, pred_mask, score, output_path, file_id):
    """Save visualization with original filename"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Input Image', 'Ground Truth', f'Prediction (mIoU: {score:.3f})']
    images = [image, decode_segmap(true_mask), decode_segmap(pred_mask)]
    
    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=120, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # Configuration
    IMAGE_PATH = "./DAGM_SSL_25/test/test_images/"
    MASK_PATH = "./DAGM_SSL_25/test/test_annot/"
    
    MODEL1_PATH = "./EXP/DAGM/MiT-B0_seg_SSL_singlemod_30P_20250826_161024/Checkpoints/MiT-B0_DAGM_SSL_30P.pth"
    MODEL2_PATH = "./EXP/DAGM/ConvNeXt-T_seg_supervised_singlemod_10P_20250905_150230/Checkpoints/ConvNeXt-T_DAGM_supervised_10P.pth"
    
    # Setup
    transform = A.Resize(224, 224, interpolation=cv2.INTER_NEAREST)
    test_set = NEUTestDataset(IMAGE_PATH, MASK_PATH, transform=transform)
    
    evaluator = SegmentationEvaluator(MODEL1_PATH, MODEL2_PATH, use_model1=True)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "./Output_DAGM"
    model_name = os.path.basename(MODEL1_PATH).split('.')[0]  # Get filename without extension
    output_dir = os.path.join(base_output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate full dataset
    print("Evaluating on test set...")
    results = evaluator.evaluate_dataset(test_set, metrics=['iou'])
    
    # Save random samples
    print("Saving sample visualizations...")
    random.seed(42)  # set the seed for reproducibility
    random_indices = random.sample(range(len(test_set)), min(250, len(test_set)))
    
    for idx in random_indices:
        image, true_mask, file_id = test_set[idx]
        pred_mask, score = evaluator.predict(image, true_mask, 'iou')
        
        output_filename = f"{file_id}_mIoU_{score:.3f}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        save_visualization(image, true_mask, pred_mask, score, output_path, file_id)
    
    # Print results
    mean_iou = np.nanmean(results['iou']) * 100
    results_table = [["mIoU", f"{mean_iou:.2f}%"]]
    
    print("\n" + "="*50)
    print(f"Evaluation Results - {evaluator.model_name}")
    print("="*50)
    print(tabulate(results_table, headers=["Metric", "Score"], tablefmt="grid"))
    print("="*50)
    
    # Save results to file
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {evaluator.model_name}\n")
        f.write(f"Dataset: {len(test_set)} samples\n")
        f.write(f"mIoU: {mean_iou:.2f}%\n")
        f.write(f"Evaluation time: {timestamp}\n")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()