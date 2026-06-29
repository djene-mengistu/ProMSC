import argparse
import os
from datetime import datetime
from pyexpat import model
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from itertools import cycle
from tensorboardX import SummaryWriter
from torch.autograd import Variable
# from torch.nn.modules.loss import CrossEntropyLoss
# from data_loaders.tut_loaders_multitask import load_DEF_dataloaders
from data_loaders.neu_main_dataloaders import load_DEF_dataloaders
from utilities.metrics import mIoU, mDice, pixel_accuracy 
from utilities.losses import DiceCELoss, FeatureSimilarityLoss
from utilities.ramps import sigmoid_rampup
# from utilities.positionspatial import FeatureConsistencyLoss #, refine_with_correlation
from utilities.positionspatial_correlation_cotrain import FeatureConsistencyLoss
from models.segmodels_multitasking import CONVNEXTMODEL, SEGFORMER
from utilities.utilities import get_logger, create_dir 
from utilities.cutmix_mixup import cutmix_tensor
from utilities.protomatch import consistency_loss, compute_prototypes, prototype_loss
# from sklearn.cluster import KMeans
from einops import rearrange
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model1', type=str, default='MiT-B0', help='model_name')
parser.add_argument('--model2', type=str, default='ConvNeXt-T', help='model_name')
parser.add_argument('--init_weight_1', type=str,  default="./models/weights/mit_b0.pth", help='initial model weights')
parser.add_argument('--init_weight_2', type=str,  default="./models/weights/convnext_tiny_1k_224_ema.pth", help='initial model weights')
parser.add_argument('--training_type', type=str, default='SSL', choices=['SSL', 'supervised'], help='Training type (default: SSL)')
parser.add_argument('--dataset', type=str, default='TUT', choices=['NEU', 'DAGM', 'MTD', 'TUT', 'CSD', 'Crack500'], help='Dataset name (default: NEU)')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--unlabeled_ratio', type=float, default=0.9, help='labeled data')
parser.add_argument('--train_img_path', type=str, default="./NEU_VOC/train/train_images/", help='train image path')
parser.add_argument('--train_mask_path', type=str, default="./NEU_VOC/train/train_annot/", help='train mask path')
parser.add_argument('--test_img_path', type=str, default="./NEU_VOC/test/test_images/", help='test image path')
parser.add_argument('--test_mask_path', type=str, default="./NEU_VOC/test/test_annot/", help='test mask path')
parser.add_argument('--consistency', type=float, default=0.5, help='consistency')# Maximum consistency coefficient for pseudo-supervision
parser.add_argument('--ramp_length', type=float, default=200.0, help='ramp length where the consistency coefficient increases from 0 to the maximum value')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

args = parser.parse_args()
dataset_classes = {
    'NEU': 4,
    'DAGM': 7,
    'MTD': 6,
    'CSD': 2,
    'TUT': 2,
    'Crack500': 2
}

args.num_classes = dataset_classes[args.dataset]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # specify the GPU id's, GPU id's start from 0.

epochs = args.num_epochs
base_lr = args.base_lr
# max_iterations = args.max_iterations
sim_loss = FeatureSimilarityLoss()
criterion = DiceCELoss(num_classes=args.num_classes, dice_weight=0.5, ce_weight=0.5, smooth=1e-8, ignore_background=False) #CHECK
spcon_loss = FeatureConsistencyLoss(pos_weight=1.0, spatial_weight=1.0, use_l1=False, max_spatial_size=14)  # Position and spatial consistency loss
labeled_ratio = round(1 - args.unlabeled_ratio, 2)
labeled_ratio_str = f"{int(labeled_ratio * 100)}P"
 
#Define Model
# model_name = args.model.lower()

# if 'convnext' in model_name:
#     model = CONVNEXTMODEL(args.model, args.num_classes)
# elif 'mit' in model_name:
#     model = SEGFORMER(args.model, args.num_classes)
# else:
#     raise ValueError(f"Unsupported model name: {args.model}")
# model = SEGFORMER(args.model, args.num_classes)
model1 = SEGFORMER(args.model1, args.num_classes)
model2 = CONVNEXTMODEL(args.model2, args.num_classes)
model1.init_pretrained(args.init_weight_1)
model2.init_pretrained(args.init_weight_2)

def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.ramp_length)
#Date
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model1 = model1
        self.model2 = model2
        # self.teacher_model = teacher_model  # For teacher-student training
        self.save_best_model_1 = False 
        self.save_best_model_2 = False
        self._init_logger()
    def _init_logger(self):

        log_dir = f"./EXP/{args.dataset}/COTRAIN_seg_{args.training_type}_singlemod_{labeled_ratio_str}_{current_time}/" #'trained_weights/NEU_seg/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        self.model1.to(device)
        self.model2.to(device)
        optimizer_1 = torch.optim.Adam(self.model1.parameters(), lr=base_lr)
        optimizer_2 = torch.optim.Adam(self.model2.parameters(), lr=base_lr)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", factor=0.1, min_lr = 0.000001, patience=40, verbose=True)
        scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, mode="max", factor=0.1, min_lr = 0.000001, patience=40, verbose=True)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, min_lr = 0.000001, patience=40, verbose=True)
        
        loaders = load_DEF_dataloaders(args.train_img_path, args.train_mask_path, args.test_img_path, args.test_mask_path, args.batch_size, args.unlabeled_ratio)
        train_loader, unlabeled_loader, test_loader = loaders['train'], loaders['train_u'], loaders['test'] #DAGM and Others
        
        self.logger.info("🚀💥🚀 Training started for: {} | 🏷️ Ratio: {} | 🔥 Training Type: {} | 🔢 labeled_loader: {} | 📈 unlabeled_loader: {} ".format(
            args.dataset, labeled_ratio_str, args.training_type, len(train_loader), len(unlabeled_loader)))
        self.logger.info('===========================++++============================+++============================+++++====================++++====')

        # model1.train()
        iter_num = 0
        num_classes = args.num_classes  # Example: cracks, scratches
       
        for epoch in range(1, epochs):

            running_train_iou = 0.0; running_train_dice = 0.0; running_consis_loss = 0.0; 
            running_train_loss = 0.0; running_cps_loss = 0.0;  running_val_loss = 0.0; 
            running_val_iou_1 = 0.0; running_val_dice_1 = 0.0; running_val_accuracy_1 = 0.0; 
            running_val_iou_2 = 0.0; running_val_dice_2 = 0.0; running_val_accuracy_2 = 0.0; 
            running_corr_loss = 0.0; running_proto_loss = 0.0; running_img_loss = 0.0; 
            running_mixup_loss = 0.0; 
                        
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            
            self.model1.train()
            self.model2.train()

            semi_dataloader = iter(zip(cycle(train_loader), cycle(unlabeled_loader)))
            iter_per_epoch = 80
                    
            for iteration in range (1, iter_per_epoch): #(zip(train_loader, unlabeled_train_loader)):
                                
                data = next(semi_dataloader)
                
                (inputs_l, labels_l), (inputs_U_W, inputs_U_S, labels_U) = data #data[0][0], data[0][1]
                inputs_l, labels_l = Variable(inputs_l), Variable(labels_l)
                inputs_l, labels_l = inputs_l.to(device), labels_l.to(device)
                inputs_U_W, inputs_U_S, labels_U = Variable(inputs_U_W), Variable(inputs_U_S), Variable(labels_U)
                inputs_U_W, inputs_U_S, labels_U = inputs_U_W.to(device), inputs_U_S.to(device), labels_U.to(device)


                self.model1.train()
                feat1, _, outputs1 = self.model1(inputs_l) 
                feat2, _, outputs2 = self.model2(inputs_l)                 
                #Unlabeled samples output
                unfeat_w, _, un_outputs_w = self.model1(inputs_U_W)
                un_outputs_soft_w = torch.softmax(un_outputs_w, dim=1)

                unfeat_s, _, un_outputs_s = self.model2(inputs_U_S)
                un_outputs_soft_s = torch.softmax(un_outputs_s, dim=1)

                sup_loss =criterion(outputs1, labels_l.long())  + criterion(outputs2, labels_l.long()) #Supervised loss on labeled samples
                
                #Compute the Correlation loss and correlation maps
                corr_loss, _ = spcon_loss(unfeat_w, unfeat_s)

                #Creating the pseudo-labels
                pseudo1 = torch.argmax(un_outputs_soft_w.detach(), dim=1, keepdim=False)                  
                pseudo2 = torch.argmax(un_outputs_soft_s.detach(), dim=1, keepdim=False)
                ps1 = criterion(un_outputs_w, pseudo2)
                ps2 = criterion(un_outputs_s, pseudo1)  
                cps_loss = (ps1 + ps2) 

                loss_con = consistency_loss(un_outputs_w, un_outputs_s, temperature=0.1, loss_type='mse') # Compute consistency loss between weak and strong augmentations
                #Sim Loss
                # sim_loss_out_lbl = 0.5*(sim_loss(outputs, outputs_1) + sim_loss(outputs, outputs_2))
                sim_loss_out =(sim_loss(un_outputs_w, un_outputs_s.detach()) + sim_loss(un_outputs_w.detach(), un_outputs_s))/2
                # sim_loss_1 = 0.5*(sim_loss(unfeat_w[0], unfeat_s[0].detach()) + sim_loss(unfeat_w[0].detach(), unfeat_s[0]))
                # sim_loss_2 = 0.5*(sim_loss(unfeat_w[1], unfeat_s[1].detach()) + sim_loss(unfeat_w[1].detach(), unfeat_s[1]))
                # sim_loss_3 = 0.5*(sim_loss(unfeat_w[2], unfeat_s[2].detach()) + sim_loss(unfeat_w[2].detach(), unfeat_s[2]))
                sim_loss_4 = (sim_loss(unfeat_w[3], unfeat_s[3].detach()) + sim_loss(unfeat_w[3].detach(), unfeat_s[3]))/2
                
                # imgSLoss = sim_loss_out + sim_loss_1 + sim_loss_2 + sim_loss_3 + sim_loss_4 #Image-lvel similarirty loss
                imgSLoss = sim_loss_out + sim_loss_4 #Image-lvel similarirty loss    
                #PROROTYPE LEARNING
                features_labeled = feat1[-1].detach() # [B, 256, 14, 14] (Detached to avoid gradients and consider the labeld proto as a target)
                # feat_l_up = F.interpolate(features_labeled, size=labels_l.shape[1:], mode='bilinear')
                features_unlabeled_w = unfeat_w[-1]  # [B, 256, 14, 14]
                # feat_u_w_up = F.interpolate(features_unlabeled_w, size=labels_l.shape[1:], mode='bilinear')
                features_unlabeled_s = unfeat_s[-1]  # [B, 256, 14, 14]
                # feat_u_s_up = F.interpolate(features_unlabeled_s, size=labels_l.shape[1:], mode='bilinear')

                with torch.no_grad():
                    # comb_prob = F.interpolate((un_outputs_w_or + un_outputs_s_or)/2, (14,14), mode='bilinear', align_corners=True) #Average of strong and weak outputs 
                    # comb_prob = un_outputs_w  # Average of strong and weak outputs
                    comb_prob_soft= torch.softmax(un_outputs_w, dim=1)  # Apply softmax to get probabilities
                    max_probs, pseudo_t = torch.max(comb_prob_soft, dim=1)  # Shapes: [12, 14, 14]
                    confident_mask = max_probs > 0.6 #bst is obtained at 0.8        
                labels_l_down = F.interpolate(labels_l.unsqueeze(1).float(), size=(14, 14), mode='nearest').squeeze(1).long()  # Shape: [12, 14, 14]
                pseudo_labels_down = F.interpolate(pseudo_t.unsqueeze(1).float(), size=(14, 14), mode='nearest').squeeze(1).long()  # Shape: [12, 14, 14]
                confident_mask_down = F.interpolate(confident_mask.unsqueeze(1).float(), size=(14, 14), mode='nearest').squeeze(1).bool()  # Shape: [12, 14, 14]

                # Compute prototypes
                # prototypes_l = compute_prototypes(feat_l_up, labels_l, num_classes)
                # prototypes_u_w = compute_prototypes(feat_u_w_up, pseudo_t, num_classes, mask=None)
                # prototypes_u_s = compute_prototypes(feat_u_s_up, pseudo_t, num_classes, mask=confident_mask)
                # prototypes_u = (prototypes_u_w + prototypes_u_s) / 2
                prototypes_l = compute_prototypes(features_labeled, labels_l_down, num_classes)
                prototypes_u_w = compute_prototypes(features_unlabeled_w, pseudo_labels_down, num_classes, mask=confident_mask_down)
                # prototypes_u_s = compute_prototypes(features_unlabeled_s, pseudo_labels_down, num_classes, mask=confident_mask_down)
                # prototypes_u = (prototypes_u_w + prototypes_u_s) / 2

                # Compute prototype loss
                loss_proto = prototype_loss(prototypes_l, prototypes_u_w, loss_type='cosine') #+  prototype_loss(prototypes_l, prototypes_u_s, loss_type='cosine')#CHECK
                
                #CUTMIX 
                b_l = inputs_l.size(0)  
                inputs_U_W = inputs_U_W[:b_l]
                un_outputs_w = un_outputs_w[:b_l]
                with torch.no_grad(): 
                    pseudo_labels = torch.softmax(un_outputs_w.detach(), dim=1)  # [B, C, H, W]
                    pseudo_hard = pseudo_labels.argmax(dim=1)  # [B, H, W]
                
                # Apply CutMix per sample
                cutmixed_imgs = []
                cutmixed_labels = []

                for i in range(b_l):
                    img_l = inputs_l[i]
                    img_u = inputs_U_W[i]
                    lbl_l = labels_l[i]
                    lbl_u = pseudo_hard[i]

                    mixed_img, mixed_lbl, _ = cutmix_tensor(img_l, img_u, lbl_l, lbl_u)
                    cutmixed_imgs.append(mixed_img)
                    cutmixed_labels.append(mixed_lbl)
                # Stack into batch
                cutmixed_imgs = torch.stack(cutmixed_imgs)       # Shape: [B, 3, H, W]
                cutmixed_labels = torch.stack(cutmixed_labels)   # Shape: [B, H, W]
                _, _, mixed_outputs_1 = self.model1(cutmixed_imgs)  # [B, C, H, W]  
                _, _, mixed_outputs_2 = self.model2(cutmixed_imgs)  # [B, C, H, W] 

                loss_mix = criterion(mixed_outputs_1, cutmixed_labels.long()) + criterion(mixed_outputs_2, cutmixed_labels.long())  # Cross-entropy loss for mixed outputs and labels                
                consistency_weight = get_current_consistency_weight(iter_num // 80) #Consistency weight multipliers             
                loss = sup_loss + consistency_weight * (0.6*cps_loss + 0.6*corr_loss + 0.4*loss_con + 0.4*loss_proto + 20.0*imgSLoss + loss_mix) #+ 0.5*imgSLoss #+ 0.5*sim_loss_out_lbl
                
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()                
                loss.backward()
                # optimizer.step()
                optimizer_1.step()
                optimizer_2.step()

                running_train_loss += loss.item()
                running_cps_loss += cps_loss.item() 
                running_corr_loss += corr_loss.item()#CHECK
                running_consis_loss += loss_con.item() #CHECK
                running_img_loss += imgSLoss.item() 
                running_proto_loss += loss_proto.item() #CHECK  
                running_mixup_loss += loss_mix.item() #CHECK              
                running_train_iou += mIoU(outputs1, labels_l, args.num_classes)
                running_train_dice += mDice(outputs1, labels_l, args.num_classes)
                
                for param_group in optimizer_1.param_groups:
                    lr_1 = param_group['lr'] #For plotting the learning rate change during the training process  
                for param_group in optimizer_2.param_groups:
                    lr_2 = param_group['lr'] #For plotting the learning rate change during the training process              
                                
                # Update teacher weights
                # update_teacher_params(self.model, self.teacher_model, 0.99, iter_num)
                iter_num = iter_num + 1
            
            epoch_train_dice = ( running_train_dice) / (iter_per_epoch)
            epoch_train_iou = ( running_train_iou) / (iter_per_epoch)

            epoch_loss = (running_train_loss) / (iter_per_epoch)            
            epoch_cps_loss = (running_cps_loss) / (iter_per_epoch)
            epoch_mixup_loss = (running_mixup_loss) / (iter_per_epoch)
            epoch_corr_loss = (running_corr_loss) / (iter_per_epoch)
            epoch_consis_loss = (running_consis_loss) / (iter_per_epoch)
            epoch_img_loss = (running_img_loss) / (iter_per_epoch)
            epoch_proto_loss = (running_proto_loss) / (iter_per_epoch)

            # self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)      
            # self.writer.add_scalar('Train/bnc-Loss', epoch_bnc_loss, epoch)           
            self.writer.add_scalar('Train/CPS-Loss', epoch_cps_loss, epoch)
            self.writer.add_scalar('Train/Corr-Loss', epoch_corr_loss, epoch)
            self.writer.add_scalar('Train/Consis-Loss', epoch_consis_loss, epoch)
            self.writer.add_scalar('Train/IMG-Loss', epoch_img_loss, epoch)
            self.writer.add_scalar('Train/Proto-Loss', epoch_proto_loss, epoch)
            self.writer.add_scalar('Train/Mixup-Loss', epoch_mixup_loss, epoch)
                      
            self.writer.add_scalar('Train/Dice', epoch_train_dice, epoch)
            self.writer.add_scalar('Train/IoU', epoch_train_iou, epoch) 
            # self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/lr1', lr_1, epoch)
            self.writer.add_scalar('info/lr2', lr_2, epoch)
            self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] | Loss: {:.4f} | CPS: {:.4f}| corr: {:.4f} | CON: {:.4f}| IMG: {:.4f}| Proto: {:.4f}| MIX: {:.4f}| lr: {:.6f}'.format(
                    datetime.now(), epoch, epochs, epoch_loss, epoch_cps_loss, epoch_corr_loss, epoch_consis_loss, epoch_img_loss, epoch_proto_loss, epoch_mixup_loss, lr_1))
            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] |Train: | Dice: {:.4f} | IoU: {:.4f}'.format(
                    datetime.now(), epoch, epochs, epoch_train_dice, epoch_train_iou))
            
            torch.cuda.empty_cache()

            self.model1.eval()
            self.model2.eval()
            for i, pack in enumerate(test_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images, gts = images.to(device), gts.to(device)                    
                    _, _, pred1 = self.model1(images)  
                    _, _, pred2 = self.model2(images)
                     

                val_loss = criterion(pred1, gts.long()) + criterion(pred2, gts.long())
                running_val_loss += val_loss.item()  
                              
                running_val_iou_1 += mIoU(pred1, gts, args.num_classes)
                running_val_dice_1 += mDice(pred1, gts, args.num_classes)
                running_val_accuracy_1 += pixel_accuracy(pred1, gts)     
                
                running_val_iou_2 += mIoU(pred2, gts, args.num_classes)
                running_val_dice_2 += mDice(pred2, gts, args.num_classes)
                running_val_accuracy_2 += pixel_accuracy(pred2, gts)         
                 
            epoch_loss_val = running_val_loss / len(test_loader)            
            epoch_dice_val_1 = running_val_dice_1 / len(test_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(test_loader)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(test_loader)
            
            epoch_dice_val_2 = running_val_dice_2 / len(test_loader)
            epoch_iou_val_2 = running_val_iou_2 / len(test_loader)
            epoch_accuracy_val_2 = running_val_accuracy_2 / len(test_loader)

            scheduler_1.step(epoch_dice_val_1)   
            scheduler_2.step(epoch_dice_val_2)     
            #Model-1 training
            self.writer.add_scalar('Val/loss', epoch_loss_val, epoch)
            self.writer.add_scalar('Val/IoU1', epoch_iou_val_1, epoch)            
            self.writer.add_scalar('Val/DSC1', epoch_dice_val_1, epoch)
            self.writer.add_scalar('Val/Accuracy1', epoch_accuracy_val_1, epoch)
            
            self.writer.add_scalar('Val/IoU2', epoch_iou_val_2, epoch)            
            self.writer.add_scalar('Val/DSC2', epoch_dice_val_2, epoch)
            self.writer.add_scalar('Val/Accuracy2', epoch_accuracy_val_2, epoch)
            
            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] | Loss: {:.4f} | Val: | Dice1: {:.4f} | IoU1: {:.4f} | Dice2: {:.4f} | IoU2: {:.4f}'.format(
                    datetime.now(), epoch, epochs, epoch_loss_val, epoch_dice_val_1, epoch_iou_val_1, epoch_dice_val_2, epoch_iou_val_2))
            mdice_coeff_1 =  epoch_dice_val_1
            mdice_coeff_2 =  epoch_dice_val_2

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True
                self.patience_1 = 0
            else:
                self.save_best_model_1= False
                self.patience_1 += 1    
            
            if self.best_dice_coeff_2 < mdice_coeff_2:
                self.best_dice_coeff_2 = mdice_coeff_2
                self.save_best_model_2 = True
                self.patience_2 = 0
            else:
                self.save_best_model_2= False
                self.patience_2 += 1        
            
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                state_1 = {
                "epoch": epoch,
                "best_dice_1": self.best_dice_coeff_1,
                "state_dict": self.model1.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                }
            if self.save_best_model_2:
                state_2 = { 
                "epoch": epoch,
                "best_dice_2": self.best_dice_coeff_2,
                "state_dict": self.model2.state_dict(),
                "optimizer": optimizer_2.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                filename1 = f"{args.model1}_{args.dataset}_{args.training_type}_{labeled_ratio_str}.pth"
                filename2 = f"{args.model2}_{args.dataset}_{args.training_type}_{labeled_ratio_str}.pth"

                save_path1 = os.path.join(Checkpoints_Path, filename1)
                save_path2 = os.path.join(Checkpoints_Path, filename2)
                torch.save(state_1, save_path1)
                torch.save(state_2, save_path2)

            self.logger.info(
                'Best dice 1: {} | Patience_1:{} | Best dice 2: {} | Patience_2:{}| Conweight:{} '.format(
                    self.best_dice_coeff_1, self.patience_1, self.best_dice_coeff_2, self.patience_2, consistency_weight))
            self.logger.info('=====================================++++============================+++==================+++++=========')
if __name__ == '__main__':
    train_network = Network()
    train_network.run()