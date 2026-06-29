import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from itertools import cycle
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from data_loaders.tut_loaders_multitask import load_DEF_dataloaders
from data_loaders.neu_main_dataloaders import load_DEF_dataloaders
from utilities.metrics import mIoU, mDice, pixel_accuracy 
from utilities.losses import DiceCELoss, FeatureSimilarityLoss
from utilities.ramps import sigmoid_rampup
from utilities.positionspatial_correlation import FeatureConsistencyLoss
from models.segmodels_multitasking import CONVNEXTMODEL, SEGFORMER
from utilities.utilities import get_logger, create_dir 
from utilities.cutmix_mixup import cutmix_tensor
from utilities.protomatch import consistency_loss, compute_prototypes, prototype_loss
# from sklearn.cluster import KMeans
from einops import rearrange
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MiT-B0', help='model_name')
parser.add_argument('--init_weight', type=str,  default="./models/weights/mit_b0.pth", help='initial model weights')
parser.add_argument('--training_type', type=str, default='SSL', choices=['SSL', 'supervised'], help='Training type (default: SSL)')
parser.add_argument('--dataset', type=str, default='TUT', choices=['NEU', 'DAGM', 'MTD', 'TUT', 'CSD', 'Crack500'], help='Dataset name (default: NEU)')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
# parser.add_argument('--max_iterations', type=int, default=12000, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--multi_lr', type=float,  default=5.0, help='multiplier learning rate')
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
model_name = args.model.lower()

if 'convnext' in model_name:
    model = CONVNEXTMODEL(args.model, args.num_classes)
elif 'mit' in model_name:
    model = SEGFORMER(args.model, args.num_classes)
else:
    raise ValueError(f"Unsupported model name: {args.model}")
# model = SEGFORMER(args.model, args.num_classes)
model.init_pretrained(args.init_weight)

def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.ramp_length)
#Date
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff = False
        self.model = model
        # self.teacher_model = teacher_model  # For teacher-student training
        self.save_best_model = False 
        self._init_logger()
    def _init_logger(self):

        log_dir = f"./EXP/{args.dataset}/{args.model}_seg_{args.training_type}_singlemod_{labeled_ratio_str}_{current_time}/" #'trained_weights/NEU_seg/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr) 
        # optimizer = torch.optim.Adam(
        # [
        #     {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': args.base_lr},
        #     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': args.base_lr * args.multi_lr}
        # ], 
        # lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.01 )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.7, min_lr = 0.000001, patience=40, verbose=True)
        
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
            running_val_iou = 0.0; running_val_dice = 0.0; running_val_accuracy = 0.0; 
            running_corr_loss = 0.0; running_proto_loss = 0.0; running_img_loss = 0.0; 
            running_mixup_loss = 0.0; running_bbg_loss = 0.0; 
                        
            optimizer.zero_grad()
            
            self.model.train()

            semi_dataloader = iter(zip(cycle(train_loader), cycle(unlabeled_loader)))
            iter_per_epoch = 120
                    
            for iteration in range (1, iter_per_epoch): #(zip(train_loader, unlabeled_train_loader)):
                                
                data = next(semi_dataloader)
                
                (inputs_l, labels_l), (inputs_U_W, inputs_U_S, labels_U) = data #data[0][0], data[0][1]
                inputs_l, labels_l = Variable(inputs_l), Variable(labels_l)
                inputs_l, labels_l = inputs_l.to(device), labels_l.to(device)
                inputs_U_W, inputs_U_S, labels_U = Variable(inputs_U_W), Variable(inputs_U_S), Variable(labels_U)
                inputs_U_W, inputs_U_S, labels_U = inputs_U_W.to(device), inputs_U_S.to(device), labels_U.to(device)

                self.model.train()
                feat, _, outputs = self.model(inputs_l)                
                #Unlabeled samples output
                unfeat_w, _, un_outputs_w = self.model(inputs_U_W)
                un_outputs_soft_w = torch.softmax(un_outputs_w, dim=1)
                
                unfeat_s, _, un_outputs_s = self.model(inputs_U_S)
                un_outputs_soft_s = torch.softmax(un_outputs_s, dim=1)

                sup_loss =criterion(outputs, labels_l.long())          
                
                #Compute the Correlation loss and correlation maps
                corr_loss, corr_matrix = spcon_loss(unfeat_w, unfeat_s)

                #Creating the pseudo-labels
                pseudo1 = torch.argmax(un_outputs_soft_w.detach(), dim=1, keepdim=False)                  
                pseudo2 = torch.argmax(un_outputs_soft_s.detach(), dim=1, keepdim=False)
                ps1 = criterion(un_outputs_w, pseudo2)
                ps2 = criterion(un_outputs_s, pseudo1)  
                cps_loss = (ps1 + ps2) 

                #If apply the corr_matrix to the pseudo-labels, then uncomment the following lines
                # corr_matrix = corr_matrix.detach()  # Detach the correlation matrix to avoid gradients
                # un_outputs_w_corr = torch.einsum('bchw,bc->bchw', un_outputs_soft_w.detach(), corr_matrix)
                # un_outputs_s_corr = torch.einsum('bchw,bc->bchw', un_outputs_soft_s.detach(), corr_matrix)
                # pseudo1 = torch.argmax(un_outputs_w_corr, dim=1, keepdim=False)                  
                # pseudo2 = torch.argmax(un_outputs_s_corr, dim=1, keepdim=False)
                # ps1 = criterion(un_outputs_w, pseudo2)
                # ps2 = criterion(un_outputs_s, pseudo1)  
                # cps_loss = (ps1 + ps2)


                loss_con = consistency_loss(un_outputs_w, un_outputs_s, temperature=0.7, loss_type='mse') # Compute consistency loss between weak and strong augmentations
                #Sim Loss
                # sim_loss_out_lbl = 0.5*(sim_loss(outputs, outputs_1) + sim_loss(outputs, outputs_2))
                # sim_loss_out =(sim_loss(un_outputs_w, un_outputs_s.detach()) + sim_loss(un_outputs_w.detach(), un_outputs_s))/2
                sim_loss_1 = 0.5*(sim_loss(unfeat_w[0], unfeat_s[0].detach()) + sim_loss(unfeat_w[0].detach(), unfeat_s[0]))
                sim_loss_2 = 0.5*(sim_loss(unfeat_w[1], unfeat_s[1].detach()) + sim_loss(unfeat_w[1].detach(), unfeat_s[1]))
                sim_loss_3 = 0.5*(sim_loss(unfeat_w[2], unfeat_s[2].detach()) + sim_loss(unfeat_w[2].detach(), unfeat_s[2]))
                sim_loss_4 = (sim_loss(unfeat_w[3], unfeat_s[3].detach()) + sim_loss(unfeat_w[3].detach(), unfeat_s[3]))/2
                
                imgSLoss = sim_loss_1 + sim_loss_2 + sim_loss_3 + sim_loss_4 #Image-lvel similarirty loss
                # imgSLoss = sim_loss_out + sim_loss_3 + sim_loss_4 #Image-lvel similarirty loss    
                #PROROTYPE LEARNING
                features_labeled = feat[-1].detach() # [B, 256, 14, 14] (Detached to avoid gradients and consider the labeld proto as a target)
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
                # prototypes_u_w = compute_prototypes(feat_u_w_up, pseudo_t, num_classes, mask=confident_mask)
                # prototypes_u_s = compute_prototypes(feat_u_s_up, pseudo_t, num_classes, mask=confident_mask)
                # prototypes_u = (prototypes_u_w + prototypes_u_s) / 2
                prototypes_l = compute_prototypes(features_labeled, labels_l_down, num_classes, mask=None)
                prototypes_u_w = compute_prototypes(features_unlabeled_w, pseudo_labels_down, num_classes, mask=confident_mask_down)
                prototypes_u_s = compute_prototypes(features_unlabeled_s, pseudo_labels_down, num_classes, mask=confident_mask_down)
                # prototypes_u = (prototypes_u_w + prototypes_u_s) / 2

                # Compute prototype loss
                loss_proto = prototype_loss(prototypes_l, prototypes_u_w, loss_type='cosine') +  prototype_loss(prototypes_l, prototypes_u_s, loss_type='cosine')#CHECK
                
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
                _, _, mixed_outputs = self.model(cutmixed_imgs)  # [B, C, H, W]                
                loss_mix = criterion(mixed_outputs, cutmixed_labels.long())  # Cross-entropy loss for mixed outputs and labels                
                consistency_weight = get_current_consistency_weight(iter_num // 120) #Consistency weight multipliers             
                loss = sup_loss + consistency_weight * (0.8*cps_loss + 0.8*corr_loss + 0.6*loss_con + 0.4*loss_proto + imgSLoss + 0.8*loss_mix) #+ 0.5*imgSLoss #+ 0.5*sim_loss_out_lbl
                # loss = sup_loss + 0.2*cps_loss + 0.3*corr_loss + 0.2*loss_con + 0.2*loss_proto + 10.0*imgSLoss + 0.3*loss_mix #+ 0.5*imgSLoss #+ 0.5*sim_loss_out_lbl
                
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                running_cps_loss += cps_loss.item() 
                running_corr_loss += corr_loss.item()#CHECK
                running_consis_loss += loss_con.item() #CHECK
                running_img_loss += imgSLoss.item() 
                running_proto_loss += loss_proto.item() #CHECK  
                running_mixup_loss += loss_mix.item() #CHECK   
                # running_bbg_loss += bg_loss.item()           
                running_train_iou += mIoU(outputs, labels_l, args.num_classes)
                running_train_dice += mDice(outputs, labels_l, args.num_classes)
                
                for param_group in optimizer.param_groups:
                    lr_1 = param_group['lr'] #For plotting the learning rate change during the training process                
                                
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
            # epoch_bbg_loss = (running_bbg_loss) / (iter_per_epoch)

            # self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)      
            # self.writer.add_scalar('Train/bnc-Loss', epoch_bnc_loss, epoch)           
            self.writer.add_scalar('Train/CPS-Loss', epoch_cps_loss, epoch)
            self.writer.add_scalar('Train/Corr-Loss', epoch_corr_loss, epoch)
            self.writer.add_scalar('Train/Consis-Loss', epoch_consis_loss, epoch)
            self.writer.add_scalar('Train/IMG-Loss', epoch_img_loss, epoch)
            self.writer.add_scalar('Train/Proto-Loss', epoch_proto_loss, epoch)
            self.writer.add_scalar('Train/Mixup-Loss', epoch_mixup_loss, epoch)
            # self.writer.add_scalar('Train/BBG-Loss', epoch_bbg_loss, epoch)
                      
            self.writer.add_scalar('Train/Dice', epoch_train_dice, epoch)
            self.writer.add_scalar('Train/IoU', epoch_train_iou, epoch) 
            # self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/lr1', lr_1, epoch)
            self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] | Loss: {:.4f} | CPS: {:.4f}| corr: {:.4f} | CON: {:.4f}| IMG: {:.4f}| Proto: {:.4f}| MIX: {:.4f}| lr: {:.6f}'.format(
                    datetime.now(), epoch, epochs, epoch_loss, epoch_cps_loss, epoch_corr_loss, epoch_consis_loss, epoch_img_loss, epoch_proto_loss, epoch_mixup_loss, lr_1))
            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] |Train: | Dice: {:.4f} | IoU: {:.4f}'.format(
                    datetime.now(), epoch, epochs, epoch_train_dice, epoch_train_iou))
            
            torch.cuda.empty_cache()

            self.model.eval()
            for i, pack in enumerate(test_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images, gts = images.to(device), gts.to(device)                    
                    _, _, pred = self.model(images)   

                val_loss = criterion(pred, gts.long()) 
                running_val_loss += val_loss.item()  
                              
                running_val_iou += mIoU(pred, gts, args.num_classes)
                running_val_dice += mDice(pred, gts, args.num_classes)
                running_val_accuracy += pixel_accuracy(pred, gts)              

            epoch_loss_val = running_val_loss / len(test_loader)            
            epoch_dice_val = running_val_dice / len(test_loader)
            epoch_iou_val = running_val_iou / len(test_loader)
            epoch_accuracy_val = running_val_accuracy / len(test_loader)

            scheduler.step(epoch_dice_val)        
            #Model-1 training
            self.writer.add_scalar('Val/loss', epoch_loss_val, epoch)
            self.writer.add_scalar('Val/IoU', epoch_iou_val, epoch)            
            self.writer.add_scalar('Val/DSC', epoch_dice_val, epoch)
            self.writer.add_scalar('Val/Accuracy', epoch_accuracy_val, epoch)
            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] | Loss: {:.4f} | Val: | Dice: {:.4f} | IoU: {:.4f} '.format(
                    datetime.now(), epoch, epochs, epoch_loss_val, epoch_dice_val, epoch_iou_val))
            mdice_coeff =  epoch_dice_val

            if self.best_dice_coeff < mdice_coeff:
                self.best_dice_coeff = mdice_coeff
                self.save_best_model = True
                self.patience = 0
            else:
                self.save_best_model = False
                self.patience += 1            
            
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model:
                state = {
                "epoch": epoch,
                "best_dice": self.best_dice_coeff,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                filename = f"{args.model}_{args.dataset}_{args.training_type}_{labeled_ratio_str}.pth"
                save_path1 = os.path.join(Checkpoints_Path, filename)
                torch.save(state, save_path1)           

            self.logger.info(
                'Best dice: {} | Patience:{} Conweight:{} '.format(
                    self.best_dice_coeff, self.patience, consistency_weight))
            self.logger.info('=====================================++++============================+++==================+++++=========')
if __name__ == '__main__':
    train_network = Network()
    train_network.run()