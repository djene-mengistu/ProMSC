import argparse
import os
import sys
sys.path.append('./ProMSC')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used
from datetime import datetime
from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
# from data.crack_loaders_q import load_DEF_dataloaders 
# from data.neu_Q_loaders import load_DEF_dataloaders
from data_loaders.dagm_loaders import load_DEF_dataloaders
from utilities.metrics import mIoU, pixel_accuracy, mDice
from utilities.losses import DiceCELoss
from utilities.ramps import sigmoid_rampup
from utilities.utilities import get_logger, count_params
from models.segmodels_multitasking import CONVNEXTMODEL, SEGFORMER
import os
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MiT-B0', help='model_name')
parser.add_argument('--init_weight', type=str,  default="./models/weights/mit_b0.pth", help='initial model weights')
parser.add_argument('--training_type', type=str, default='SSL', choices=['SSL', 'supervised'], help='Training type (default: SSL)')
parser.add_argument('--dataset', type=str, default='TUT', choices=['NEU', 'DAGM', 'MTD', 'CSD', 'TUT', 'Crack500'], help='Dataset name (default: NEU)')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=30250, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--unlabeled_ratio', type=float, default=0.9, help='labeled data')
parser.add_argument('--train_img_path', type=str, default="./NEU_VOC/train/train_images/", help='train image path')
parser.add_argument('--train_mask_path', type=str, default="./NEU_VOC/train/train_annot/", help='train mask path')
parser.add_argument('--test_img_path', type=str, default="./NEU_VOC/test/test_images/", help='test image path')
parser.add_argument('--test_mask_path', type=str, default="./NEU_VOC/test/test_annot/", help='test mask path')
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
max_iterations = args.max_iterations
criterion = DiceCELoss(num_classes=args.num_classes, dice_weight=0.5, ce_weight=0.5, smooth=1e-8, ignore_background=False)
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
# print(model)
#Date
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model = model
        
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
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.00001, patience=30, verbose=True)
      
        loaders = load_DEF_dataloaders(args.train_img_path, args.train_mask_path, args.test_img_path, args.test_mask_path, args.batch_size, args.unlabeled_ratio)
            
        train_loader, train_u_loader, test_loader = loaders['train'], loaders['train_u'], loaders['test']
        # self.logger.info("train_loader {} test_loader {} test_loader {}".format(len(train_loader), len(test_loader), len(test_loader)))
        params = count_params(self.model)
        
        # labeled_ratio = round(1 - args.unlabeled_ratio, 2)
        self.logger.info("🚀💥🚀 Training started for: {} | 🧠 Model:{} | Params:{:.1f}M | 🏷️ Ratio: {} | 🔥 Training Type: {} | 🔢 train_loader: {} | 📈 test_loader: {} ".format(
            args.dataset, args.model, params, labeled_ratio_str, args.training_type, len(train_loader), len(test_loader)))
        self.logger.info('===========================++++============================+++==================+++++====================++++========')

        # model1.train()
        iter_num = 0
        iter_per_epoch = len(train_loader) #Change accordingly for each dataset and data proportion
       
        for epoch in range(0, epochs):

            running_train_loss = 0.0
            running_train_iou_1 = 0.0
            running_train_dice_1 = 0.0            
            running_val_loss = 0.0                        
            running_val_iou_1 = 0.0
            running_val_dice_1 = 0.0
            # running_val_accuracy_1 = 0.0
            
            optimizer_1.zero_grad()
            
            self.model.train()
            data_dataloader = iter(cycle(train_loader))
                    
            # for iteration, data in enumerate (train_loader): #(zip(train_loader, unlabeled_train_loader)): 
            for iteration in range (1, iter_per_epoch): #(zip(train_loader, unlabeled_train_loader)):               
                data = next(data_dataloader)
                inputs_l, labels_l = data            
                inputs_l, labels_l = Variable(inputs_l), Variable(labels_l)
                inputs_l, labels_l = inputs_l.to(device), labels_l.to(device)              
                
                self.model.train()
                _, _, outputs_1 = self.model(inputs_l)
                # outputs_soft_1 = torch.softmax(outputs_1, dim=1)
                  
                loss = criterion(outputs_1, labels_l.long()) #+ 0.2*loss_con                            
                optimizer_1.zero_grad()
                
                loss.backward()
                optimizer_1.step()
                running_train_loss += loss.item()                
                running_train_iou_1 += mIoU(outputs_1, labels_l, num_classes=args.num_classes, include_background=False)
                running_train_dice_1 += mDice(outputs_1, labels_l, num_classes=args.num_classes, include_background=False)                
                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    lr_ = param_group['lr']                
                iter_num = iter_num + 1            
            epoch_loss = (running_train_loss) / (len(train_loader))
            epoch_train_iou = (running_train_iou_1) / (len(train_loader))
            epoch_train_dice = (running_train_dice_1) / (len(train_loader))

            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
            self.writer.add_scalar('Train/mDice', epoch_train_dice, epoch)
            self.writer.add_scalar('Train/mIoU', epoch_train_iou, epoch)
            self.writer.add_scalar('info/lr', lr_, epoch)

            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] | Loss: {:.4f} | Dice: {:.4f} | IoU: {:.4f}| lr: {:.6f}'.format(
                    datetime.now(), epoch, epochs, epoch_loss, epoch_train_dice, epoch_train_iou, lr_))
            torch.cuda.empty_cache()

            self.model.eval()
            for i, pack in enumerate(test_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images, gts = images.to(device), gts.to(device)                    
                    _, _, prediction_1 = self.model(images)
                    # Prediction_1_soft = torch.softmax(prediction_1, dim=1)
                val_loss = criterion(prediction_1, gts.long())
                running_val_loss += val_loss                
                running_val_iou_1 += mIoU(prediction_1, gts, num_classes=args.num_classes, include_background=False)
                # running_val_accuracy_1 += pixel_accuracy(prediction_1, gts)
                running_val_dice_1 += mDice(prediction_1, gts, num_classes=args.num_classes, include_background=False)
                 
            epoch_loss_val = running_val_loss / len(test_loader)
            epoch_dice_val_1 = running_val_dice_1 / len(test_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(test_loader)
            # epoch_accuracy_val_1 = running_val_accuracy_1 / len(test_loader)
            scheduler_1.step(epoch_dice_val_1)
            self.writer.add_scalar('Val/loss', epoch_loss_val, epoch)
            self.writer.add_scalar('Val/DSC-1', epoch_dice_val_1, epoch)
            self.writer.add_scalar('Val/IoU-1', epoch_iou_val_1, epoch)
            # self.writer.add_scalar('Val/Accuracy-1', epoch_accuracy_val_1, epoch)
            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] | Val_Loss: {:.4f} | Val_Dice: {:.4f} | Val_IoU: {:.4f}'.format(
                    datetime.now(), epoch, epochs, epoch_loss_val, epoch_dice_val_1, epoch_iou_val_1))
            torch.cuda.empty_cache()            
            
            mdice_coeff_1 =  epoch_dice_val_1

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience += 1                        
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                state_1 = {
                "epoch": epoch,
                "best_dice_1": self.best_dice_coeff_1,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                # torch.save(state_1, Checkpoints_Path + '/baseline_10p.pth')
                filename = f"{args.model}_{args.dataset}_{args.training_type}_{labeled_ratio_str}.pth"
                save_path = os.path.join(Checkpoints_Path, filename)

                torch.save(state_1, save_path)

            self.logger.info(
                '{} Epoch [{:03d}/{:03d}] | Best dice coef: {:.4f} | Patience: {:.4f}'.format(
                    datetime.now(), epoch, epochs, self.best_dice_coeff_1, self.patience))
            self.logger.info('===========================++++============================+++==================+++++====================++++========')

if __name__ == '__main__':
    train_network = Network()
    train_network.run()