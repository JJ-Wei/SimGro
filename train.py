import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import argparse
import time
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from utils.loss import Classification_Loss, EMDLoss, PrototypeLoss
from utils.config import get_config
from models.models import get_model
from utils.evaluation import get_eval
from utils.dataset import ImageToImage3D, JointTransform3D

class CombinedLoss(nn.Module):
    def __init__(self, criterion1, criterion2, emd_weight=2.0):
        super(CombinedLoss, self).__init__()
        self.loss1 = criterion1
        self.loss2 = criterion2
        self.weight = emd_weight

    def forward(self, logits, target):
        loss_1 = self.loss1(logits, target)
        loss_2 = self.loss2(logits, target)
        return loss_1 + self.weight * loss_2

def add_model_subdir(path, modelname):
    return os.path.join(path, modelname) + os.sep


def main():
    parser = argparse.ArgumentParser(description='Training script for the model')
    parser.add_argument('--task', default='Nodule', help='task or dataset name')
    parser.add_argument('--scheduler', default='None', help='scheduler name')
    parser.add_argument('--n_gpu', type=int, default=1, help='total GPU count')
    parser.add_argument('--ce_weight', type=int, default=1)
    parser.add_argument('--emd_weight', type=float, default=10)
    parser.add_argument('--weight_sep', type=float, default=0.4)
    parser.add_argument('--weight_cls', type=float, default=20)
    parser.add_argument('--fold', type=int, default=0, help='fold index (0-4)')
    parser.add_argument('--updaterate', type=int, default=1)
    parser.add_argument('--warm_up', type=int, default=30, help='the number of warm up epochs before template update')
    parser.add_argument('--tau', type=float, default=0.1, help='the temperature for Sinkhorn algorithm')

    args = parser.parse_args()

    opt = get_config(args.task)
    opt.save_path_code = "_"
    opt.save_path = add_model_subdir(opt.save_path, opt.modelname)

    device = torch.device(opt.device)

    timestr = time.strftime('%m%d%H%M')
    boardpath = opt.tensorboard_path + opt.modelname + opt.save_path_code + timestr
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)
    TensorWriter = SummaryWriter(boardpath)

    seed_value = 300
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

    tf_train = JointTransform3D()
    tf_val = JointTransform3D(p_per_sample=0.0)

    train_dataset = ImageToImage3D(opt.data_path, opt.train_split, tf_train, opt.classes, opt.img_size, fold=args.fold)
    val_dataset = ImageToImage3D(opt.data_path, opt.val_split, tf_val, opt.classes, opt.img_size, fold=args.fold)

    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    model = get_model(modelname=opt.modelname, img_size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes,num_templates=opt.num_templates, feature_dim= opt.feature_dim)
    model.to(device)

    if opt.pre_trained:
        model.load_state_dict(torch.load(opt.load_path, map_location=device))
    
    if device.type == "cuda" and args.n_gpu > 1:
        model = nn.DataParallel(model)

    CE_criterion = Classification_Loss(class_num=opt.classes).to(device)
    EMD_criterion = EMDLoss(num_classes=opt.classes).to(device)
    val_criterion = CombinedLoss(CE_criterion, EMD_criterion, emd_weight=args.emd_weight)
    proto_criterion =PrototypeLoss(num_classes=opt.classes,num_templates=opt.num_templates, tau=args.tau).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=opt.learning_rate, weight_decay=1e-5)

    if args.scheduler == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(opt.epochs * 2/3), eta_min=3e-5)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
    elif args.scheduler == "Poly":
        exponent = 0.9
        lr_lambda = lambda epoch: (1 - epoch / opt.epochs) ** exponent
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    #  ========================================== begin to train the model =============================================
    warmup_epochs =  args.warm_up  
    best_acc, loss_log = 0.0, np.zeros(opt.epochs+1)
    update_epochs = args.updaterate
    
    epoch_candidates_features = []
    epoch_candidates_labels = []
    epoch_candidates_ids = []
    epoch_candidates_assigned_k = []
    templates_ready = False
    for epoch in range(opt.epochs):
        #  ------------------------------------ training ------------------------------------
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(seed_value + epoch)
        trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True,worker_init_fn=seed_worker,generator=g)
        model.train()
        train_losses = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        for batch_idx, (batch_data) in enumerate(pbar):
            current_data = batch_data['current_data'].to(opt.device, non_blocking=True) # (b, 1, 48, 96, 96)
            gt_nodule_class = batch_data['nodule_class'].to(opt.device, non_blocking=True) # (b, 1)
            batch_patient_ids = batch_data['keys']
            # ---------------------------------- forward ----------------------------------
            if isinstance(batch_patient_ids, list) and len(batch_patient_ids) == 1 and isinstance(batch_patient_ids[0], tuple):
                real_ids = batch_patient_ids[0]
            else:
                real_ids = batch_patient_ids
            # ---------------------------------- forward ----------------------------------
            pred_nodules, features = model(current_data)
                
            ce_loss = CE_criterion(pred_nodules, gt_nodule_class)
            emd_loss = EMD_criterion(pred_nodules, gt_nodule_class)
            train_loss = args.ce_weight * ce_loss + args.emd_weight * emd_loss

            log_dict = {
                "train/ce_loss": ce_loss.item(),
                "train/emd_loss": emd_loss.item()
            }

            
            if epoch >= warmup_epochs:
                real_model = model.module if isinstance(model, nn.DataParallel) else model
                current_templates = real_model.templates.detach()
                loss_sep, loss_cls, assigned_k = proto_criterion(features, gt_nodule_class, current_templates)
                if templates_ready:
                    train_loss += args.weight_sep * loss_sep + args.weight_cls * loss_cls
                    log_dict["train/loss_sep"] = loss_sep.item()
                    log_dict["train/loss_cls"] = loss_cls.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch >= warmup_epochs:
                epoch_candidates_features.append(features.detach().cpu())
                epoch_candidates_labels.append(gt_nodule_class.detach().cpu())
                epoch_candidates_ids.extend(real_ids)
                epoch_candidates_assigned_k.append(assigned_k.detach().cpu())

            train_losses += train_loss.item()
            log_dict['train/train_loss'] = train_loss.item()
            for name, value in log_dict.items():
                TensorWriter.add_scalar(name, value, epoch * len(trainloader) + batch_idx)
            pbar.set_postfix({'loss': train_loss.item()})

        if epoch >= warmup_epochs and epoch % update_epochs == 0:
            templates_ready = True
            
            all_feats = torch.cat(epoch_candidates_features, dim=0)
            all_labels = torch.cat(epoch_candidates_labels, dim=0).view(-1)
            all_assigned_k = torch.cat(epoch_candidates_assigned_k, dim=0)
            real_model = model.module if isinstance(model, nn.DataParallel) else model
            
            for c in range(opt.classes):
                c_mask = (all_labels == c)
                if not c_mask.any():
                    continue
                    
                c_feats = all_feats[c_mask] # 当前类的所有真实特征
                c_assigned = all_assigned_k[c_mask]
                c_ids = [epoch_candidates_ids[i] for i, m in enumerate(c_mask.tolist()) if m]
                
                virtual_centers = []
                for k in range(opt.num_templates):
                    k_mask = (c_assigned == k)
                    if k_mask.any():
                        center = F.normalize(c_feats[k_mask].mean(dim=0), p=2, dim=0)
                        virtual_centers.append(center)
                    else:
                        if len(virtual_centers) > 0:
                            current_active = torch.stack(virtual_centers)
                            dists = torch.cdist(c_feats, current_active)
                            min_dists, _ = torch.min(dists, dim=1)
                            farthest_idx = torch.argmax(min_dists).item()
                            wakeup_feat = c_feats[farthest_idx]
                        else:
                            random_idx = torch.randint(0, c_feats.size(0), (1,)).item()
                            wakeup_feat = c_feats[random_idx]

                        wakeup_feat = F.normalize(wakeup_feat, p=2, dim=0)
                        virtual_centers.append(wakeup_feat)

                virtual_centers = torch.stack(virtual_centers)
                dists_matrix = torch.cdist(virtual_centers, c_feats)
                chosen_indices = []

                for k in range(opt.num_templates):
                    nearest_idx = torch.argmin(dists_matrix[k]).item()
                    chosen_indices.append(nearest_idx)
                    dists_matrix[:, nearest_idx] = float('inf')

                for k, idx in enumerate(chosen_indices):
                    new_feat = c_feats[idx].to(device)
                    real_model.templates.data[c, k] = new_feat
                    real_model.current_template_id_map[(c, k)] = c_ids[idx]

                print(f"  -> Class {c} templates updated: {[c_ids[i] for i in chosen_indices]}")

            epoch_candidates_features = []
            epoch_candidates_labels = []
            epoch_candidates_ids = []
            epoch_candidates_assigned_k = []
                                


        if args.scheduler == "CosineAnnealing" or args.scheduler == "Poly":
            scheduler.step()

        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        current_lr = optimizer.param_groups[0]['lr']
        loss_log[epoch] = train_losses / (batch_idx + 1)
        TensorWriter.add_scalar('train/avg_loss', loss_log[epoch], epoch)
        TensorWriter.add_scalar('train/learning_rate', current_lr, epoch)

        if epoch % opt.eval_freq == 0:
            mean_acc, acc, nodule_acc, val_mae, val_losses = get_eval(valloader, model, val_criterion, opt)
            if args.scheduler == "ReduceLROnPlateau":
                scheduler.step(val_losses)

            real_model = model.module if isinstance(model, nn.DataParallel) else model
            with torch.no_grad():
                temps = F.normalize(real_model.templates.detach(), p=2, dim=2)
                sim_log_dict = {}

                for c in range(opt.classes):
                    c_temps = temps[c]
                    sim_matrix = torch.matmul(c_temps, c_temps.T)

                    if opt.num_templates > 1:
                        mask = torch.eye(opt.num_templates, dtype=torch.bool, device=device)
                        sim_matrix.masked_fill_(mask, 0.0)
                        avg_sim = sim_matrix.sum().item() / (opt.num_templates * (opt.num_templates - 1))
                        sim_log_dict[f"template_sim/class_{c}"] = avg_sim

                for name, value in sim_log_dict.items():
                    TensorWriter.add_scalar(name, value, epoch)

            print(f'Epoch [{epoch}/{opt.epochs}] Val Loss: {val_losses:.4f} | Acc: {mean_acc:.4f} | MAE: {val_mae:.4f} | Nodule Acc: {nodule_acc:.4f}')
            print("acc of each class:", acc)
            TensorWriter.add_scalar('val/loss', val_losses, epoch)
            TensorWriter.add_scalar('val/acc', mean_acc, epoch)
            TensorWriter.add_scalar('val/nodule_acc', nodule_acc, epoch)
            TensorWriter.add_scalar('val/mae', val_mae, epoch)

            if mean_acc > best_acc:
                best_acc = mean_acc
                save_dir = opt.save_path
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"{opt.modelname}{opt.save_path_code}_{epoch}_{best_acc:.4f}")
                real_model = model.module if isinstance(model, nn.DataParallel) else model
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'template_id_map': real_model.current_template_id_map,
                }
                torch.save(save_dict, save_path + ".pth", _use_new_zipfile_serialization=False)

        if epoch % opt.save_freq == 0 or epoch == (opt.epochs - 1):
            save_dir = opt.save_path
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"{opt.modelname}{opt.save_path_code}_{epoch}")
            real_model = model.module if isinstance(model, nn.DataParallel) else model
            save_dict = {
                'model_state_dict': model.state_dict(),
                'template_id_map': real_model.current_template_id_map,
            }
            torch.save(save_dict, save_path + ".pth", _use_new_zipfile_serialization=False)

    TensorWriter.close()


if __name__ == '__main__':
    main()

