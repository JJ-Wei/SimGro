import argparse
import io
import os
import random
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.models import get_model
from utils.config import get_config
from utils.dataset import ImageToImage3D, JointTransform3D
from utils.evaluation import get_eval
from utils.loss import Classification_Loss


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    if isinstance(checkpoint, dict) and "template_id_map" in checkpoint:
        real_model = model.module if isinstance(model, nn.DataParallel) else model
        real_model.current_template_id_map = checkpoint["template_id_map"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate SimGro")
    parser.add_argument("--modelname", default="SimGro", type=str)
    parser.add_argument("--task", default="Nodule")
    parser.add_argument("--fold", type=int, default=0, help="fold index (0-4)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path; defaults to opt.load_path")
    args = parser.parse_args()

    opt = get_config(args.task)
    opt.device = opt.device
    opt.mode = "eval"
    opt.eval_mode = "class"
    checkpoint_path = args.checkpoint or opt.load_path
    device = torch.device(opt.device)

    seed_value = 300
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

    with redirect_stdout(io.StringIO()):
        test_transform = JointTransform3D(p_per_sample=0.0)
        test_dataset = ImageToImage3D(
            opt.data_path,
            opt.test_split,
            test_transform,
            opt.classes,
            opt.img_size,
            fold=args.fold,
        )
        testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

        model = get_model(
            modelname=args.modelname,
            img_size=opt.img_size,
            img_channel=opt.img_channel,
            classes=opt.classes,
            num_templates=opt.num_templates,
            feature_dim=opt.feature_dim,
        ).to(device)
        load_checkpoint(model, checkpoint_path, device)
        model.eval()

        criterion = Classification_Loss(class_num=opt.classes).to(device)
        results = get_eval(testloader, model, criterion=criterion, opt=opt)

    (macro_acc,class_acc,macro_f1,nodule_acc,nodule_precision,nodule_recall,nodule_f1,nodule_specificity,mae) = results

    class_acc_text = ", ".join(f"{value:.4f}" for value in class_acc)
    print("Final test results")
    print(f"Macro accuracy: {macro_acc:.4f}")
    print(f"Per-class accuracy: [{class_acc_text}]")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Nodule accuracy: {nodule_acc:.4f}")
    print(f"Nodule precision: {nodule_precision:.4f}")
    print(f"Nodule recall: {nodule_recall:.4f}")
    print(f"Nodule F1: {nodule_f1:.4f}")
    print(f"Nodule specificity: {nodule_specificity:.4f}")
    print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
