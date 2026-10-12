import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def get_classification_matrix(pred: torch.Tensor, gt: torch.Tensor):
    batch_size = gt.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    gt_flat = gt.reshape(batch_size, -1)

    true_positive = (pred_flat * gt_flat).sum(0)
    false_negative = gt_flat.sum(0) - true_positive
    true_negative = ((1 - pred_flat) * (1 - gt_flat)).sum(0)
    false_positive = pred_flat.sum(0) - true_positive
    return true_positive, false_positive, true_negative, false_negative


def eval_class(valloader, model, criterion, opt):
    model.eval()
    losses = []
    predictions = []
    targets = []
    confusion_parts = []

    with torch.no_grad():
        for batch_data in valloader:
            current_data = batch_data['current_data'].to(opt.device, non_blocking=True)
            target = batch_data['nodule_class'].to(opt.device, non_blocking=True).view(-1).long()

            logits = model(current_data)
            loss = criterion(logits, target)
            predicted = torch.argmax(logits, dim=1)

            one_hot_pred = torch.nn.functional.one_hot(predicted, num_classes=opt.classes)
            one_hot_target = torch.nn.functional.one_hot(target, num_classes=opt.classes)
            confusion_parts.append(get_classification_matrix(one_hot_pred, one_hot_target))

            losses.append(float(loss.detach().cpu()))
            predictions.append(predicted.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

    if not predictions:
        raise ValueError('The evaluation dataloader produced no samples.')

    pred_labels = np.concatenate(predictions)
    true_labels = np.concatenate(targets)
    labels = np.arange(opt.classes)

    confusion = confusion_matrix(true_labels, pred_labels, labels=labels)
    class_totals = confusion.sum(axis=1)
    class_acc = np.divide(
        confusion.diagonal(),
        class_totals,
        out=np.zeros(opt.classes, dtype=float),
        where=class_totals != 0,
    )
    macro_acc = float(np.mean(class_acc))
    macro_f1 = float(
        f1_score(true_labels, pred_labels, labels=labels, average='macro', zero_division=0)
    )
    mae = float(np.mean(np.abs(pred_labels - true_labels)))
    mean_loss = float(np.mean(losses))

    true_positive = sum(part[0] for part in confusion_parts).cpu().numpy()
    false_positive = sum(part[1] for part in confusion_parts).cpu().numpy()
    true_negative = sum(part[2] for part in confusion_parts).cpu().numpy()
    false_negative = sum(part[3] for part in confusion_parts).cpu().numpy()

    epsilon = 0.001
    nodule_acc = float(np.sum(pred_labels == true_labels) / (len(true_labels) + epsilon))
    nodule_precision = float(np.mean(true_positive / (true_positive + false_positive + epsilon)))
    nodule_recall = float(np.mean(true_positive / (true_positive + false_negative + epsilon)))
    nodule_f1 = float(
        2 * nodule_precision * nodule_recall
        / (nodule_precision + nodule_recall + epsilon)
    )
    nodule_specificity = float(
        np.mean(true_negative / (true_negative + false_positive + epsilon))
    )

    if opt.mode == 'train':
        return macro_acc, class_acc, nodule_acc, mae, mean_loss

    return (
        macro_acc,
        class_acc,
        macro_f1,
        nodule_acc,
        nodule_precision,
        nodule_recall,
        nodule_f1,
        nodule_specificity,
        mae,
    )


def get_eval(valloader, model, criterion, opt):
    if opt.eval_mode != 'class':
        raise RuntimeError(f'Unsupported evaluation mode: {opt.eval_mode}')
    return eval_class(valloader, model, criterion, opt)
