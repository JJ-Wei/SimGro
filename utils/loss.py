import torch
from torch import nn
import torch.nn.functional as F

class Classification_Loss(nn.Module):
 
    def __init__(self, class_num=6, weight=None, reduction='mean'):

        super(Classification_Loss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.class_num = class_num

        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                self.weight = torch.tensor(self.weight, dtype=torch.float32)
            assert self.weight.shape == (class_num,), "the shape of weight must match with the class number"
            self.weight = weight
        else:
            self.weight = None

        self.criterion = nn.CrossEntropyLoss(weight=self.weight, reduction=reduction)

    def forward(self, Pred, Label):
        """
        :param Pred: prediction with shape of (b, class_num)
        :param Label: label with shape pf (b,)
        :return: the classification loss
        """
        loss = self.criterion(Pred, Label)

        return loss
    
class EMDLoss(nn.Module):
    """
    Earth Mover's Distance (EMD) Loss / Wasserstein Loss
    """
    def __init__(self, num_classes=5):
        super(EMDLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.arange(self.num_classes, device=logits.device).unsqueeze(0) >= targets.unsqueeze(1)
        target_cdf = target_cdf.float()
        loss = torch.mean((pred_cdf - target_cdf) ** 2)
        
        return loss

class PrototypeLoss(nn.Module):
    def __init__(
        self,
        num_classes=5,
        num_templates=4,
        tau=0.1,
        epsilon=0.05,
        sinkhorn_iters=10,
        ordinal_sep_alpha=1.0,
        ordinal_sep_gamma=2.0,
        distance_norm="sample",
        intra_proto_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_templates = num_templates
        self.tau = tau
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.ordinal_sep_alpha = ordinal_sep_alpha
        self.ordinal_sep_gamma = ordinal_sep_gamma
        self.distance_norm = distance_norm
        self.intra_proto_weight = intra_proto_weight
        self.ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def get_sinkhorn_assignments(self, features, labels, templates):
        B = features.shape[0]
        assigned_k = torch.zeros(B, dtype=torch.long, device=features.device)
        for c in range(self.num_classes):
            mask = (labels == c)
            N_c = mask.sum().item()
            if N_c == 0: continue
            c_feats = features[mask]
            c_temps = templates[c]
            sim = torch.matmul(c_feats, c_temps.T)
            if N_c < self.num_templates:
                assigned_k[mask] = torch.argmax(sim, dim=1)
                continue
            sim_stable = sim - torch.max(sim, dim=1, keepdim=True)[0]
            Q = torch.exp(sim_stable / self.epsilon)
            Q /= (Q.sum() + 1e-8)
            r = torch.ones(N_c, device=features.device) / N_c
            c_marg = torch.ones(self.num_templates, device=features.device) / self.num_templates
            for _ in range(self.sinkhorn_iters):
                Q *= (r / (Q.sum(dim=1) + 1e-8)).unsqueeze(1)
                Q *= (c_marg / (Q.sum(dim=0) + 1e-8)).unsqueeze(0)
            assigned_k[mask] = torch.argmax(Q, dim=1)
        return assigned_k

    def ordinal_sep_loss(self, sims_flat, labels, target_sep):
        B = sims_flat.shape[0]
        device = sims_flat.device
        logits = sims_flat / self.tau

        proto_classes = torch.arange(
            self.num_classes, device=device
        ).repeat_interleave(self.num_templates)
        class_dist = (labels.unsqueeze(1) - proto_classes.unsqueeze(0)).abs().float()

        if self.distance_norm == "sample":
            left_dist = labels.float()
            right_dist = (self.num_classes - 1 - labels).float()
            denom = torch.maximum(left_dist, right_dist).clamp_min(1.0)
            norm_dist = class_dist / denom.unsqueeze(1)
        elif self.distance_norm == "global":
            denom = max(self.num_classes - 1, 1)
            norm_dist = class_dist / denom
        else:
            raise ValueError(f"Unknown distance_norm: {self.distance_norm}")

        log_weight = self.ordinal_sep_alpha * norm_dist.pow(self.ordinal_sep_gamma)

        row_idx = torch.arange(B, device=device)
        pos_mask = torch.zeros_like(logits, dtype=torch.bool)
        pos_mask[row_idx, target_sep] = True
        log_weight = log_weight.masked_fill(pos_mask, 0.0)

        if self.intra_proto_weight != 1.0:
            same_class_non_pos = (class_dist == 0) & (~pos_mask)
            intra_log_weight = torch.log(
                torch.tensor(max(self.intra_proto_weight, 1e-8), device=device)
            )
            log_weight = log_weight.masked_fill(same_class_non_pos, intra_log_weight)

        adjusted_logits = logits + log_weight
        pos_logits = logits[row_idx, target_sep]
        return (-pos_logits + torch.logsumexp(adjusted_logits, dim=1)).mean()

    def forward(self, features, labels, templates):
        B = features.shape[0]
        labels = labels.view(-1).long()
        
        features = F.normalize(features, p=2, dim=1)
        templates = F.normalize(templates, p=2, dim=2)
        
        assigned_k = self.get_sinkhorn_assignments(features, labels, templates)
        sims_flat = torch.matmul(features, templates.view(-1, features.shape[-1]).T)
        target_sep = labels * self.num_templates + assigned_k
        if self.ordinal_sep_alpha > 0:
            loss_sep = self.ordinal_sep_loss(sims_flat, labels, target_sep)
        else:
            loss_sep = self.ce(sims_flat / self.tau, target_sep)

        assigned_sims = sims_flat[torch.arange(B), target_sep]
        sample_cls_losses = (1 - assigned_sims).pow(2)

        loss_cls = sample_cls_losses.mean()
        
        return loss_sep, loss_cls, assigned_k