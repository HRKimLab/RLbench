import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, y):
        """ Loss function for C51 algorithm
        Args:
            pred: distribution of prediction (already applied softmax, sum=1)
                shape: (B, class_num)
            y: distribution of y (already applied softmax, sum=1)
                shape: (B, class_num)
        Return:
            out, tensor(int)
        """
        return torch.mean(y * (-torch.log(pred + 1e-8)))


class QuantileHuberLoss(nn.Module):
    def __init__(self, kappa=1.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, pred, y, quants_range):
        """ Loss function for QR-DQN algorithm
        Args:
            pred: predicted quantiles
                shape: (B, n_quant)
            y: target quantiles
                shape: (B, n_quant)
        Return:
            out, tensor(int)
        """
        target_quant = y.unsqueeze(-2)
        pred_quant = pred.unsqueeze(-1)
        u = target_quant - pred_quant # (B, n_quant, n_quant)

        weight = torch.abs(u.le(0.).float() - quants_range.view(1, -1, 1)) # (B, n_quant, n_quant)
        loss = F.huber_loss(
            pred_quant, target_quant,
            reduction='none',
            delta=self.kappa
        ) # (B, n_quant, n_quant)
        loss = torch.sum(weight * loss, dim=1).mean(dim=-1)
        loss = loss.mean()
        # TODO
        # Not sure. Original paper suggested to do sum first, and then get mean.
        # Directly getting mean without first summation can reduce the magnitude 
        # of the loss, resulting in more stable training I guess

        return loss
