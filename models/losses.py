import torch
import torch.nn as nn

import numpy as np

from config.cfg import cfg
from models.ssim import SSIMLoss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss definition
    """

    def __init__(self, mse_w=0.4, cos_w=0.4, ssim_w=0.2):
        super(ReconstructionLoss, self).__init__()

        self.mse_w = mse_w
        self.cos_w = cos_w
        self.ssim_w = ssim_w

        self.mse_criterion = nn.MSELoss()
        self.cosine_criterion = nn.CosineSimilarity()
        self.ssim_criterion = SSIMLoss()

    def forward(self, pred, gt):
        mse_loss = self.mse_criterion(pred, gt)
        cosine_loss = self.cosine_criterion(pred, gt)
        ssim_criterion = self.cosine_criterion(pred, gt)

        reconstruction_loss = self.mse_w * torch.abs(mse_loss) + self.cos_w * torch.abs(cosine_loss) \
                              + self.ssim_w * torch.abs(ssim_criterion)

        return reconstruction_loss


class ExpectationLoss(nn.Module):
    """
    Expectation Loss definition
    """

    def __init__(self):
        super(ExpectationLoss, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() and cfg['use_gpu'] else 'cpu')
        self.mae = nn.L1Loss()

    def forward(self, probs, cls, gts):
        cls = torch.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float).T).to(self.device)
        return self.mae(torch.mm(probs, cls.float()).view(-1), gts)


class CombinedLoss(nn.Module):
    """
    CombinedLoss = \alpha \|y_i - \hat{y}_i\|^2 + \beta \|\sum_{i} softmax_i\times i - y_i\|^2 + CrossEntropyLoss
    """

    def __init__(self, alpha=1, beta=1):
        super(CombinedLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.mae_criterion = nn.L1Loss()
        self.expectation_criterion = ExpectationLoss()
        self.xent_criterion = nn.CrossEntropyLoss()

    def forward(self, pred_score, gt_score, pred_probs, pred_cls, gt_cls):
        mae_loss = self.mae_criterion(pred_score, gt_score)
        expectation_loss = self.expectation_criterion(pred_probs, pred_cls, gt_score)
        xent_loss = self.xent_criterion(pred_probs, gt_cls)

        return self.alpha * mae_loss + self.beta * expectation_loss + xent_loss
