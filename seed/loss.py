"""Spatially Varying Prior (SVP) loss functions."""

import torch
import torch.nn as nn


class SVPLossBase(nn.Module):
    """Base class that resolves per-sample prior probabilities p0."""

    def __init__(self, p0_dict=None, eps=1e-7):
        super().__init__()
        self.p0_dict = p0_dict or {}
        self.eps = eps

    def forward(self, p_pred, y_true, p0=None, p0_key=None):
        raise NotImplementedError

    def _format_inputs(self, p_pred, y_true):
        p_pred = p_pred.float().squeeze()
        y_true = y_true.float().squeeze()
        assert p_pred.shape == y_true.shape, (
            f"p_pred and y_true shapes differ: {p_pred.shape} vs {y_true.shape}"
        )
        return p_pred, y_true

    def _resolve_p0(self, p_pred, p0, p0_key):
        if p0 is not None:
            if torch.is_tensor(p0):
                p0 = p0.float().squeeze()
            else:
                p0 = torch.full_like(p_pred, float(p0))
        elif p0_key is not None:
            if p0_key not in self.p0_dict:
                raise KeyError(f"p0_key '{p0_key}' not in p0_dict")
            p0 = torch.full_like(p_pred, float(self.p0_dict[p0_key]))
        else:
            raise ValueError("Either p0 or p0_key must be provided")

        assert p0.shape == p_pred.shape, (
            f"p0 shape mismatch: p0={p0.shape}, p_pred={p_pred.shape}"
        )
        return p0


class SVP_BalancedFocalLoss(SVPLossBase):
    """Balanced focal loss modulated by a spatially varying prior p0."""

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, p_pred, y_true, p0=None, p0_key=None):
        p_pred, y_true = self._format_inputs(p_pred, y_true)
        p0 = self._resolve_p0(p_pred, p0, p0_key)

        loss_fp = (
            -p0
            * (1 - self.alpha)
            * torch.pow(1 - p_pred, self.gamma)
            * torch.log(p_pred + self.eps)
        )
        loss_fn = (
            -(1 - p0)
            * self.alpha
            * torch.pow(p_pred, self.gamma)
            * torch.log(1 - p_pred + self.eps)
        )
        loss = torch.where(y_true == 1, loss_fp, loss_fn)
        return loss.mean()
