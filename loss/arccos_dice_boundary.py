"""
Arccosine Dice Loss + Boundary (Surface) Loss for Polyp Segmentation.

References:
    Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., & Ben Ayed, I. (2019).
    Boundary Loss for Highly Unbalanced Segmentation.
    Medical Imaging with Deep Learning (MIDL 2019), PMLR 102, pp. 285-296.

Combines two complementary terms:
    L = (2 / pi) * arccos(DSC_w) + alpha * L_Boundary

1. Arccos Dice (region term):
    Same [0, 1] range and monotonicity as ``1 - DSC``, but with a steeper
    gradient near DSC = 1, so training keeps pushing once a prediction is
    already close to the target. DSC is boundary-weighted (PraNet-style
    weight map) and the arccos backward gradient is clipped to ``max_grad``.

2. Boundary loss (distance term):
    mean(prob * SDF) with the normalized signed distance map of the GT
    (> 0 outside, < 0 inside). Overlap-based terms ignore WHERE an error is;
    this term penalizes false positives by their distance to the GT contour,
    so far-away spurious blobs cost more than slight boundary spill.

Tip:
    Kervadec et al. ramp ``alpha`` up during training (small at the start,
    when predictions are still noisy). Pass a scheduled value from the
    training loop, e.g. ``alpha = min(0.5, 0.01 * epoch)``.
"""

from loss import register_loss
import math
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage


def _boundary_weight_map(mask: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    """Up to 6x weight for pixels near the mask boundary, 1x deep inside a flat region."""
    return 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) - mask
    )


def _dice_coefficient(prob: torch.Tensor, mask: torch.Tensor,
                      weight: torch.Tensor, smooth: float) -> torch.Tensor:
    """Soft Dice similarity coefficient per image, weighted by ``weight``."""
    inter = (prob * mask * weight).sum(dim=(1, 2, 3))
    union = ((prob + mask) * weight).sum(dim=(1, 2, 3))
    return (2.0 * inter + smooth) / (union + smooth)


class _SafeArccosFn(torch.autograd.Function):
    """arccos(x) with the backward gradient magnitude clipped to ``max_grad``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, max_grad: float):
        ctx.save_for_backward(x)
        ctx.max_grad = max_grad
        return torch.arccos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        denom = torch.sqrt(torch.clamp(1.0 - x * x, min=1e-8))
        local_grad = torch.clamp(-1.0 / denom, min=-ctx.max_grad, max=ctx.max_grad)
        return grad_output * local_grad, None


def _compute_sdf(mask_np: np.ndarray) -> np.ndarray:
    """Compute normalized signed distance map from a binary ground truth mask."""
    posmask = mask_np > 0.5
    if not posmask.any():
        return np.ones_like(mask_np, dtype=np.float32)
    negmask = ~posmask
    posdis = scipy.ndimage.distance_transform_edt(posmask)
    negdis = scipy.ndimage.distance_transform_edt(negmask)
    sdf = negdis - posdis
    # Normalize to [-1, 1] range
    max_d = max(np.abs(sdf).max(), 1.0)
    return (sdf / max_d).astype(np.float32)


@register_loss('arccos_dice_boundary')
@register_loss('arccos_boundary')
def arccos_dice_boundary_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    smooth: float = 1.0,
    kernel_size: int = 31,
    max_grad: float = 10.0
) -> torch.Tensor:
    """
    Combined Arccos Dice + Boundary Surface Loss.

    Args:
        pred: Predicted logits of shape (B, 1, H, W) or (B, H, W).
        mask: Ground truth mask of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        alpha: Weight multiplier for the boundary distance term (default: 0.5).
        smooth: Laplace smoothing added to the Dice ratio.
        kernel_size: Window used to detect boundary pixels for the weight map.
        max_grad: Gradient-magnitude clip for the arccos backward pass.

    Returns:
        Scalar loss tensor.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.shape != mask.shape:
        mask = mask.view_as(pred)
    mask = mask.float()

    prob = torch.sigmoid(pred)
    b = pred.size(0)

    # 1. Arccos Dice component (DSC is always in (0, 1] when smooth > 0)
    weight = _boundary_weight_map(mask, kernel_size)
    dsc = _dice_coefficient(prob, mask, weight, smooth)
    l_arccos = ((2.0 / math.pi) * _SafeArccosFn.apply(dsc, max_grad)).mean()

    # 2. Boundary distance component
    mask_np = mask.detach().cpu().numpy()
    sdf_maps = []
    for i in range(b):
        m_2d = mask_np[i].squeeze()
        sdf_maps.append(_compute_sdf(m_2d))

    sdf_tensor = torch.from_numpy(np.stack(sdf_maps)).to(pred.device).view_as(prob)
    l_boundary = (prob * sdf_tensor).mean()

    return l_arccos + alpha * l_boundary
