"""
Model output and loss utilities for Polyp Segmentation.

Provides standardized handling for:
1. Single vs. multi-stage / deep-supervision model outputs.
2. Inference output fusion / primary prediction extraction.
3. Loss computation with auxiliary deep-supervision heads.
"""

from typing import Any, Callable, List, Optional, Tuple, Union
import torch


def split_model_output(
    outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Split model outputs into the primary (finest) prediction and auxiliary maps.

    Convention in polyp segmentation deep supervision (e.g. PraNet, Polyp-PVT, CaraNet):
    - When a model returns a tuple/list, the final element is the primary/finest prediction map,
      and preceding elements are intermediate / coarse supervision maps.
    - When a model returns a single tensor, it is the primary prediction and aux list is empty.

    Args:
        outputs: Tensor or sequence of Tensors returned by model.forward().

    Returns:
        tuple: (primary_pred, list_of_aux_preds)
    """
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 0:
            raise ValueError("Model output sequence is empty.")
        if len(outputs) == 1:
            return outputs[0], []
        return outputs[-1], list(outputs[:-1])
    elif isinstance(outputs, torch.Tensor):
        return outputs, []
    else:
        raise TypeError(f"Unsupported model output type: {type(outputs)}")


def fuse_outputs(
    outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> torch.Tensor:
    """Extract or fuse the definitive inference prediction from model outputs.

    In standard polyp benchmark evaluation, the finest-resolution map (the primary prediction)
    is evaluated against ground truth.

    Args:
        outputs: Raw model output (Tensor or sequence of Tensors).

    Returns:
        torch.Tensor: The primary prediction map for evaluation.
    """
    primary, _ = split_model_output(outputs)
    return primary


def compute_loss_with_aux(
    outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Compute total training loss supporting deep supervision across multi-scale maps.

    Args:
        outputs: Tensor or sequence of Tensors from model(inputs).
        targets: Ground truth segmentation mask tensor.
        loss_fn: Loss function callable, signature fn(pred, target) -> Tensor.
        weights: Optional list of loss weights corresponding to each output tensor.
                 If None, all outputs are weighted equally (1.0).

    Returns:
        torch.Tensor: Scalar loss for backpropagation.
    """
    if isinstance(outputs, (tuple, list)):
        if weights is not None:
            if len(weights) != len(outputs):
                raise ValueError(
                    f"Length mismatch: {len(weights)} weights provided for {len(outputs)} model outputs."
                )
            total_loss = sum(w * loss_fn(out, targets) for w, out in zip(weights, outputs))
        else:
            total_loss = sum(loss_fn(out, targets) for out in outputs)
        return total_loss
    return loss_fn(outputs, targets)
