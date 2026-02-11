"""Classification metrics."""

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy: fraction of correct predictions."""
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
) -> float:
    """Top-k accuracy: target in top-k predictions."""
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    return correct[:, :k].any(dim=1).float().mean().item()
