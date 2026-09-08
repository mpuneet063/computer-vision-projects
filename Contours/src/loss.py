# Custom loss function for training the HED contour detector.

"""
Standard cross-entropy is dominated by background in edge detection, because edge
pixels are only a small minority of all pixels. So we use a class-balanced version:
the rare edge pixels get a large weight, the common background pixels a small one.
"""

import torch
import torch.nn.functional as F


def class_balanced_bce(pred, target, edge_threshold=0.3):
    """
    pred:   RAW LOGITS straight from the network. Do NOT apply sigmoid here --
            binary_cross_entropy_with_logits already does it internally.
    target: ground-truth edge map with values in [0, 1].
    """

    # The ground truth starts out as a crisp 0/1 map, but resizing it to 256x256
    # blurs the lines into soft grey halos. Pixels clearly above the threshold count
    # as edges, pixels at exactly 0 count as background, and the ambiguous grey band
    # in between is ignored entirely rather than guessed at.
    edge_mask = (target > edge_threshold).float()
    valid_mask = ((target > edge_threshold) | (target == 0)).float()

    num_edge = edge_mask.sum()
    num_valid = valid_mask.sum()
    num_background = num_valid - num_edge

    if num_edge == 0 or num_background == 0:
        # Fallback for a batch that contains no edges at all.
        return F.binary_cross_entropy_with_logits(pred, edge_mask)

    # beta is the fraction of valid pixels that are background (typically ~0.9).
    # Edge pixels get weight beta, background pixels get weight (1 - beta), so the
    # two classes end up contributing roughly equally to the total loss.
    beta = num_background / num_valid
    weight = (edge_mask * beta + (1.0 - edge_mask) * (1.0 - beta)) * valid_mask

    loss = F.binary_cross_entropy_with_logits(
        pred, edge_mask, weight=weight, reduction="sum"
    )

    return loss / num_valid


def hed_loss(outputs, targets):
    """
    Deep-supervision loss for HED.
    outputs: list of 5 side-output tensors + 1 fused tensor (all raw logits)
    targets: ground-truth edge map tensor
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    loss = 0.0

    # Every side output is trained against the ground truth directly.
    for side_output in outputs[:-1]:
        loss += class_balanced_bce(side_output, targets)

    # ...and so is the fused output.
    loss += class_balanced_bce(outputs[-1], targets)

    return loss