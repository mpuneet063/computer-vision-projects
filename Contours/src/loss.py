# writing custom loss function to train model for contours detection

""""
The basic idea is that standard cross-entropy loss will be efficient in pictures where edge pixels
form only a minority of all pixels. Therefore, a weighted cross-entropy is required
"""
from math import inf
from sympy.multipledispatch.conflict import edge
import torch
import torch.nn as nn
import torch.nn.functional as F 

def cross_entropy_loss_certainty(pred, target):
    """
    Computes class-balanced cross-entropy loss for edge detection, 
    accounting for extreme class imbalance between edge and background pixels
    """
    pred = torch.sigmoid(pred)

    # calculate weight beta based on the ratio of background pixels to total pixels
    # target > 0.5 represents edge pixels, target <= 0.5 represents background
    inf_mask = (target > 0.05).float()

    # count total pixels and edge pixels
    total_pixels = target.numel()   # numel menas number of elements in a tensor
    edge_pixels = torch.sum(inf_mask)
    non_edge_pixels = total_pixels - edge_pixels

    if edge_pixels == 0 or non_edge_pixels == 0:
        # fallback to standard binary cross-entropy if a batch is completely blank
        return F.binary_cross_entropy_with_logits(pred, target)

    beta = non_edge_pixels / total_pixels

    # compute weighted binary cross-entropy elements
    # weight applied to positive (edge) class is beta, negative class is (1-beta)

    weight_factor = inf_mask * beta + (1-inf_mask)*(1-beta)

    loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight_factor)

    return loss

def hed_loss(outputs, targets):
    """
    Computes total deep supervision loss for HED.
    outputs: list containing 5 side output tensors + 1 fused tensor 
    targets: ground truth edge map tensor
    """

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    loss = 0.0
    # calculate loss for each of the 5 side-output layeres
    for d in outputs[:-1]:
        loss += cross_entropy_loss_certainty(d, targets)

    # calculate loss for each of the 5 side-output layers
    fuse_output = outputs[-1]
    loss += cross_entropy_loss_certainty(fuse_output, targets)

    return loss