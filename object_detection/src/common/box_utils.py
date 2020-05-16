"""
https://github.com/amdegroot/ssd.pytorch
のbox_utils.pyより使用
関数matchを行うファイル

本章の実装はGitHub：amdegroot/ssd.pytorch [4] を参考にしています。
MIT License
Copyright (c) 2017 Max deGroot, Ellis Brown

"""

import numpy as np
# import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (ndarray) center-size default boxes from priorbox layers.
    Return:
        boxes: (ndarray) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    # return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     # boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax
    return np.concatenate(
            (boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
             boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (ndarray) point_form boxes
    Return:
        boxes: (ndarray) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    # return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     # boxes[:, 2:] - boxes[:, :2], 1)  # w, h
    return np.concatenate((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (ndarray) bounding boxes, Shape: [A,4].
      box_b: (ndarray) bounding boxes, Shape: [B,4].
    Return:
      (ndarray) intersection area, Shape: [A,B].
    """
    # A = box_a.size(0)
    # B = box_b.size(0)
    # max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       # box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       # box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # inter = torch.clamp((max_xy - min_xy), min=0)
    
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.min([
        np.repeat(np.expand_dims(box_a[:, 2:], 1), B, 1),
        np.repeat(np.expand_dims(box_b[:, 2:], 0), A, 0),
        ], axis=0)
    min_xy = np.max([
        np.repeat(np.expand_dims(box_a[:, :2], 1), B, 1),
        np.repeat(np.expand_dims(box_b[:, :2], 0), A, 0),
        ], axis=0)
    inter = np.clip((max_xy - min_xy), 0, None)

    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (ndarray) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (ndarray) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (ndarray) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    
    # area_a = ((box_a[:, 2]-box_a[:, 0]) *
              # (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    # area_b = ((box_b[:, 2]-box_b[:, 0]) *
              # (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    area_a = np.repeat(np.expand_dims(
                (box_a[:, 2]-box_a[:, 0])*(box_a[:, 3]-box_a[:, 1]), 1),
                inter.shape[1], 1)  # [A,B]
    area_b = np.repeat(np.expand_dims(
                (box_b[:, 2]-box_b[:, 0]) *(box_b[:, 3]-box_b[:, 1]), 0),
                inter.shape[0], 0)  # [A,B]
    
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def box_iou(truths, priors):
    # jaccard index
    iou = jaccard(
        truths,
        priors
    )
    return iou


if __name__ == "__main__":
    # truths = torch.tensor([[10,10,20,30],[10,10,20,30]], dtype=torch.float32)    # xmin, ymin, xmax, ymax
    # priors = torch.tensor([
        # [5,5,15,25],
        # [7,27,12,35],
        # [15,12,25,17],
        # ], dtype=torch.float32)
    truths = np.array([[10,10,20,30]], dtype=np.float32)    # xmin, ymin, xmax, ymax
    priors = np.array([
        [5,5,15,25],
        [7,27,12,35],
        [15,12,25,17],
        ], dtype=np.float32)
    iou = box_iou(
        truths,
        priors
    )
    print(iou)
