import torch
import torch.nn as nn
import numpy as np


def getCrossEntropyLoss(train_labels, weighted=False, ignore=False):
    if not weighted:
        if not ignore:
            return nn.CrossEntropyLoss()
        elif ignore:
            return nn.CrossEntropyLoss(ignore_index=0)
    if weighted:
        (unique, counts) = np.unique(train_labels, return_counts=True)
        weights = 1 - counts / np.sum(counts)
        weights = torch.FloatTensor(weights)
        # weights[0] = 0
        if torch.cuda.is_available():
            weights = weights.cuda()

        if not ignore:
            return nn.CrossEntropyLoss(weight=weights)
        if ignore:
            return nn.CrossEntropyLoss(weight=weights, ignore_index=0)

