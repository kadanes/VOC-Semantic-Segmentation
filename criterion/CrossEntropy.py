import torch
import numpy as np

def getCrossEntropyLoss():
    loss = torch.nn.CrossEntropyLoss()
    return loss

def weightedCrossEntropy(train_labels):
    (unique, counts) = np.unique(train_labels, return_counts=True)
    weights = 1 - counts/np.sum(counts)
    weights = torch.FloatTensor(weights)
    # weights[0] = 0
    if torch.cuda.is_available():
        weights = weights.cuda()

    # loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss = torch.nn.CrossEntropyLoss(weights)
    return loss