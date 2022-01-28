import torch
import torch.nn as nn

class Nllloss(nn.Module):
    def __init__(self):
        super(Nllloss, self).__init__()

    def forward(self, y_pred, y, num_obj):
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_pred = torch.abs(y_pred - y.float())
        ret = torch.sum(y_pred) / (y_pred.shape[0]*y_pred.shape[1])
        return ret