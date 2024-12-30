import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class HIGRU(nn.Module):
    def __init__(self, out_dim, num_of_nodes, higru_hid, higru_out, DEVICE):
        super(HIGRU, self).__init__()
        self.DEVICE = DEVICE
        self.short_term_gru = nn.GRU(out_dim * num_of_nodes, higru_hid, 1, True, True, 0.0, True)
        self.long_term_gru = nn.GRU(2 * higru_hid, higru_out, 1, True, True, 0.0, True)

    def forward(self, x):
        short_result, _ = self.short_term_gru(x)
        long_input = short_result[:, int(int(x.shape[1]) / 2), :]#取中间时间片特征
        long_input = long_input.unsqueeze(dim=0)#reshape
        long_result, _ = self.long_term_gru(long_input)
        result = long_result.squeeze()
        return result