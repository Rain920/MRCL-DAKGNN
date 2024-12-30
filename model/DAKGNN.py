import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model.Graph_Construction import *
from model.AKGCN import *
from model.HIGRU import *
from torch.nn import Module

class DAKGNN(nn.Module):
    def __init__(self, bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, DEVICE):
        super(DAKGNN, self).__init__()
        self.DEVICE = DEVICE
        self.graph_construction = graph_construction(bandwidth, num_of_nodes, DEVICE)
        self.AKGCN = AKGCN(K, in_dim, out_dim, num_of_nodes, DEVICE)
        self.HIGRU = HIGRU(out_dim, num_of_nodes, higru_hid, higru_out, DEVICE)

    def forward(self, x):
        # 图构建，用中间时间片构建图
        graph_x = x[:, int(int(x.shape[1]) / 2), :, :]
        adj = self.graph_construction(graph_x)
        #自适应核图卷积
        akgcn = self.AKGCN(x, adj)
        #reshape
        higru_in = akgcn.flatten(start_dim=-2, end_dim=-1)
        #分层GRU
        result = self.HIGRU(higru_in)
        return result