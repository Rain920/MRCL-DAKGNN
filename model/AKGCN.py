import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class AKGCN(nn.Module):
    def __init__(self, K, in_dim, out_dim, num_of_nodes, DEVICE):
        super(AKGCN, self).__init__()
        self.ModuleList = ModuleList([AKGCN_Layer(in_dim=in_dim,
                                                  out_dim=out_dim,
                                                  num_of_nodes=num_of_nodes,
                                                  DEVICE=DEVICE) for _ in range(K)])

    def forward(self, x, adj):
        #进行K层自适应图卷积
        hid = x
        for akgcn in self.ModuleList:
            hid = akgcn(hid, adj)

        #使用残差连接防止过平滑
        res = F.relu(x)
        output = res + torch.sigmoid(hid)
        return output

class AKGCN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_of_nodes, DEVICE):
        super(AKGCN_Layer, self).__init__()
        self.DEVICE = DEVICE
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_of_nodes = num_of_nodes
        self.lambda_ = torch.nn.Parameter(torch.tensor(1.).to(DEVICE), requires_grad=True)
        self.theta = torch.nn.Parameter(torch.rand(in_dim, out_dim).to(DEVICE),requires_grad=True)

    def get_actual_lambda(self):
        return 1 + F.relu(self.lambda_)

    def forward(self, x, adj):
        batch_size, num_of_timesteps, num_of_nodes, in_channels = x.shape
        lambda_ = self.get_actual_lambda()
        x = torch.matmul(x,self.theta)
        outputs = []
        for time_step in range(num_of_timesteps):
            h = x[:, time_step, :, :]
            E = torch.eye(int(num_of_nodes)).to(self.DEVICE)
            E = E.unsqueeze(dim=0)
            E = torch.repeat_interleave(E, batch_size, dim=0)
            v_1 = ((2 * lambda_ - 2) / lambda_) * E
            v_2 = (2 / lambda_) * adj
            v = v_1 + v_2
            v_ = torch.sum(v, dim=-1)
            v_ = v_.unsqueeze(dim=-1)
            v_ = torch.repeat_interleave(v_, num_of_nodes, dim=-1)
            v = torch.div(v, v_)
            h = torch.matmul(v, h)
            outputs.append(h.unsqueeze(dim=1))
        gcn = F.relu(torch.cat(outputs, dim=1))
        return gcn