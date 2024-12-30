import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DAKGNN import *
from model.contrastive_learning import *
from torch.nn import Module

class MR_DAKGNN(nn.Module):
    def __init__(self, bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE):
        super(MR_DAKGNN, self).__init__()
        self.scale = scale
        self.AGCN = DAKGNN(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, DEVICE)
        self.Linear1 = torch.nn.Linear(5 *2 * higru_out, 5)
        self.CL = CL(temperature, DEVICE)
        #self.ww = torch.nn.Parameter(torch.rand(5, 1).to(DEVICE), requires_grad=True)

    def forward(self,d1,d2,d3,d4,a4,label):
        aa4 = self.AGCN(a4)
        dd4 = self.AGCN(d4)
        dd3 = self.AGCN(d3)
        dd2 = self.AGCN(d2)
        dd1 = self.AGCN(d1)

        yy = torch.cat((aa4,dd4,dd3,dd2,dd1),dim=-1)

        #w = nn.Softmax(dim=0)(self.ww)
        #yy = 0.2*(aa4+ dd4+ dd3+ dd2+ dd1)
        result = self.Linear1(yy)

        model_loss = nn.CrossEntropyLoss()
        loss1 = model_loss(result,label)

        ori_a4 = aa4.unsqueeze(dim=1)
        ori_d4 = dd4.unsqueeze(dim=1)
        ori_d3 = dd3.unsqueeze(dim=1)
        ori_d2 = dd2.unsqueeze(dim=1)
        ori_d1 = dd1.unsqueeze(dim=1)

        ori_a4d4 = torch.cat((ori_a4, ori_d4), dim=1)
        ori_a4d3 = torch.cat((ori_a4, ori_d3), dim=1)
        ori_a4d2 = torch.cat((ori_a4, ori_d2), dim=1)
        ori_a4d1 = torch.cat((ori_a4, ori_d1), dim=1)

        ori_d4d3 = torch.cat((ori_d4, ori_d3), dim=1)
        ori_d4d2 = torch.cat((ori_d4, ori_d2), dim=1)
        ori_d4d1 = torch.cat((ori_d4, ori_d1), dim=1)

        ori_d3d2 = torch.cat((ori_d3, ori_d2), dim=1)
        ori_d3d1 = torch.cat((ori_d3, ori_d1), dim=1)

        ori_d2d1 = torch.cat((ori_d2, ori_d1), dim=1)

        loss_a4d4 = self.CL(ori_a4d4,label)
        loss_a4d3 = self.CL(ori_a4d3,label)
        loss_a4d2 = self.CL(ori_a4d2,label)
        loss_a4d1 = self.CL(ori_a4d1,label)

        loss_d4d3 = self.CL(ori_d4d3,label)
        loss_d4d2 = self.CL(ori_d4d2,label)
        loss_d4d1 = self.CL(ori_d4d1,label)

        loss_d3d2 = self.CL(ori_d3d2,label)
        loss_d3d1 = self.CL(ori_d3d1,label)

        loss_d2d1 = self.CL(ori_d2d1,label)

        loss2 = loss_a4d4 + loss_a4d3 + loss_a4d2 + loss_a4d1 + loss_d4d3 + loss_d4d2 + loss_d4d1 + loss_d3d2 + loss_d3d1 + loss_d2d1


        #print(loss2)
        loss = loss1  + self.scale * loss2

        return result, loss

def make_model(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE):
    model = MR_DAKGNN(bandwidth, K, in_dim, out_dim, num_of_nodes, higru_hid, higru_out, temperature, scale, DEVICE)
    model.float()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model