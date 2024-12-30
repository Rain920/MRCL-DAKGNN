import torch
import torch.nn as nn

class CL(nn.Module):
    def __init__(self, temperature, DEVICE):
        super(CL, self).__init__()
        self.temperature = temperature
        self.DEVICE = DEVICE

    def forward(self, features, labels):
        DEVICE = self.DEVICE
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)  # 开辟连续内存存放标签
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(DEVICE)

        contrast_count = features.shape[1]#创建几个变换后的正样本
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)#n_view维度进行切片，

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(DEVICE),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss