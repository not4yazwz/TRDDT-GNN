import torch
from torch import nn
import torch.nn.functional as F


class TripartiteAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(TripartiteAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 定义线性变换
        self.W_q = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)

        # 最终输出层
        self.fc = nn.Linear(hidden_dim * num_heads, 1)

    def forward(self, H_a, H_b, H_c):
        N_a, _ = H_a.size()
        N_b, _ = H_b.size()
        N_c, _ = H_c.size()

        # 线性变换并拆分多头(N_a, num_heads, hidden_dim)
        Q = self.W_q(H_a).view(N_a, self.num_heads, self.hidden_dim)
        K_b = self.W_k(H_b).view(N_b, self.num_heads, self.hidden_dim)
        K_c = self.W_k(H_c).view(N_c, self.num_heads, self.hidden_dim)
        V_b = self.W_v(H_b).view(N_b, self.num_heads, self.hidden_dim)
        V_c = self.W_v(H_c).view(N_c, self.num_heads, self.hidden_dim)

        # 计算注意力分数
        scores_ab = torch.einsum('and,bnd->anb', Q, K_b) / (self.hidden_dim ** 0.5)  # (N_a, num_heads, N_b)
        scores_ac = torch.einsum('and,cnd->anc', Q, K_c) / (self.hidden_dim ** 0.5)  # (N_a, num_heads, N_c)
        scores_bc = torch.einsum('bnd,cnd->bnc', K_b, K_c) / (self.hidden_dim ** 0.5)  # (N_b, num_heads, N_c)

        # 合并注意力分数
        scores = scores_ab.unsqueeze(3) + scores_ac.unsqueeze(2) + scores_bc.unsqueeze(0)  # (N_a, num_heads, N_b, N_c)

        # 计算注意力权重
        alpha = F.softmax(scores.view(N_a, self.num_heads, -1), dim=-1).view(N_a, self.num_heads, N_b, N_c)

        # 获取值向量
        V_b_expanded = V_b.unsqueeze(0).unsqueeze(3).expand(N_a, self.num_heads, N_b, N_c, self.hidden_dim)
        V_c_expanded = V_c.unsqueeze(0).unsqueeze(2).expand(N_a, self.num_heads, N_b, N_c, self.hidden_dim)

        # 加权求和
        context = alpha.unsqueeze(-1) * (V_b_expanded + V_c_expanded)  # (N_a, num_heads, N_b, N_c, hidden_dim)

        # 合并多头
        context = context.view(N_a, N_b, N_c, -1)  # (N_a, N_b, N_c, num_heads * hidden_dim)

        # 最终输出
        output = torch.sigmoid(self.fc(context).squeeze(-1))  # (N_a, N_b, N_c)

        return output