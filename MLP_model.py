import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.dense.linear import Linear


class GNNModel(torch.nn.Module):
    def __init__(self, drug_number, disease_number, target_number, hidden_dim, conv_type):
        super(GNNModel, self).__init__()
        if conv_type == "GCN":
            self.drug_proj = Linear(disease_number * target_number, hidden_dim, weight_initializer='glorot', bias=True)
            self.drug_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.drug_conv2 = GCNConv(hidden_dim, hidden_dim)

            self.disease_proj = Linear(drug_number * target_number, hidden_dim, weight_initializer='glorot', bias=True)
            self.disease_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.disease_conv2 = GCNConv(hidden_dim, hidden_dim)

            self.target_proj = Linear(disease_number * drug_number, hidden_dim, weight_initializer='glorot', bias=True)
            self.target_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.target_conv2 = GCNConv(hidden_dim, hidden_dim)

        if conv_type == "GAT":
            self.drug_proj = Linear(disease_number * target_number, hidden_dim, weight_initializer='glorot', bias=True)
            self.drug_conv1 = GATConv(hidden_dim, hidden_dim)
            self.drug_conv2 = GATConv(hidden_dim, hidden_dim)

            self.disease_proj = Linear(drug_number * target_number, hidden_dim, weight_initializer='glorot', bias=True)
            self.disease_conv1 = GATConv(hidden_dim, hidden_dim)
            self.disease_conv2 = GATConv(hidden_dim, hidden_dim)

            self.target_proj = Linear(disease_number * drug_number, hidden_dim, weight_initializer='glorot', bias=True)
            self.target_conv1 = GATConv(hidden_dim, hidden_dim)
            self.target_conv2 = GATConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, drug_graph, drug_x, disease_graph, disease_x, target_graph, target_x):
        drug_x = drug_x.reshape((124, -1))
        disease_x = disease_x.reshape((177, -1))
        target_x = target_x.reshape((104, -1))

        drug_x = self.drug_proj(drug_x)
        drug_x = torch.relu(self.drug_conv1(drug_x, drug_graph))
        drug_x = self.drug_conv2(drug_x, drug_graph)

        disease_x = self.disease_proj(disease_x)
        disease_x = torch.relu(self.disease_conv1(disease_x, disease_graph))
        disease_x = self.disease_conv2(disease_x, disease_graph)

        target_x = self.target_proj(target_x)
        target_x = torch.relu(self.target_conv1(target_x, target_graph))
        target_x = self.target_conv2(target_x, target_graph)

        drug_x = self.dropout(drug_x)
        disease_x = self.dropout(disease_x)
        target_x = self.dropout(target_x)

        # 计算外积，生成融合特征矩阵 Z
        z = torch.einsum('ik,jk,lk->ijl', drug_x, disease_x, target_x)
        z = torch.relu(z)

        return z
