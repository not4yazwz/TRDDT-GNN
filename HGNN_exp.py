import matplotlib.pyplot as plt
import dhg
import torch
from torch import nn

# draw a graph
g = dhg.random.graph_Gnm(10, 12)
g.draw()

# draw a hypergraph
hg = dhg.random.hypergraph_Gnm(10, 8)
hg.draw()
# show figures
plt.show()

# class HGNNPConv(nn.Module):
#     def __init__(self,):
#         super().__init__()
#         ...
#         self.reset_parameters()
#
#     def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
#         # apply the trainable parameters ``theta`` to the input ``X``
#         X = self.theta(X)
#         # perform vertex->hyperedge->vertex message passing in hypergraph
#         #  with message passing function ``v2v``, which is the combination
#         #  of message passing function ``v2e()`` and ``e2v()``
#         X = hg.v2v(X, aggr="mean")
#         X = F.relu(X)
#         return X