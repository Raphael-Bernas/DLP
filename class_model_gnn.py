import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# Define model
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=nn.ELU()):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(2 * out_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.sigma = sigma

    def forward(self, x, edge_index):
        N = x.size(0)
        h = self.W(x)  # N x out_dim

        # Initialize attention output with very negative values
        attn_output = torch.full((N, N), float('-inf')).to(x.device)

        # Compute attention only for edges in edge_index
        for i, j in edge_index.t():
            attn_input = torch.cat([h[i], h[j]], dim=0)  # 2*out_dim
            attn_output[i, j] = self.a(attn_input).squeeze(0)

        attn_output = self.leaky_relu(attn_output)
        attn_output = self.softmax(attn_output)

        # Update node embedding
        h_prime = torch.matmul(attn_output, h)
        h_prime = self.sigma(h_prime)

        return h_prime

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=1, concat=True):
        super(MultiHeadGAT, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.attention_heads = nn.ModuleList([GATLayer(in_dim, hidden_dim) for _ in range(num_heads)])
        self.out_layer = GATLayer(hidden_dim * num_heads if concat else hidden_dim, out_dim)

    def forward(self, x, adj):
        head_outputs = [head(x, adj) for head in self.attention_heads]

        if self.concat:
            x = torch.cat(head_outputs, dim=1)
        else:
            x = torch.stack(head_outputs, dim=0).mean(dim=0)

        x = self.out_layer(x, adj)

        return x

class MyGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=1, num_inter_layers=1, concat=True, implemented = False, dropout=0.):
        super(MyGAT, self).__init__()
        self.layers = nn.ModuleList()
        if implemented:
            Head_GAT_in = MultiHeadGAT(in_dim, hidden_dim, hidden_dim, num_heads, concat)
            Head_GAT_inter = MultiHeadGAT(hidden_dim * num_heads if concat else hidden_dim, hidden_dim, hidden_dim, num_heads, concat)
            Head_GAT_out = nn.Linear(hidden_dim * num_heads if concat else hidden_dim, out_dim)
        else:
            Head_GAT_in = GATConv(in_dim, hidden_dim, heads=num_heads, concat=concat, dropout=dropout)
            Head_GAT_inter = GATConv(hidden_dim * num_heads if concat else hidden_dim, hidden_dim, heads=num_heads, concat=concat, dropout=dropout)
            Head_GAT_out = nn.Linear(hidden_dim * num_heads if concat else hidden_dim, out_dim)

        # Add initial multi-head GAT layer
        self.layers.append(Head_GAT_in)

        # Add intermediate multi-head GAT layers
        for _ in range(num_inter_layers):
            self.layers.append(Head_GAT_inter)

        # Add final averaging GAT layer
        self.layers.append(Head_GAT_out)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        n = len(self.layers)
        for i in range(n-1):
            x = self.layers[i](x, adj)
            x = self.leaky_relu(x)
        x = self.layers[-1](x).flatten(1)
        return x