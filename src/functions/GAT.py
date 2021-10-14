# https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class NET(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(NET, self).__init__()

        self.gat1 = GAT(in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        out_dim=out_dim,
                        num_heads=num_heads).to("cuda")
        self.gat2 = GAT(in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        out_dim=100,
                        num_heads=num_heads).to("cuda")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, h):
        gat_output = self.gat1(g, h)
        gat_output = self.relu(gat_output)
        gat_output = self.dropout(gat_output)
        gat_output = self.gat2(g, gat_output)
        return gat_output


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')
#
# # https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class GAT(nn.Module):
#     def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
#         super(GAT, self).__init__()
#         self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
#         # Be aware that the input dimension is hidden_dim*num_heads since
#         # multiple head outputs are concatenated together. Also, only
#         # one attention head in the output layer.
#         self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
#
#     def forward(self, h):
#         h = self.layer1(h)
#         h = F.elu(h)
#         h = self.layer2(h)
#         return h
#
# class MultiHeadGATLayer(nn.Module):
#     def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
#         super(MultiHeadGATLayer, self).__init__()
#         self.heads = nn.ModuleList()
#         for i in range(num_heads):
#             self.heads.append(GATLayer(g, in_dim, out_dim))
#         self.merge = merge
#
#     def forward(self, h):
#         head_outs = [attn_head(h) for attn_head in self.heads]
#         if self.merge == 'cat':
#             # concat on the output feature dimension (dim=1)
#             return torch.cat(head_outs, dim=1)
#         else:
#             # merge using average
#             return torch.mean(torch.stack(head_outs))
#
# class GATLayer(nn.Module):
#     def __init__(self, g, in_dim, out_dim):
#         super(GATLayer, self).__init__()
#         self.g = g
#         # equation (1)
#         self.fc = nn.Linear(in_dim, out_dim, bias=False)
#         # equation (2)
#         self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """Reinitialize learnable parameters."""
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.fc.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
#
#     def edge_attention(self, edges):
#         # edge UDF for equation (2)
#         z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
#         a = self.attn_fc(z2)
#         return {'e': F.leaky_relu(a)}
#
#     def message_func(self, edges):
#         # message UDF for equation (3) & (4)
#         return {'z': edges.src['z'], 'e': edges.data['e']}
#
#     def reduce_func(self, nodes):
#         # reduce UDF for equation (3) & (4)
#         # equation (3)
#         alpha = F.softmax(nodes.mailbox['e'], dim=1)
#         # equation (4)
#         h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
#         return {'h': h}
#
#     def forward(self, h):
#         # equation (1)
#         z = self.fc(h)
#         self.g.ndata['z'] = z
#         # equation (2)
#         self.g.apply_edges(self.edge_attention)
#         # equation (3) & (4)
#         self.g.update_all(self.message_func, self.reduce_func)
#         return self.g.ndata.pop('h')