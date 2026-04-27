import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class EGFRGraphNet(torch.nn.Module):
    """Graph Neural Network for molecular property (pIC50) prediction.

    Uses GCN or GAT layers with global pooling and an MLP head.
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Graph convolution layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First layer
        if conv_type == "gat":
            self.convs.append(GATConv(num_node_features, hidden_channels, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            if conv_type == "gat":
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        # MLP head (after pooling: concat mean + max → 2 * hidden)
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.bn_lin = BatchNorm1d(hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling (mean + max concatenation)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP head
        x = self.lin1(x)
        x = self.bn_lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x.squeeze(-1)
