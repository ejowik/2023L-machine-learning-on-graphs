import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch.nn import Linear


def get_pool_method(pool_method: str):
    if pool_method == "add":
        return tg.nn.global_add_pool
    elif pool_method == "mean":
        return tg.nn.global_mean_pool
    else:
        return tg.nn.global_max_pool


class GIN(torch.nn.Module):
    def __init__(
        self, in_channels, sizes, num_layers=2, out_channels=5, pool_method="avg"
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for out_size in sizes:
            mlp = tg.nn.MLP(
                in_channels=in_channels,
                hidden_channels=out_size,
                out_channels=out_size,
                num_layers=num_layers,
            )
            self.convs.append(tg.nn.GINEConv(mlp, train_eps=False, edge_dim=1))
            in_channels = out_size

        self.pool = get_pool_method(pool_method)
        self.lin = Linear(in_features=out_size, out_features=out_channels)

    def forward(self, x, edge_index, egde_weights, batch):
        for conv in self.convs:
            x = conv(x, edge_index, egde_weights).relu()

        x = self.pool(x, batch)
        return self.lin(x)


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        sizes,
        out_channels=5,
        pool_method="max",
        aggr_method="mean",
        normalize=True,
    ):
        super().__init__()
        self.pool_method = get_pool_method(pool_method)
        self.convs = torch.nn.ModuleList()
        for out_size in sizes:
            self.convs.append(
                tg.nn.SAGEConv(
                    in_channels=in_channels,
                    out_channels=out_size,
                    aggr=aggr_method,
                    normalize=normalize,
                )
            )
            in_channels = out_size

        self.lin = Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = self.pool_method(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.lin(x)


class GATC(torch.nn.Module):
    def __init__(self, in_channels, sizes, out_channels=5, pool_method="max", heads=1):
        super().__init__()
        self.pool_method = get_pool_method(pool_method)
        self.convs = torch.nn.ModuleList()
        for out_size in sizes:
            self.convs.append(
                tg.nn.GATConv(
                    in_channels=in_channels, out_channels=out_size, heads=heads
                )
            )
            in_channels = out_size

        self.lin = Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = self.pool_method(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.lin(x)
