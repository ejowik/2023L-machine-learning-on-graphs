import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch.nn import Linear
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


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


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mlp1 = tg.nn.MLP(
            in_channels=1,
            hidden_channels=16,
            out_channels=16,
            num_layers=2,
        )
        self.conv1 = tg.nn.GINEConv(mlp1, edge_dim=1)
        self.pool1 = TopKPooling(16, ratio=0.8)
        mlp2 = tg.nn.MLP(
            in_channels=16,
            hidden_channels=16,
            out_channels=16,
            num_layers=2,
        )
        self.conv2 = tg.nn.GINEConv(mlp2, edge_dim=1)
        self.pool2 = TopKPooling(16, ratio=0.8)
        mlp3 = tg.nn.MLP(
            in_channels=16,
            hidden_channels=16,
            out_channels=16,
            num_layers=2,
        )
        self.conv3 = tg.nn.GINEConv(mlp3, edge_dim=1)
        self.pool3 = TopKPooling(16, ratio=0.8)
        mlp4 = tg.nn.MLP(
            in_channels=16,
            hidden_channels=16,
            out_channels=16,
            num_layers=2,
        )
        self.conv4 = tg.nn.GINEConv(mlp4, edge_dim=1)
        self.pool4 = TopKPooling(16, ratio=0.8)

        self.lin1 = torch.nn.Linear(32, 16)
        self.lin2 = torch.nn.Linear(16, 16)
        self.lin3 = torch.nn.Linear(16, 5)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, edge_attr, batch
        )
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr, batch
        )
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(
            x, edge_index, edge_attr, batch
        )
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(
            x, edge_index, edge_attr, batch
        )
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        print(x.shape)

        return x
