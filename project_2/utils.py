import os
from collections import Counter
from os import path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import ts2vg
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_networkx
from tqdm.notebook import tqdm


def df_to_visibility_graph(
    data: pd.DataFrame,
    y_col: str,
) -> nx.Graph:
    """Convert a pandas DataFrame into a visibility graph
    (for now only y_col is transformed into node-features but it could be easily extended in the future)
    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        y_col (str): The name of the column in `data` to be used as the y-values.

    Returns:
        nx.Graph: The visibility graph constructed from the data.
    """
    data = data.copy()
    data["x"] = data[y_col]
    vs = ts2vg.NaturalVG()
    vis_graph = vs.build(data["x"].to_numpy()).as_networkx().to_undirected()
    values_dict = data.loc[:, ["x"]].to_dict(orient="index")
    nx.set_node_attributes(
        vis_graph,
        values=values_dict,
    )
    return vis_graph


def generate_probabilities_from_random_walk(walk: np.array) -> list[tuple]:
    """Generate a list of most probable transition probabilities given a random walk.

    Args:
        walk (numpy.ndarray): A 1D array representing a sequence of states in a random walk.

    Returns:
        List[Tuple]: A list of tuples representing the transition probabilities between adjacent
        states in the input walk. Each tuple has three elements: the initial state, the target state,
        and the probability of transitioning from the initial state to the target state. The
        probabilities are normalized by the number of times the initial state appears in the walk.
    """

    transitions = Counter(zip(walk, walk[1:]))
    transitions_prob = [
        (i, j, transitions[(i, j)] / sum(walk == i)) for i, j in transitions.keys()
    ]
    return transitions_prob


def df_to_quantile_graph(
    data: pd.DataFrame,
    y_col: str,
    n_quantiles: int = 5,
) -> nx.DiGraph:
    """Convert a pandas DataFrame into a quantile graph
    (for now only maximal value in the given quantile is transformed into node feature but it could be extended in the future)
    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        y_col (str): The name of the column in `data` to be used as the y-values.
        n_quantiles (int): the number of quantiles to divide time-series into. Defaults to 5.

    Returns:
        nx.Graph: The quantile graph constructed from the data.
    """
    fractions = np.linspace(0, 1, num=n_quantiles + 1)
    quantiles = np.quantile(data[y_col], fractions, interpolation="midpoint")
    quantilized_ts = np.clip(
        np.searchsorted(quantiles, data[y_col], side="right") / n_quantiles,
        a_min=None,
        a_max=1,
    )
    quantile_graph = nx.DiGraph()
    quantile_graph.add_weighted_edges_from(
        generate_probabilities_from_random_walk(quantilized_ts)
    )
    nodes = sorted(quantile_graph.nodes)
    # print({nodes[it]: quantiles[it] for it in range(len(nodes))})
    nx.set_node_attributes(
        quantile_graph,
        values={nodes[it]: quantiles[it] for it in range(len(nodes))},
        name="x",
    )
    print(quantile_graph.nodes)
    for elem in quantile_graph.nodes:
        print(quantile_graph.nodes[elem]["x"])
    return quantile_graph


class GraphDataset(Dataset):
    def __init__(
        self,
        dirpath,
        dataset,
        train=True,
        quantile: bool = True,
        n_quantiles: int = 100,
        cache: bool = True,
    ):
        traintest = "TRAIN" if train else "TEST"
        subdirname = "quantile_" + str(n_quantiles) if quantile else "visibility"
        self.ts_path = path.join(dirpath, dataset, f"{dataset}_{traintest}.txt")
        X_ts, labels = GraphDataset.readucr(self.ts_path)
        self.X_ts = pd.DataFrame(X_ts.T)
        self.labels = torch.tensor(labels, dtype=int)
        self.quantile = quantile

        self.path = path.join(dirpath, dataset, subdirname, traintest)
        if path.exists(self.path):
            return
        os.makedirs(self.path)
        for idx, col in tqdm(
            enumerate(self.X_ts.columns), total=len(self.X_ts.columns)
        ):
            if quantile:
                G = df_to_quantile_graph(self.X_ts, y_col=col, n_quantiles=n_quantiles)
                torch.save(
                    from_networkx(
                        G, group_edge_attrs=["weight"], group_node_attrs=["x"]
                    ),
                    path.join(self.path, f"{idx}.pt"),
                )
            else:
                G = df_to_visibility_graph(self.X_ts, y_col=col)
                torch.save(
                    from_networkx(G, group_node_attrs=["x"]),
                    path.join(self.path, f"{idx}.pt"),
                )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.load(path.join(self.path, f"{idx}.pt"))
        data.y = self.labels[idx] - 1
        data.x = data.x  # .to(torch.float32)
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr.to(torch.float32)
        # data.x = torch.cat((data.x, torch.full(data.x.shape, idx)), dim=1)
        # print(data.x.shape)
        # # data.edge_attr = (
        # #     (data.weight.unsqueeze(1) * 100).to(torch.float32)
        # #     if self.quantile
        # #     else None
        # # )
        data.n_graph = torch.tensor(idx)
        data.X_ts = torch.tensor(
            GraphDataset.readucr(self.ts_path)[0], dtype=torch.float32
        )[idx].reshape(1, -1)
        return data

    def readucr(filename):
        data = np.loadtxt(filename)
        return data[:, 1:], data[:, 0].astype(int)
