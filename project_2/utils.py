import os
from collections import Counter
from os import path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import ts2vg
from statsmodels.tsa.arima.model import ARIMA
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
    n_quantiles: int,
) -> nx.DiGraph:
    """Convert a pandas DataFrame into a quantile graph
    (for now only maximal value in the given quantile is transformed into node feature but it could be extended in the future)
    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        y_col (str): The name of the column in `data` to be used as the y-values.
        n_quantiles (int): the number of quantiles to divide time-series into.

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
    nx.set_node_attributes(
        quantile_graph,
        values={nodes[it]: quantiles[it] for it in range(len(nodes))},
        name="x",
    )
    return quantile_graph


class GraphDataset(Dataset):
    def __init__(
        self,
        dirpath,
        dataset,
        train=True,
        quantile: bool = True,
        n_quantiles: int = 100,
    ):
        traintest = "TRAIN" if train else "TEST"
        subdirname = "quantile_" + str(n_quantiles) if quantile else "visibility"
        self.ts_path = path.join(dirpath, dataset, f"{dataset}_{traintest}.txt")
        X_ts, labels = GraphDataset.readucr(self.ts_path)
        print(X_ts.shape)
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
            G = (
                df_to_quantile_graph(self.X_ts, y_col=col, n_quantiles=n_quantiles)
                if quantile
                else df_to_visibility_graph(self.X_ts, y_col=col)
            )
            ts_stats = get_ts_stats(self.X_ts[col])
            for name, value in ts_stats.items():
                nx.set_node_attributes(G, values=value, name=name)
            torch.save(
                from_networkx(G, group_node_attrs=["x"] + list(ts_stats.keys())),
                path.join(self.path, f"{idx}.pt"),
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.load(path.join(self.path, f"{idx}.pt"))
        data.y = self.labels[idx] - 1
        # to -1 bo na https://www.timeseriesclassification.com/ klasy zawsze zaczynają numerację od 1 a my chcemy od 0
        data.x = data.x.to(torch.float32)
        if self.quantile:
            data.edge_attr = data.weight.unsqueeze(1).to(torch.float32)
        else:
            data.edge_attr = None
        return data

    def readucr(filename):
        data = np.loadtxt(filename)
        return data[:, 1:], data[:, 0].astype(int)


def get_ts_stats(time_series):
    model = ARIMA(time_series, order=(1, 0, 0))
    model_fit = model.fit()

    summary_statistics = {
        # "Mean": time_series.mean(),
        # "Median": time_series.median(),
        # "Minimum": time_series.min(),
        # "Maximum": time_series.max(),
        # "Standard Deviation": time_series.std(),
        # "Variance": time_series.var(),
        "Skewness": time_series.skew(),
        "Kurtosis": time_series.kurtosis(),
        "Autocorrelation (lag 1)": time_series.autocorr(1),
        "Autocorrelation (lag 12)": time_series.autocorr(12),
        "Autocorrelation (lag 24)": time_series.autocorr(24),
        "Partial Autocorrelation (lag 1)": pd.Series(
            sm.tsa.stattools.pacf(time_series, nlags=1)
        ).iloc[-1],
        "Partial Autocorrelation (lag 12)": pd.Series(
            sm.tsa.stattools.pacf(time_series, nlags=12)
        ).iloc[-1],
        "Partial Autocorrelation (lag 24)": pd.Series(
            sm.tsa.stattools.pacf(time_series, nlags=24)
        ).iloc[-1],
        "Augmented Dickey-Fuller Test (ADF)": sm.tsa.stattools.adfuller(time_series)[0],
        "KPSS Test": sm.tsa.stattools.kpss(time_series)[0],
        "Ljung-Box Q-Statistic": sm.stats.diagnostic.acorr_ljungbox(
            time_series, lags=[1]
        ).loc[1][0],
        "Breusch-Godfrey LM Test (lag 1)": sm.stats.diagnostic.acorr_breusch_godfrey(
            model_fit, nlags=1
        )[0],
        # "Jarque-Bera Test": sm.stats.stattools.jarque_bera(model_fit.resid)[0],
    }

    # Create a DataFrame to display the summary statistics
    return summary_statistics
