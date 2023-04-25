from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import ts2vg


def csv_to_visibility_graph(
    path: str = "../data/PCEC96_perc_diff_lag_1.csv",
    y_col: str = "PCEC96_perc_diff_lag_1",
    time_col: str = "reference_date",
    add_node_features: bool = True,
) -> nx.Graph:
    """Generate a visibility graph from a given CSV file.

    Args:
        path (str): The path to the CSV file.
        y_col (str): The name of the column containing the time series data.
        time_col (str): The name of the column containing the timestamps.
        add_node_features (bool): Whether to add node features to the graph.

    Returns:
        A NetworkX Graph object representing the visibility graph.
    """
    cols = None if add_node_features else [y_col, time_col]
    data = pd.read_csv(path, usecols=cols)
    timestamps = pd.to_datetime(data[time_col]).apply(lambda x: x.timestamp())
    vs = ts2vg.NaturalVG()
    vis_graph = (
        vs.build(data[y_col].to_numpy(), timestamps).as_networkx().to_undirected()
    )
    nx.set_node_attributes(vis_graph, values=data[time_col].to_dict(), name=time_col)
    if add_node_features:
        nx.set_node_attributes(
            vis_graph,
            values=data.drop(columns=[y_col, time_col]).to_dict(orient="index"),
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


def csv_to_quantile_graph(
    path: str = "../data/PCEC96_perc_diff_lag_1.csv",
    y_col: str = "PCEC96_perc_diff_lag_1",
    n_quantiles: int = 5,
) -> nx.DiGraph:
    """Generate a quantile graph (see "Novel features for time series analysis: a complex networks approach") from a given CSV file.

    Args:
        path (str): Path to the CSV file. Defaults to "../data/PCEC96_perc_diff_lag_1.csv".
        y_col (str): Name of the column containing the values to be discretized. Defaults to "PCEC96_perc_diff_lag_1".
        n_quantiles (int): Number of quantiles to use. Defaults to 5.

    Returns:
        nx.DiGraph: A directed graph where each node represents a quantile and edges represent transitions between
        quantiles. The weight of each edge represents the probability of transitioning from one quantile to another.
    """
    data = pd.read_csv(path, usecols=[y_col])
    fractions = np.linspace(0, 1, num=n_quantiles + 1)
    quantiles = np.quantile(data[y_col], fractions, interpolation="midpoint")
    quantilized_ts = np.clip(
        np.searchsorted(quantiles, data[y_col], side="right") / n_quantiles,
        a_min=None,
        a_max=1,
    )
    trans_probs = generate_probabilities_from_random_walk(quantilized_ts)
    quantile_graph = nx.DiGraph()
    quantile_graph.add_weighted_edges_from(trans_probs)
    return quantile_graph
