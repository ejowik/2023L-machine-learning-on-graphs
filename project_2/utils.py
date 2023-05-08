from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import ts2vg


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
    nx.set_node_attributes(
        quantile_graph,
        values={nodes[it]: quantiles[it] for it in range(len(nodes))},
        name="x",
    )
    return quantile_graph
