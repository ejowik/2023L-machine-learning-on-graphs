import json
import os
import pickle
import sys
import time
from typing import List, Tuple

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split


def read_stargazers_dataset(path: str, verbose: bool = True) -> Tuple[List, pd.Series]:
    """
    Function to read the graphs and their labels from github Stargazers dataset.
    Args:
        path (str): path to the dataset

    Returns:
        Tuple[List,pd.Series]: A tuple consisting of a list of NetworkX graphs and a pd.Series of their respective labels
    """
    data_path = os.path.join(path, "git_edges.json")
    target_path = os.path.join(path, "git_target.csv")
    with open(data_path) as f:
        json_content = json.load(f)
    target = pd.read_csv(target_path, index_col="id").squeeze("columns")
    data = [nx.Graph(elem) for key, elem in json_content.items()]
    if verbose:
        print(f"Loaded {len(data)} graphs")
    return data, target


def calculate_measure(func, G, kwargs={}):
    return func(G, **kwargs)


def cross_validate(
    X, y, estimator=LogisticRegression(), verbose: bool = False, **kwargs
):
    results = cross_val_score(estimator, X=X, y=y, **kwargs)
    if verbose:
        print(f"Estimated accuracy:{results.mean():.04f} std:{results.std():.04f}")
    return results


def create_logreg_model(X, y, get_pred: bool = False, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    if get_pred:
        return model.predict_proba(X_test), y_test
    return (model.predict(X_test) == y_test).mean()


class CSL(torch.utils.data.Dataset):
    """
    Circular Skip Link Graphs:
    Source: https://github.com/PurdueMINDS/RelationalPooling/
    """

    def __init__(self, path="../datasets/CSL/"):
        self.name = "CSL"
        self.adj_list = pickle.load(
            open(os.path.join(path, "graphs_Kary_Deterministic_Graphs.pkl"), "rb")
        )
        self.graph_labels = torch.load(
            os.path.join(path, "y_Kary_Deterministic_Graphs.pt")
        )
        self.graph_lists = []

        self.n_samples = len(self.graph_labels)
        self.num_node_type = 1  # 41
        self.num_edge_type = 1  # 164
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing Circular Skip Link Graphs v4 ...")
        for sample in self.adj_list:
            _g = dgl.from_scipy(sample)
            g = dgl.remove_self_loop(_g)
            g.ndata["feat"] = torch.zeros(g.number_of_nodes()).long()

            # adding edge features as generic requirement
            g.edata["feat"] = torch.zeros(g.number_of_edges()).long()

            self.graph_lists.append(g)
        self.num_node_type = self.graph_lists[0].ndata["feat"].size(0)
        self.num_edge_type = self.graph_lists[0].edata["feat"].size(0)
        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


def weighted_mean(alpha: float):
    return lambda x, y: alpha * x + (1 - alpha) * y


def select_by_var(arr: np.array, alpha: float):
    variances_ord = np.var(arr, axis=0).argsort()[::-1]
    return arr[:, variances_ord][:, : int(alpha * arr.shape[1])]


def by_weighted_variance(alpha: float):
    return lambda x, y: np.concatenate(
        (select_by_var(x, alpha), select_by_var(y, 1 - alpha)), axis=1
    )


def project_v_on_w(v, w):
    """
    here x and y are vectors not matrices!!!
    """
    return w * np.vdot(v, w) / np.linalg.norm(w, 2) ** 2


def get_projected_concatenation(
    vectors, keep_first: bool = True, proj_on_sum: bool = True
):
    """
    again applicable if we're having vectors!!!
    """
    x, y = vectors
    v = y if keep_first else x
    proj_w = x + y if proj_on_sum else x if keep_first else y
    return np.concatenate((x + y - v, project_v_on_w(v, proj_w)))


def split(x: np.array):
    half = int(x.size / 2)
    return x[:half], x[half:]


def one_by_projection(keep_skipgram: bool = True, proj_on_sum: bool = True):
    return lambda x, y: np.apply_along_axis(
        lambda z: get_projected_concatenation(split(z), keep_skipgram, proj_on_sum),
        1,
        np.concatenate((x, y), axis=1),
    )


def get_differences_of_projections(vectors):
    x, y = vectors
    return np.concatenate(
        (
            x + y,
            np.linalg.norm(x + y) / np.linalg.norm(y) * (y - project_v_on_w(y, x + y)),
        )
    )


def projection_diff():
    return lambda x, y: np.apply_along_axis(
        lambda z: get_differences_of_projections(split(z)),
        1,
        np.concatenate((x, y), axis=1),
    )


def get_weighted_projections(vectors, alpha):
    x, y = vectors
    w_m = weighted_mean(alpha)
    vecs_w_sum = w_m(x, y) * 2
    return np.concatenate(
        (project_v_on_w(x, vecs_w_sum), project_v_on_w(y, vecs_w_sum))
    )


def weighted_projections(alpha: float):
    return lambda x, y: np.apply_along_axis(
        lambda z: get_weighted_projections(split(z), alpha),
        1,
        np.concatenate((x, y), axis=1),
    )
