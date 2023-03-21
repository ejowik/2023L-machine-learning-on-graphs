import sys
import os
import json

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
from typing import Tuple, List

from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.linear_model import LogisticRegression

from IPython.display import Markdown, display
from tqdm.notebook import tqdm

sys.path.append("../")
from src.graph2vec import OurGraph2Vec, Ensemble_G2V


def read_stargazers_dataset(path: str) -> Tuple[List, pd.Series]:
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
    return [nx.Graph(elem) for key, elem in json_content.items()], target


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
