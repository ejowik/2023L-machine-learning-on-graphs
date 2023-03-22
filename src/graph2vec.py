# The code is only a little modification of the code available at https://karateclub.readthedocs.io/en/latest/_modules/karateclub/graph_embedding/graph2vec.html#Graph2Vec

from typing import Callable, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.estimator import Estimator
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing


class OurGraph2Vec(Estimator):
    r"""An implementation of `"Graph2Vec" <https://arxiv.org/abs/1707.05005>`_
    from the MLGWorkshop '17 paper "Graph2Vec: Learning Distributed Representations of Graphs".
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurrence matrix is decomposed in order
    to generate representations for the graphs.

    The procedure assumes that nodes have no string feature present and the WL-hashing
    defaults to the degree centrality. However, if a node feature with the key "feature"
    is supported for the nodes the feature extraction happens based on the values of this key.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        erase_base_features (bool): Erasing the base features. Default is False.
        use_cbow (bool): whether to use PV-DM (cbow-like) or PV-DBOW (skipgram-like) like approach
        window_size (int): when using PV-DM version defines the window size used for prediction
        features_per_wl_iteration (bool): When using nodes ordering, when fitting the dataset, whether to sort first by nodes order or W-L iteration. Default is False
    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
        cbowlike: bool = False,
        window_size: int = 4,
        features_per_wl_iteration: bool = False
        # TODO think on the way of using window_size and how to define subgraphs that are "close"
    ):
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features
        self.cbowlike = int(cbowlike)
        self.window_size = window_size if cbowlike else 0
        self.features_per_wl_iteration = features_per_wl_iteration

    def fit(
        self, graphs: list[nx.classes.graph.Graph], orderings: Union[list, None] = None
    ):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(list of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)

        documents = [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, self.attributed, self.erase_base_features
            )
            for graph in graphs
        ]

        documents = [
            TaggedDocument(
                words=doc.get_graph_features()
                if orderings is None
                else self.arrange_features(doc.get_graph_features(), orderings[i]),
                tags=[str(i)],
            )
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            dm=self.cbowlike,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        self._embedding = [self.model.dv[str(i)] for i, _ in enumerate(documents)]

    def arrange_features(self, features: list[str], ordering: list[int]):
        """Arranges `features` according to given `ordering` of nodes. If self.features_per_wl_iteration=True then
        the 1st level sorting is per iteration otherwise per ordering given, otherwise the levels are switched
        """
        feat_per_node = 1 + self.wl_iterations
        arrangement = (
            [
                feat_per_node * node + i
                for i in range(feat_per_node)
                for node in ordering
            ]
            # [0,2,1]->[0,6,3,1,7,4,2,8,5]
            if self.features_per_wl_iteration
            else [
                feat_per_node * node + i
                for node in ordering
                for i in range(feat_per_node)
            ]
            # [0,2,1]->[0,1,2,6,7,8,3,4,5]
        )
        return [features[it] for it in arrangement]

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Infer the graph embeddings.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()
        graphs = self._check_graphs(graphs)
        documents = [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, self.attributed, self.erase_base_features
            )
            for graph in graphs
        ]

        documents = [doc.get_graph_features() for _, doc in enumerate(documents)]

        embedding = np.array(
            [
                self.model.infer_vector(
                    doc, alpha=self.learning_rate, min_alpha=0.00001, epochs=self.epochs
                )
                for doc in documents
            ]
        )

        return embedding


class Ensemble_G2V(Estimator):
    def __init__(
        self,
        skipgram: OurGraph2Vec = OurGraph2Vec(),
        cbow: OurGraph2Vec = OurGraph2Vec(cbowlike=True),
        weighting_function: Callable = np.add,
    ):
        self.skipgram = skipgram
        self.cbow = cbow
        self.weighting_function = weighting_function

    def fit(
        self, graphs: list[nx.classes.graph.Graph], orderings: Union[list, None] = None
    ):
        try:
            self.skipgram.get_embedding()
        except:
            self.skipgram.fit(graphs, orderings)
        try:
            self.cbow.get_embedding()
        except:
            self.cbow.fit(graphs, orderings)

    def set_weighting_function(self, function):
        self.weighting_function = function

    def get_embedding(self, part: int = 2) -> np.array:
        return self.weighting_function(
            self.skipgram.get_embedding(), self.cbow.get_embedding()
        )

    def infer(self, graphs, part: int = 2) -> np.array:
        return self.weighting_function(
            self.skipgram.infer(graphs), self.cbow.infer(graphs)
        )
