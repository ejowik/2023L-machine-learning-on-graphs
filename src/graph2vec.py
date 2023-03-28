# The code is only a little modification of the code available at https://karateclub.readthedocs.io/en/latest/_modules/karateclub/graph_embedding/graph2vec.html#Graph2Vec

from typing import Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.estimator import Estimator
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing


class ExtendedGraph2Vec(Estimator):
    r"""An extention of the `"Graph2Vec" <https://arxiv.org/abs/1707.05005>` implementation from Karateclub package <https://karateclub.readthedocs.io/en/latest/index.html>.
    Apart from original functionalities allows to use a model based on the PV-DM approach.
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
        use_pv_dm (bool): whether to use PV-DM (cbow-like) or PV-DBOW (skipgram-like) like approach
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
        use_pv_dm: bool = False,
        window_size: int = 4,
        features_per_wl_iteration: bool = False,
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
        self.use_pv_dm = int(use_pv_dm)
        self.window_size = window_size if use_pv_dm else 0
        self.features_per_wl_iteration = features_per_wl_iteration

    def fit(
        self, graphs: list[nx.classes.graph.Graph], orderings: Optional[list] = None
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
            dm=self.use_pv_dm,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        self._embedding = [self.model.dv[str(i)] for i, _ in enumerate(documents)]

    def arrange_features(self, features: list[str], ordering: list[int]):
        """Arranges `features` according to given `ordering` of nodes. Relevant only for PV-DM-like model.
        If self.features_per_wl_iteration=True then the 1st level sorting is per iteration, otherwise per ordering given, otherwise the levels are switched
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
    allowed_weighting_functions = Union[
        Callable[[np.array, np.array], np.array],
        Literal["concatenate", "w_mean", "w_projection", "partial_projection"],
    ]
    """An ensemble Graph2Vec model that utilizes both PV-DBOW and PV-DM architectures.
     Args:
            pv_dbow (ExtendedGraph2Vec): PV-DBOW model to utilize. Defaults to ExtendedGraph2Vec().
            pv_dm (ExtendedGraph2Vec): PV-DM model to utilize. Defaults to ExtendedGraph2Vec(use_pv_dm=True).
            weighting_function (allowed_weighting_functions): Weighting function to use, see set_weighting_function method for details. Defaults to "w_mean".
            alpha (float): See set_weighting_function method for details. Defaults to 0.5.
            keep_dbow (bool): See set_weighting_function method for details. Defaults to True.
    """

    def __init__(
        self,
        pv_dbow: ExtendedGraph2Vec = ExtendedGraph2Vec(),
        pv_dm: ExtendedGraph2Vec = ExtendedGraph2Vec(use_pv_dm=True),
        weighting_function: allowed_weighting_functions = "w_mean",
        alpha: float = 0.5,
        keep_dbow: bool = True,
    ):
        self.pv_dbow = pv_dbow
        self.pv_dm = pv_dm
        self.set_weighting_function(
            weighting_function, alpha=alpha, keep_dbow=keep_dbow
        )

    def fit(
        self, graphs: list[nx.classes.graph.Graph], orderings: Optional[list] = None
    ):
        """
        Fitting a Ensemble G2V model.

        Arg types:
            * **graphs** *(list of NetworkX graphs)* - The graphs to be embedded.
        """
        self.pv_dbow.fit(graphs, None)
        self.pv_dm.fit(graphs, orderings)

    def set_weighting_function(
        self,
        function: allowed_weighting_functions = "w_mean",
        alpha: float = 0.5,
        keep_dbow: bool = True,
    ):
        """Method to change models weighting function.

        Args:
            function: Allowed are:
            - "concatenate", resulting function is a simple concatenation of PV-DBOW and PV-DM embeddings,
            - "w_mean", resulting function is `alpha`*w+(1-`alpha`)*v where w is PV-DBOW embedding and v is PV-DM embedding vector
            - "w_projection", resulting vector is a concatenation of projections of PV-DBOW and PV-DM embeddings onto 2*`alpha`*w+(2-2*`alpha`)*v
            - "partial_projection", resulting vector is a concatenation of PV-DBOW (or PV-DM if keep_dbow=False) embedding and a projection of the second embedding on their sum
            - Callable, any function that takes as an argument 2 np.arrays and returns an numpy array. Defaults to "w_mean".
            alpha (float): Parameter used for weighted mean and weighted projection. Defaults to 0.5.
            keep_dbow (bool): Parameter used for partial projection function. Defaults to True.
        """
        if function == "concatenate":
            self.weighting_function = lambda x, y: np.concatenate([x, y], axis=1)
        elif function == "w_mean":
            self.weighting_function = lambda x, y: alpha * x + (1 - alpha) * y
        elif function == "w_projection":
            self.weighting_function = Ensemble_G2V.weighted_projections(alpha)
        elif function == "partial_projection":
            self.weighting_function = Ensemble_G2V.partial_projection(keep_dbow)
        elif callable(function):
            self.weighting_function = function

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return self.weighting_function(
            self.pv_dbow.get_embedding(), self.pv_dm.get_embedding()
        )

    def infer(self, graphs) -> np.array:
        """Infer the graph embeddings.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return self.weighting_function(
            self.pv_dbow.infer(graphs), self.pv_dm.infer(graphs)
        )

    @staticmethod
    def project_v_on_w(A: np.array, B: np.array) -> np.array:
        """
        here v and w have to be 2D np.arrays of the same size. Returns a np.array of the same shape in which i-th row is
        a projection of i-th row from A on i-th row of B
        """
        dot_products = np.sum(A * B, axis=1)
        arr2_magnitudes = np.sum(B**2, axis=1)
        return (B.T * dot_products / arr2_magnitudes).T

    @staticmethod
    def weighted_projections(
        alpha: float = 0.5,
    ) -> Callable[[np.array, np.array], np.array]:
        """Function factory, creates a weighted projections weighting function.
        Creates a function that returns a concatenation of projections of embedding vectors on their weighted sum.
        Weighted sum is given by a formula:
        2 * `alpha` * v + (2 - 2*`alpha`) * v, where `alpha` is the factory's parameter.

        Args:
            alpha (float): Weight of the PV-DBOW in the weighted sum. Defaults to 0.5.

        Returns:
            Callable[[np.array, np.array], np.array]: weighting function
        """

        def w_function(x, y):
            vecs_w_sum = 2 * alpha * x + (2 - 2 * alpha) * y
            return np.concatenate(
                (
                    Ensemble_G2V.project_v_on_w(x, vecs_w_sum),
                    Ensemble_G2V.project_v_on_w(y, vecs_w_sum),
                ),
                axis=1,
            )

        return w_function

    @staticmethod
    def partial_projection(
        keep_dbow: bool = True,
    ) -> Callable[[np.array, np.array], np.array]:
        """Function factory, creates a partial_projection weighting function.
        Creates a function that returns concatenation of a unchanged embedding vector (by default PV-DBOW)
        and a projection of the second one on their sum

        Args:
            keep_dbow (bool): Whether to keep PV-DBOW or PV-DM embedding unchanged. Defaults to True.

        Returns:
            Callable[[np.array, np.array], np.array]: weighting function
        """

        def w_function(x, y):
            v = x if keep_dbow else y
            return np.concatenate(
                [x + y - v, Ensemble_G2V.project_v_on_w(v, x + y)], axis=1
            )

        return w_function
