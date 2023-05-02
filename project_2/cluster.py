"""
@source: https://www.kaggle.com/code/nassehkhodaie/fixing-multicollinearity-by-feature-clustering/notebook
"""
import pandas as pd


class cluster:
    def __init__(self, pairs=None):
        self.pairs = set()
        self.nodes = set()
        self.name = None
        if pairs != None:
            for pair in pairs:
                self.nodes.update([pair[0], pair[1]])
                self.pairs.add(pair)

    def update_with(self, pair, force_update=False):
        if force_update:
            self.nodes.update([pair[0], pair[1]])
            self.pairs.add(pair)
        else:
            if self.can_accept(pair):
                self.nodes.update([pair[0], pair[1]])
                self.pairs.add(pair)
            else:
                raise Exception(
                    f"The pair {pair} can not be added to this cluster because it does not have any shared node with the current cluster nodes."
                )

    def can_accept(self, pair):
        return pair[0] in self.nodes or pair[1] in self.nodes

    def merge_with_cluster(self, cluster2, force_merge=False):
        def merge():
            self.nodes = self.nodes.union(cluster2.nodes)
            self.pairs = self.pairs.union(cluster2.pairs)

        if force_merge:
            merge()
        else:
            if self.nodes.intersection(cluster2.nodes) != set():
                merge()
            else:
                raise Exception(
                    f"The clusters can not be merged because they do not have any common node."
                )
