from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy import interp
from sklearn import metrics


class MultiClassificationEvaluator:
    """Class to evaluate multilabel classification predictions"""

    def __init__(self, predictions: pd.DataFrame):
        """MultiClassificationEvaluator constructor

        Args:
            predictions (pd.DataFrame): DataFrame with predictions
               (should contain columns CommentId, method, probability_prediction, binary_prediction, true_label)
        """
        if np.all(
            np.isin(
                [
                    "graph_id",
                    "method",
                    "probability_prediction",
                    "binary_prediction",
                    "true_label",
                ],
                predictions,
            )
        ):
            raise Exception(
                "Not all the required columns are in the given predictions dataframe"
            )
        self.predictions = predictions
        self.methods = predictions["method"].unique()
        self.n_classes = len(predictions["true_label"].unique())

    def calculate_classification_metrics(
        self, if_return_melted_not_pivoted: bool
    ) -> pd.DataFrame:
        """Function to calculate metrics for given predictions

        Following metrics are calculated:
        - accuracy
        - precision
        - recall
        - F1-score
        - AUC

        Args:
            if_return_melted_not_pivoted (bool): id returned dataframe should be in melted or pivoted format

        Returns:
            pd.DataFrame: dataframe with metrics
        """

        metric_df = pd.DataFrame()

        for method in self.methods:
            current_predictions = self.predictions.loc[
                self.predictions.method == method
            ]
            accuracy_score = metrics.accuracy_score(
                current_predictions["true_label"], current_predictions["binary_prediction"]
            )
            precision_score = metrics.precision_score(
                current_predictions["true_label"],
                current_predictions["binary_prediction"],
                average="macro",
            )
            recall_score = metrics.recall_score(
                current_predictions["true_label"],
                current_predictions["binary_prediction"],
                average="macro",
            )
            f1_score = metrics.f1_score(
                current_predictions["true_label"],
                current_predictions["binary_prediction"],
                average="macro",
            )
            roc_auc_score = metrics.roc_auc_score(
                current_predictions["true_label"],
                np.array(
                    [i for i in current_predictions["probability_prediction"].values]
                ),
                multi_class="ovr",
                average="macro",
            )

            metric_df = metric_df.append(
                {
                    "method": method,
                    "accuracy": accuracy_score,
                    "macro_f1": f1_score,
                    "macro_precision": precision_score,
                    "macro_recall": recall_score,
                    "macro_AUC": roc_auc_score,
                },
                ignore_index=True,
            )

        if if_return_melted_not_pivoted:
            metric_df = pd.melt(
                metric_df,
                id_vars=["method"],
                var_name="metric_name",
                value_name="metric_value",
            )
        return metric_df

    def plot_roc_curves(self, width: int, height: int):
        """Function to plot ROC curves

        Args:
            plot_title (str): Title for the plot
        """
        fig = go.Figure()
        
        for method in self.methods:
            current_predictions = self.predictions.loc[
                self.predictions.method == method
            ]
            scores = np.array(
                [i for i in current_predictions["probability_prediction"].values]
            )
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fig = go.Figure()
            for i in range(self.n_classes):
                current_true_labels = np.where(
                    current_predictions["true_label"] == i, 1, 0
                )
                fpr[i], tpr[i], _ = metrics.roc_curve(
                    current_true_labels, scores[:, i], pos_label=1
                )
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])

                fig.add_trace(
                    go.Scatter(
                        x=fpr[i],
                        y=tpr[i],
                        mode="lines",
                        name=f"Curve of class {i} (AUC = {roc_auc[i]:0.3f})",
                    )
                )

            # Compute macro-average
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= self.n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

            fig.add_trace(
                go.Scatter(
                    x=fpr["macro"],
                    y=tpr["macro"],
                    mode="lines",
                    name=f"Macro-average ROC curve (AUC = {roc_auc['macro']:0.3f})",
                )
            )

            fig.update_layout(
                title=f"ROC {method}",
                xaxis_title="FPR",
                yaxis_title="TPR",
                template="plotly_white",
                width=width,
                height=height,
            )
            fig.show()

    def plot_confusion_matrices(self):
        """Plot confusion matrices for given predictions

        Args:
            threshold (float, optional): Threshold for predictions. Defaults to 0.5.
        """

        def pd_confusion_matrix(df):
            return metrics.confusion_matrix(df["true_label"], df["binary_prediction"])

        confusion_matrix_per_group = (
            self.predictions.groupby(["method"])
            .apply(pd_confusion_matrix)
            .reset_index(name="confusion_matrix")
        )

        for index, row in confusion_matrix_per_group.iterrows():
            _ = sns.heatmap(
                row["confusion_matrix"], annot=True, square=True, fmt="g", cmap="YlGnBu"
            )

            title = "\n".join(wrap(row["method"], 20))

            plt.title(title, fontsize=20, y=1.1)
            plt.xlabel("Predicted value", fontsize=14)
            plt.ylabel("Actual value", fontsize=14)
            plt.show()

    def plot_precision_recall(self, width: int, height: int):

        colors = [
            "#574368",
            "#8488d3",
            "#cfd3c1",
            "#f8c868",
            "#8ddb34",
            "#69cfef",
            "#d1b3ff",
            "#ff8e65",
        ]

        for method in self.methods:
            current_predictions = self.predictions.loc[
                self.predictions.method == method
            ]
            scores = np.array(
                [i for i in current_predictions["probability_prediction"].values]
            )

            fig = go.Figure()

            for i in range(self.n_classes):
                current_true_labels = np.where(
                    current_predictions["true_label"] == i, 1, 0
                )
                precision, recall, thresholds = metrics.precision_recall_curve(
                    current_true_labels, scores[:, i], pos_label=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=thresholds,
                        y=recall,
                        mode="lines",
                        name=f"Recall for class {i}",
                        line=dict(color=colors[i % len(colors)]),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=thresholds,
                        y=precision,
                        name=f"Precision for class {i}",
                        line=dict(color=colors[i % len(colors)], dash="dot"),
                    )
                )

            fig.update_layout(
                title_text=f"Precision/Recall {method}",
                template="plotly_white",
                legend_title="<b>Class:</b>",
                xaxis_title="Threshold value",
                yaxis_title="Metric value",
                width=width,
                height=height,
            )

            fig.show()
