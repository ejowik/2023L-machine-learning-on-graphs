from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn import metrics


class ClassificationEvaluator:
    """Class to evaluate classification predictions"""

    def __init__(self, predictions: pd.DataFrame):
        """ClassificationEvaluator constructor

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
                current_predictions["true_label"], current_predictions["binary_prediction"]
            )
            recall_score = metrics.recall_score(
                current_predictions["true_label"], current_predictions["binary_prediction"]
            )
            f1_score = metrics.f1_score(
                current_predictions["true_label"], current_predictions["binary_prediction"]
            )
            roc_auc_score = metrics.roc_auc_score(
                current_predictions["true_label"],
                current_predictions["probability_prediction"],
            )

            metric_df = metric_df.append(
                {
                    "method": method,
                    "accuracy": accuracy_score,
                    "f1": f1_score,
                    "precision": precision_score,
                    "recall": recall_score,
                    "AUC": roc_auc_score,
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

    def plot_confusion_matrices_grid(
        self, plot_title: str, cols_number_on_plot: int, threshold: float = 0.5
    ):
        """Plot confusion matrices for given predictions

        Args:
            plot_title (str): Title for the plot
            cols_number_on_plot (int): Numbers of the columns on the facet plot
            threshold (float, optional): Threshold for predictions. Defaults to 0.5.
        """

        def pd_confusion_matrix(df, threshold):
            return metrics.confusion_matrix(
                df["true_label"],
                np.where(
                    df["probability_prediction"] > threshold,
                    1,
                    0,
                ),
            )

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop("data")
            sns.heatmap(
                data["confusion_matrix"].tolist()[0],
                annot=True,
                square=True,
                fmt="g",
                **kwargs,
            )

        confusion_matrix_per_group = (
            self.predictions.groupby(["method"])
            .apply(pd_confusion_matrix, threshold=(threshold))
            .reset_index(name="confusion_matrix")
        )

        conf_matrix_values = np.concatenate(
            np.array(confusion_matrix_per_group["confusion_matrix"])
        ).ravel()

        fg = sns.FacetGrid(
            confusion_matrix_per_group, col="method", col_wrap=cols_number_on_plot
        )
        cbar_ax = fg.fig.add_axes([1., 0.3, 0.02, 0.4])
        fg.map_dataframe(
            draw_heatmap, cbar_ax=cbar_ax, vmin=0, vmax=conf_matrix_values.max()
        )
        fg.set_titles(col_template="{col_name}")
        fg.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fg.set_axis_labels("Predicted", "Actual")
        fg.fig.suptitle(f"{plot_title}", y=1.05)
        fg.fig.show()

    def plot_roc_curves(self, plot_title: str, width: int, height: int):
        """Function to plot ROC curves

        Args:
            plot_title (str): Title for the plot
        """
        fig = go.Figure()
        
        colors = [
            "#574368",
            "#8488d3",
            "#cfd3c1",
            "#f8c868",
            "#8ddb34",
            "#69cfef",
            "#d1b3ff",
            "#ff8e65",
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf'
        ]
        
        for index, method in enumerate(self.methods):

            fpr, tpr, _ = metrics.roc_curve(
                self.predictions.loc[
                    self.predictions["method"] == method, "true_label"
                ],
                self.predictions.loc[
                    self.predictions["method"] == method, "probability_prediction"
                ],
            )
            auc_score = metrics.auc(fpr, tpr)

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{method} (AUC: {auc_score:.2f})",
                    line=dict(color=colors[index % len(colors)])
                )
            )

        fig.update_layout(
            template="plotly_white",
            legend_title="<b>Model:</b>",
            title_text=plot_title,
            xaxis_title="FPR",
            yaxis_title="TPR",
            width=width,
            height=height,
        )
        fig.show()

    def plot_confusion_matrices(self, threshold: float = 0.5):
        """Plot confusion matrices for given predictions

        Args:
            threshold (float, optional): Threshold for predictions. Defaults to 0.5.
        """

        def pd_confusion_matrix(df, threshold):
            return metrics.confusion_matrix(
                df["true_label"],
                np.where(
                    df["probability_prediction"] > threshold,
                    1,
                    0,
                ),
            )

        confusion_matrix_per_group = (
            self.predictions.groupby(["method"])
            .apply(pd_confusion_matrix, threshold=(threshold))
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
        def pd_precision_recall(df):
            precision, recall, thresholds = metrics.precision_recall_curve(
                df["true_label"], df["probability_prediction"]
            )

            return pd.Series(
                [precision, recall, thresholds],
                index=["precision", "recall", "thresholds"],
            )

        precision_recall_per_group = (
            self.predictions.groupby(["method"])
            .apply(pd_precision_recall)
            .reset_index()
        )

        colors = [
            "#574368",
            "#8488d3",
            "#cfd3c1",
            "#f8c868",
            "#8ddb34",
            "#69cfef",
            "#d1b3ff",
            "#ff8e65",
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf'
        ]

        fig = go.Figure()
        for index, row in precision_recall_per_group.iterrows():

            fig.add_trace(
                go.Scatter(
                    x=row.thresholds,
                    y=row.recall,
                    mode="lines",
                    name=f"Recall {row.method}",
                    line=dict(color=colors[index % len(colors)]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=row.thresholds,
                    y=row.precision,
                    name=f"Precision {row.method}",
                    line=dict(color=colors[index % len(colors)], dash="dot"),
                )
            )

        fig.update_layout(
            title_text="Precision/Recall curves",
            template="plotly_white",
            legend_title="<b>Model:</b>",
            xaxis_title="Threshold value",
            yaxis_title="Metric value",
            width=width,
            height=height,
        )

        fig.show()
