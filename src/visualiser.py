"""Visualisation utilities for EDA and model evaluation."""

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings("ignore")


class Visualiser:
    """Handle all visualisation tasks."""

    def __init__(self, style: Optional[str] = None):
        """Initialise the Visualiser class.
        
        Args:
            style: Optional matplotlib style to use. If None, uses seaborn style.
        """
        # Set up matplotlib style
        if style and style in plt.style.available:
            plt.style.use(style)
        else:
            # Use seaborn style if available, otherwise default
            if "seaborn-v0_8" in plt.style.available:
                plt.style.use("seaborn-v0_8")
            else:
                plt.style.use("default")

        # Set up seaborn
        sns.set_theme()

        # Colour palette
        self.colours = px.colors.qualitative.Set3

    def plot_target_distribution(
        self, df: pd.DataFrame, target_col: str, title: str = "Target Distribution"
    ) -> go.Figure:
        """Plot distribution of target variable.
        
        Args:
            df: DataFrame containing the target variable.
            target_col: Name of the target column to plot.
            title: Title for the plot.
            
        Returns:
            Plotly Figure object with the target distribution plot.
        """
        counts = df[target_col].value_counts()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    marker_color=self.colours[: len(counts)],
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Class",
            yaxis_title="Count",
            showlegend=False,
            height=500,
        )

        return fig

    def plot_correlation_heatmap(
        self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """Plot correlation heatmap.
        
        Args:
            df: DataFrame containing numerical features.
            figsize: Tuple specifying figure size (width, height).
            
        Returns:
            Matplotlib Figure object with the correlation heatmap.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            corr_matrix,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()

        return fig

    def plot_confusion_matrix(
        self, cm: np.ndarray, labels: List[str], normalise: bool = True
    ) -> go.Figure:
        """Plot interactive confusion matrix.
        
        Args:
            cm: Confusion matrix as numpy array.
            labels: List of class labels.
            normalise: Whether to normalise the confusion matrix.
            
        Returns:
            Plotly Figure object with the confusion matrix.
        """
        if normalise:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=np.around(cm, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=600,
            width=700,
        )

        return fig

    def plot_feature_importance(
        self, importance: np.ndarray, feature_names: List[str], top_n: int = 20
    ) -> go.Figure:
        """Plot feature importance.
        
        Args:
            importance: Array of feature importance values.
            feature_names: List of feature names corresponding to importance values.
            top_n: Number of top features to display.
            
        Returns:
            Plotly Figure object with the feature importance plot.
        """
        # Sort features by importance
        indices = np.argsort(importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=top_importance,
                    y=top_features,
                    orientation="h",
                    marker_color="lightgreen",
                )
            ]
        )

        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600,
            yaxis=dict(autorange="reversed"),
        )

        return fig

    def plot_pca_2d(
        self, X: pd.DataFrame, y: pd.Series, labels: List[str]
    ) -> go.Figure:
        """Plot 2D PCA visualisation.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            labels: List of class labels.
            
        Returns:
            Plotly Figure object with the 2D PCA plot.
        """
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig = go.Figure()

        for i, label in enumerate(labels):
            mask = y == i
            if mask.sum() > 0:  # Only plot if there are samples for this label
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[mask, 0],
                        y=X_pca[mask, 1],
                        mode="markers",
                        name=label,
                        marker=dict(size=5, opacity=0.7),
                    )
                )

        fig.update_layout(
            title="PCA 2D Visualisation",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)",
            height=600,
        )

        return fig

    def plot_service_distribution(self, df: pd.DataFrame, top_n: int = 30) -> go.Figure:
        """Plot service distribution.
        
        Args:
            df: DataFrame containing service column.
            top_n: Number of top services to display.
            
        Returns:
            Plotly Figure object with the service distribution plot.
        """
        service_counts = df["service"].value_counts().head(top_n)

        # Create colour array for each bar
        colours = px.colors.sequential.Viridis
        colour_indices = np.linspace(0, len(colours) - 1, len(service_counts)).astype(int)
        bar_colours = [colours[i] for i in colour_indices]

        fig = go.Figure(
            data=[
                go.Bar(
                    y=service_counts.index,
                    x=service_counts.values,
                    orientation="h",
                    marker_color=bar_colours,
                )
            ]
        )

        fig.update_layout(
            title=f"Top {top_n} Services by Frequency",
            xaxis_title="Count",
            yaxis_title="Service",
            height=800,
            yaxis=dict(autorange="reversed"),
        )

        return fig

    def plot_metrics_comparison(
        self, metrics_dict: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """Plot comparison of metrics across models.
        
        Args:
            metrics_dict: Dictionary containing model names as keys and metric dictionaries as values.
            
        Returns:
            Plotly Figure object with the metrics comparison plot.
        """
        models = list(metrics_dict.keys())
        metrics = list(next(iter(metrics_dict.values())).keys())

        fig = go.Figure()

        for metric in metrics:
            values = [metrics_dict[model][metric] for model in models]
            fig.add_trace(go.Bar(name=metric, x=models, y=values))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            height=500,
        )

        return fig

    def plot_pca_explained_variance(self, pca) -> go.Figure:
        """Plot explained variance ratio for PCA components.
        
        Args:
            pca: Fitted PCA object.
            
        Returns:
            Plotly Figure object with the explained variance plot.
        """
        components = list(range(1, len(pca.explained_variance_ratio_) + 1))
        
        fig = go.Figure()
        
        # Individual explained variance
        fig.add_trace(go.Bar(
            x=components,
            y=pca.explained_variance_ratio_,
            name="Individual",
            marker_color="lightblue"
        ))
        
        # Cumulative explained variance
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        fig.add_trace(go.Scatter(
            x=components,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative",
            marker_color="red",
            yaxis="y2"
        ))
        
        fig.update_layout(
            title="PCA Explained Variance Ratio",
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance Ratio",
            yaxis2=dict(
                title="Cumulative Explained Variance",
                overlaying="y",
                side="right"
            ),
            height=500
        )
        
        return fig

    def plot_attack_timeline(
        self, df: pd.DataFrame, time_col: str = "duration"
    ) -> go.Figure:
        """Plot attack types over time.
        
        Args:
            df: DataFrame containing attack data.
            time_col: Name of the time column.
            
        Returns:
            Plotly Figure object with the attack timeline plot.
        """
        fig = px.scatter(
            df,
            x=time_col,
            y="labels",
            color="labels",
            title="Attack Distribution Over Connection Duration",
            labels={"labels": "Attack Type", time_col: "Duration (seconds)"},
        )

        fig.update_layout(height=600)
        return fig

    def plot_pie_chart(self, data, title):
        """Plot pie chart for categorical data.
        
        Args:
            data: Series containing categorical data with index as labels and values as counts.
            title: Title for the pie chart.
            
        Returns:
            Plotly Figure object with the pie chart.
        """
        fig = go.Figure(data=[go.Pie(
            labels=data.index,
            values=data.values,
            hole=0.3
        )])
        
        fig.update_layout(
            title=title,
            height=500
        )
        return fig

    def plot_horizontal_bar(self, data, feature_name):
        """Plot horizontal bar chart.
        
        Args:
            data: Series containing data with index as labels and values as counts.
            feature_name: Name of the feature being plotted.
            
        Returns:
            Plotly Figure object with the horizontal bar chart.
        """
        fig = go.Figure(data=[go.Bar(
            x=data.values,
            y=data.index,
            orientation='h',
            marker_color='lightblue'
        )])
        
        fig.update_layout(
            title=f"{feature_name.replace('_', ' ').title()} Distribution",
            xaxis_title="Count",
            yaxis_title=feature_name.replace('_', ' ').title(),
            height=500,
            yaxis=dict(autorange="reversed")
        )
        return fig

    def plot_correlation_clustermap(self, corr_matrix):
        """Plot correlation clustermap using seaborn.
        
        Args:
            corr_matrix: Correlation matrix as DataFrame.
            
        Returns:
            Matplotlib Figure object with the correlation clustermap.
        """
        clustermap = sns.clustermap(
            corr_matrix,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            figsize=(12, 10)
        )
        plt.title("Correlation Clustermap")
        return clustermap.fig

    def plot_correlation_network(self, corr_matrix, threshold=0.5):
        """Plot correlation network graph.
        
        Args:
            corr_matrix: Correlation matrix as DataFrame.
            threshold: Minimum correlation threshold for displaying connections.
            
        Returns:
            Plotly Figure object with the correlation network graph.
        """
        try:
            import networkx as nx
        except ImportError:
            # Fallback to simple table if networkx not available
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        strong_corrs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j], 
                            'Correlation': corr_val
                        })
            
            if strong_corrs:
                import pandas as pd
                df = pd.DataFrame(strong_corrs)
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df[col] for col in df.columns])
                )])
                fig.update_layout(title="Strong Correlations Table")
                return fig
            
            fig = go.Figure()
            fig.add_annotation(text=f"No correlations above {threshold}", 
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges
        for feature in corr_matrix.columns:
            G.add_node(feature)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    G.add_edge(
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        weight=abs(corr_val),
                        correlation=corr_val
                    )
        
        if len(G.edges()) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No correlations above {threshold}",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title="Correlation Network", height=500)
            return fig
        
        # Generate layout and create visualisation
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='black'))
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Correlation Network (threshold={threshold})",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )
        
        return fig