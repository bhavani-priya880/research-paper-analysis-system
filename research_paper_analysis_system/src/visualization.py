"""
Interactive and Academic Visualizations module.
Implements Plotly charts and Matplotlib figures for:
- Word Frequency analysis
- TF-IDF term weights
- Pairwise document similarity heatmaps
- Multi-model performance comparisons (Accuracy, Macro Precision, Recall, F1)
- Confusion matrix heatmaps
- Batch domain prediction distributions
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Safe Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_word_frequency(
    freq_data: List[Tuple[str, int]],
    title: str = "Top Word Frequencies"
):
    """Generates horizontal bar chart of top terms."""
    if not freq_data:
        return None

    terms = [item[0] for item in freq_data][::-1]
    counts = [item[1] for item in freq_data][::-1]

    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Bar(
            x=counts,
            y=terms,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale="Viridis",
                showscale=False
            ),
            hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Scientific Term",
            template="plotly_white",
            height=max(400, len(terms) * 22),
            margin=dict(l=150, r=30, t=50, b=50)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, max(4, len(terms) * 0.3)))
        ax.barh(terms, counts, color="#2E86AB")
        ax.set_title(title)
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        return fig


def plot_tfidf_terms(
    terms_list: List[Dict[str, Any]],
    doc_title: str = "Document"
):
    """Generates bar chart of TF-IDF term importance weights."""
    if not terms_list:
        return None

    terms = [item["term"] for item in terms_list][::-1]
    scores = [item["score"] for item in terms_list][::-1]

    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Bar(
            x=scores,
            y=terms,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale="Blues",
                showscale=False
            ),
            hovertemplate="<b>%{y}</b><br>TF-IDF Weight: %{x:.4f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"Distinctive TF-IDF Terms: {doc_title[:50]}",
            xaxis_title="TF-IDF Weight",
            yaxis_title="Distinctive Term",
            template="plotly_white",
            height=max(350, len(terms) * 25),
            margin=dict(l=150, r=30, t=50, b=50)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(terms, scores, color="#3F88C5")
        ax.set_title(f"TF-IDF Terms: {doc_title[:40]}")
        ax.set_xlabel("TF-IDF Score")
        plt.tight_layout()
        return fig


def plot_similarity_heatmap(
    sim_matrix: List[List[float]],
    titles: List[str]
):
    """Generates annotated pairwise cosine similarity heatmap."""
    if not sim_matrix or len(sim_matrix) < 2:
        return None

    # Truncate titles for readable heatmap labels
    short_titles = [f"{t[:25]}..." if len(t) > 25 else t for t in titles]

    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=short_titles,
            y=short_titles,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            text=[[f"{val:.2f}" for val in row] for row in sim_matrix],
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar=dict(title="Cosine Sim")
        ))
        fig.update_layout(
            title="Pairwise Document Similarity Matrix (TF-IDF Cosine Similarity)",
            template="plotly_white",
            height=max(450, len(titles) * 50),
            xaxis=dict(tickangle=-30),
            margin=dict(l=120, r=50, t=60, b=120)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(short_titles)))
        ax.set_yticks(range(len(short_titles)))
        ax.set_xticklabels(short_titles, rotation=45, ha="left")
        ax.set_yticklabels(short_titles)
        plt.tight_layout()
        return fig


def plot_model_comparison(
    summary_data: List[Dict[str, Any]]
):
    """Generates grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    if not summary_data:
        return None

    df = pd.DataFrame(summary_data)
    metrics_to_plot = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        for idx, metric in enumerate(metrics_to_plot):
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df["Model"],
                    y=df[metric],
                    marker_color=colors[idx % len(colors)],
                    text=df[metric].apply(lambda v: f"{v:.3f}"),
                    textposition="auto"
                ))

        fig.update_layout(
            barmode="group",
            title="Comparative Performance: Baseline vs BiLSTM vs SciBERT",
            yaxis=dict(title="Score", range=[0.0, 1.05]),
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        df.plot(x="Model", y=metrics_to_plot, kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Model Comparison")
        plt.tight_layout()
        return fig


def plot_confusion_matrix(
    cm: List[List[int]],
    classes: List[str],
    model_name: str = "Model"
):
    """Generates annotated confusion matrix heatmap."""
    if not cm:
        return None

    cm_array = np.array(cm)
    short_classes = [c[:18] for c in classes]

    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=short_classes,
            y=short_classes,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
            colorbar=dict(title="Paper Count")
        ))
        fig.update_layout(
            title=f"Confusion Matrix: {model_name}",
            xaxis_title="Predicted Domain",
            yaxis_title="True Domain",
            template="plotly_white",
            height=450,
            xaxis=dict(tickangle=-25),
            margin=dict(l=100, r=40, t=60, b=90)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm_array, cmap="Blues")
        fig.colorbar(cax)
        ax.set_xticks(range(len(short_classes)))
        ax.set_yticks(range(len(short_classes)))
        ax.set_xticklabels(short_classes, rotation=45, ha="left")
        ax.set_yticklabels(short_classes)
        plt.tight_layout()
        return fig


def plot_batch_domain_distribution(
    domain_counts: Dict[str, int]
):
    """Generates donut/pie chart of predicted domains across uploaded papers."""
    if not domain_counts:
        return None

    labels = list(domain_counts.keys())
    values = list(domain_counts.values())

    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            hoverinfo="label+percent+value",
            textinfo="label+value",
            marker=dict(colors=px.colors.qualitative.Plotly)
        )])
        fig.update_layout(
            title="Batch Domain Distribution",
            template="plotly_white",
            height=400,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Batch Domain Distribution")
        plt.tight_layout()
        return fig
