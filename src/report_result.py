"""
Module: report_result.py
Version: 1
Author: Shantiya Shamushaki
Description:
    Reporting and visualization module for Retail RFM Segmentation project.
    Includes cluster visualization, summary export, and daily sales analytics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D plotting
import os
import numpy as np


def visualize_clusters(rfm_labeled: pd.DataFrame, save_path: str = "../result/cluster_map.png") -> None:
    """
    Visualize customer clusters in 3D space (Recency, Frequency, Monetary).

    Parameters
    ----------
    rfm_labeled : pd.DataFrame
        RFM-scaled data with a 'Cluster' label column.
    save_path : str
        Path to save the generated 3D visualization image.

    Returns
    -------
    None
    """

    # Prepare figure
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with clusters
    scatter = ax.scatter(
        rfm_labeled['Recency'],
        rfm_labeled['Frequency'],
        rfm_labeled['Monetary'],
        c=rfm_labeled['Cluster'],
        cmap='tab10',
        s=50,
        alpha=0.8
    )

    # Set labels
    ax.set_xlabel("Recency (scaled)")
    ax.set_ylabel("Frequency (scaled)")
    ax.set_zlabel("Monetary (scaled)")
    ax.set_title("Customer Segmentation (3D RFM Space)", fontsize=13)

    # Legend mapping
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f"ActivityStatus {c}" for c in rfm_labeled['ActivityStatus']], title="Clusters", loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def export_summary(summary_df: pd.DataFrame, filename: str = "cluster_summary.csv", save_dir: str = "../result/") -> None:
    """
    Export cluster summary statistics to CSV.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame containing cluster means/aggregations.
    filename : str
        Output CSV file name.
    save_dir : str
        Directory where the file will be saved.

    Returns
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure folder exists
    full_path = os.path.join(save_dir, filename)
    summary_df.to_csv(full_path, index=False)
    print(f"Cluster summary exported to: {full_path}")


def plot_sales_distribution(df: pd.DataFrame, save_path: str = "../result/sales_distribution.png") -> None:
    """
    Visualize the log-transformed distribution of total sales amounts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales transactions containing 'TotalAmount'.
    save_path : str
        Path to save the histogram of sales distribution.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(df['TotalAmount']), bins=60, color='#4C72B0', edgecolor='black')
    plt.title("Distribution of Sales (log-transformed)", fontsize=12)
    plt.xlabel("Log(1 + TotalAmount)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_best_worst_days(df: pd.DataFrame, save_path: str = "../result/best_worst_days.png") -> None:
    """
    Plot best and worst performing days based on total daily sales.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions containing 'InvoiceDate' and 'TotalAmount'.
    save_path : str
        Path to save daily performance figure.
    """
    # Aggregate sales by day
    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
    daily_sales.rename(columns={'InvoiceDate': 'Date', 'TotalAmount': 'DailySales'}, inplace=True)

    # Identify best and worst days
    best_day = daily_sales.loc[daily_sales['DailySales'].idxmax()]
    worst_day = daily_sales.loc[daily_sales['DailySales'].idxmin()]

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=daily_sales, x='Date', y='DailySales', color='#2C7FB8', linewidth=1.5)
    plt.scatter(best_day['Date'], best_day['DailySales'], color='green', s=70, label='Best Day')
    plt.scatter(worst_day['Date'], worst_day['DailySales'], color='red', s=70, label='Worst Day')

    plt.title("Best and Worst Revenue Days", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Total Daily Sales (£)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_k_evaluation(scores: dict, save_path: str = "../result/k_evaluation.png") -> None:
    """
    Plot K range vs Silhouette Score to visualize optimal cluster choice.
    
    Parameters
    ----------
    scores : dict
        Keys: k values (int)
        Values: silhouette score (float) from find_best_k loop.
    save_path : str
        Path to save PNG plot.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o', color='#4C72B0')
    plt.title("K-Means Evaluation (Silhouette Score)", fontsize=12)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Annotate best k
    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    plt.scatter(best_k, best_score, color='red', s=70, label=f"Best k = {best_k}")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
