"""
Module: rfm_segmentation.py
Version: 0.1
Author: Shantiya Shamushaki
Description:
    Core RFM segmentation module for Online Retail II project.
    Includes functions for RFM extraction, normalization, optimal K determination,
    and customer clustering via K-Means.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from report_result import plot_k_evaluation

def compute_rfm(df: pd.DataFrame, ref_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary values per customer.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactional dataset containing columns:
        'Customer ID', 'InvoiceDate', 'Invoice', 'TotalAmount'.
    ref_date : pd.Timestamp, optional
        Reference date for recency calculation. If None, uses max(InvoiceDate) + 1 day.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Customer ID with columns ['Recency', 'Frequency', 'Monetary'].
    """
    if ref_date is None:
        ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'Invoice': 'count',
        'TotalAmount': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalAmount': 'Monetary'
    })
    
    return rfm


def normalize_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation and MinMax scaling to RFM features.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        Raw RFM metrics per customer.

    Returns
    -------
    pd.DataFrame
        Log-transformed and normalized RFM metrics ready for clustering.
    """
    # Logarithmic transformation to reduce skewness
    rfm_log = np.log1p(rfm)

    # MinMax scaling to [0, 1] range
    scaled = MinMaxScaler().fit_transform(rfm_log)
    rfm_scaled = pd.DataFrame(scaled, columns=rfm.columns, index=rfm.index)

    return rfm_scaled


def find_best_k(data: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> int:
    """
    Determine optimal number of clusters based on Silhouette Score.
    
    Parameters
    ----------
    data : pd.DataFrame
        Normalized RFM data.
    k_min : int
        Minimum number of clusters to evaluate.
    k_max : int
        Maximum number of clusters to evaluate.

    Returns
    -------
    int
        Optimal number of clusters with maximum Silhouette Score.
    """
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    return best_k, scores


def segment_customers(rfm_scaled: pd.DataFrame, best_k: int) -> pd.DataFrame:
    """
    Perform K-Means clustering and label customers by cluster id.
    
    Parameters
    ----------
    rfm_scaled : pd.DataFrame
        Scaled RFM metrics.
    best_k : int
        Number of clusters determined from find_best_k().

    Returns
    -------
    pd.DataFrame
        RFM scaled data with an additional 'Cluster' label column.
    """
    km = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    labels = km.fit_predict(rfm_scaled)

    rfm_labeled = rfm_scaled.copy()
    rfm_labeled['Cluster'] = labels

    return rfm_labeled


def classify_active_inactive(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Classify customers into 'Active' and 'Inactive' groups based on Cluster.

    Parameters
    ----------
    rfm : pd.DataFrame
        Raw RFM table containing columns ['Recency', 'Frequency', 'Monetary'].

    Returns
    -------
    pd.DataFrame
        RFM table with an additional 'ActivityStatus' column.
    """
    rfm_classified = rfm.copy()
    rfm_classified['ActivityStatus'] = [True if cluster == 1 else False for cluster in rfm_classified['Cluster']]
    rfm_classified.drop("Cluster", axis=1)
    
    return rfm_classified

