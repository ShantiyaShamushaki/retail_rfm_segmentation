### **Repository Purpose**
Data‑driven segmentation of customers based on RFM (Recency, Frequency, Monetary) metrics extracted from the `Online Retail II` dataset.  
Goal: produce meaningful behavioral clusters using K‑Means and interpret patterns visually.

---

### **1. Directory Structure**

```
retail_rfm_segmentation/
│
├── data/
│   └── online_retail_II.csv
│
├── notebooks/
│   └── setup.ipynb                    # Core project runbook (EDA + RFM Modeling)
│
├── src/
│   ├── preprocessing.py               # Data loading / cleaning / aggregation
│   ├── rfm_segmentation.py            # RFM computation, scaling, clustering
│   └── report_result.py               # Export plots and CSV summaries
│
├── result/
│   ├── plots/                         # Visual results (PNG)
│   ├── tables/                        # Segmentation CSV summaries
│   └── analytics/                     # Optional aggregate charts
│
└── README.md
```

---

### **2. Workflow Structure (Execution Path)**

| Stage | File / Notebook | Description | Output |
|-------|-----------------|-------------|---------|
| **1. Data Preparation & Exploration** | `src/preprocessing.py` & `notebooks/setup.ipynb` | Load, clean, and format transactional data. Display customer count, sales distribution, yearly growth, most/least active days. | Cleaned DataFrame + Summary plots |
| **2. RFM Computation & Segmentation** | `src/rfm_segmentation.py` | Compute RFM columns, apply log scaling and MinMax normalization. Determine optimal *k* using Silhouette Score, perform K‑Means clustering. | Cluster labels + CSV summary |
| **3. Visualization & Reporting** | `src/report_result.py` | Visualize cluster patterns: spending vs frequency, recency distribution per segment, cluster centroid plot. Export report outputs (CSV, PNG). | Result tables and plots in `/result/` |

---

### **3. Technical Module Design**

**`src/preprocessing.py`**
```python
# Version 0.1 - Data loading and cleaning module
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load transactional dataset."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls, duplicates, handle date formatting, and filter out negative quantities."""
    df = df.drop_duplicates()
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Quantity'] > 0]
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df

def summarize_sales(df: pd.DataFrame) -> dict:
    """Compute basic KPIs for sales and customer stats."""
    summary = {
        'unique_customers': df['CustomerID'].nunique(),
        'avg_sales': df['TotalAmount'].mean(),
        'total_transactions': len(df),
        'yearly_growth': df.groupby(df['InvoiceDate'].dt.year)['TotalAmount'].sum().pct_change().dropna()
    }
    return summary
```

---

**`src/rfm_segmentation.py`**
```python
# Version 0.2 - RFM computation and K-Means segmentation
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_rfm(df: pd.DataFrame, ref_date=None) -> pd.DataFrame:
    """Derive RFM values per customer."""
    if ref_date is None:
        ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalAmount': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'})
    return rfm

def normalize_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Log transform and scale RFM features."""
    rfm_log = np.log1p(rfm)
    scaled = MinMaxScaler().fit_transform(rfm_log)
    return pd.DataFrame(scaled, columns=rfm.columns, index=rfm.index)

def find_best_k(data) -> int:
    """Find optimal K using Silhouette score."""
    scores = {}
    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=42).fit(data)
        scores[k] = silhouette_score(data, model.labels_)
    return max(scores, key=scores.get)

def segment_customers(rfm_scaled: pd.DataFrame, best_k: int) -> pd.DataFrame:
    """Cluster customers and append labels."""
    km = KMeans(n_clusters=best_k, random_state=42).fit(rfm_scaled)
    labels = km.labels_
    result = rfm_scaled.copy()
    result['Cluster'] = labels
    return result
```

---

**`src/report_result.py`**
```python
# Version 0.3 - Visualization and report generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, os

def visualize_clusters(rfm_labeled: pd.DataFrame, save_path: str):
    """Create scatter plots and distribution charts for clusters."""
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=rfm_labeled, x='Recency', y='Monetary', hue='Cluster', palette='viridis')
    plt.title('Customer Segmentation by RFM')
    plt.savefig(os.path.join(save_path, 'cluster_scatter.png'), dpi=200)
    plt.close()

def export_summary(rfm_labeled: pd.DataFrame, save_path: str):
    """Export aggregated cluster summary to CSV."""
    summary = rfm_labeled.groupby('Cluster').mean().round(2)
    summary.to_csv(os.path.join(save_path, 'cluster_summary.csv'))
    return summary
```

---

### **4. Output Expectations**
Each notebook execution will produce:  
- **CSV:** `result/tables/cluster_summary.csv`  
- **Plots:** `result/plots/*.png`  
- **Metrics:** Printed KPIs (unique customers, avg sales, yearly growth)  

---

### **5. Notebook `setup.ipynb` Flow**
1. Import packages and local modules (`sys.path.append('../src')`).
2. Load raw data → Clean → Summarize (`preprocessing.py`).
3. Visualize Sales Distribution & KPIs.
4. Compute RFM → Normalize → Determine Best K → Cluster → Visualize.
5. Save CSV and plots to `/result/`.

---

### **6. Project Output Summary**
| Output Type | Example File | Description |
|--------------|---------------|--------------|
| Summary CSV | `result/tables/cluster_summary.csv` | Mean RFM per cluster |
| Cluster Plot | `result/plots/cluster_scatter.png` | Segmentation visualization |
| Stats Plot | `result/plots/sales_distribution.png` | Overall sale frequency and distribution |

