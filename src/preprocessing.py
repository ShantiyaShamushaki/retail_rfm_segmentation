"""
Module: preprocessing.py
Version: 0.1
Author: Shantiya Shamushaki
Description:
    Data loading and cleaning module for Online Retail II dataset.
    Handles duplicate removal, missing value imputation, quantity validation,
    and computation of total purchase amount per transaction.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load transactional dataset from given path.
    
    Parameters
    ----------
    path : str
        File path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Raw imported dataset.
    """
    df = pd.read_csv(path, encoding='utf-8')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing operations:
    - Remove duplicate rows
    - Drop missing 'CustomerID' or 'InvoiceDate'
    - Filter out negative quantities
    - Convert 'InvoiceDate' column to datetime
    - Create consolidated 'TotalAmount' column

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactional dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for RFM preparation.
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows with missing mandatory fields
    df = df.dropna(subset=['Customer ID', 'InvoiceDate'])

    # Convert date string to datetime type
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # Filter invalid quantities (negative or zero)
    df = df[df['Quantity'] > 0]

    # Create total purchase value
    df['TotalAmount'] = df['Quantity'] * df['Price']

    return df


def summarize_sales(df: pd.DataFrame) -> dict:
    """
    Generate key descriptive statistics for cleaned dataset.
    Calculates KPIs required for initial EDA section of setup.ipynb.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactional dataset.

    Returns
    -------
    dict
        Summary statistics including avg sale, yearly growth, transactions count.
    """
    summary = {
        'unique_customers': df['Customer ID'].nunique(),
        'avg_sales': round(df['TotalAmount'].mean(), 2),
        'total_transactions': len(df),
        'yearly_growth': df.groupby(df['InvoiceDate'].dt.year)['TotalAmount']
                              .sum()
                              .pct_change()
                              .dropna()
                              .to_dict(),
        'best_day': df.groupby(df['InvoiceDate'].dt.date)['TotalAmount']
                      .sum()
                      .idxmax(),
        'worst_day': df.groupby(df['InvoiceDate'].dt.date)['TotalAmount']
                       .sum()
                       .idxmin()
    }
    return summary
