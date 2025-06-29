import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


# Define modified version as regular function without ace_tools to avoid errors

def assess_data_quality(df: pd.DataFrame, time_col: str = 'year_quarter', country_col: str = 'country') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform initial assessment to evaluate data missing situation, time series-wise, and country-wise coverage, with visualization.

    Parameters:
        df (pd.DataFrame): DataFrame to evaluate.
        time_col (str): Column name indicating quarter or year-month.
        country_col (str): Column name indicating country code.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Missing summary, quarterly coverage rate, country coverage rate.
    """

    # ① Missing value summary
    missing_summary = df.isnull().sum().to_frame('Missing Count')
    missing_summary['Total Rows'] = len(df)
    missing_summary['Missing Ratio (%)'] = 100 * missing_summary['Missing Count'] / missing_summary['Total Rows']
    missing_summary = missing_summary.sort_values(by='Missing Ratio (%)', ascending=False)

    # ② Quarterly coverage rate
    quarter_summary = df.groupby(time_col).apply(lambda x: x.notnull().mean()).T

    # ③ Country-wise coverage rate
    country_summary = df.groupby(country_col).apply(lambda x: x.notnull().mean()).T

    # ④ Visualization
    heatmap_path = "../results/missing_heatmap.png"
    barplot_path = "../results/missing_ratio_barplot.png"

    plt.figure(figsize=(16, 8))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    missing_summary['Missing Ratio (%)'].plot(kind='bar')
    plt.title('Missing Data Ratio per Column')
    plt.ylabel('Missing Ratio (%)')
    plt.tight_layout()
    plt.savefig(barplot_path)
    plt.close()

    return missing_summary, quarter_summary, country_summary, heatmap_path, barplot_path

