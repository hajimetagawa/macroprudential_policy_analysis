import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


# エラー回避のため、ace_toolsを使用せず通常の関数として修正したバージョンを定義

def assess_data_quality(df: pd.DataFrame, time_col: str = 'year_quarter', country_col: str = 'country') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    初期アセスメントを行い、データ欠損状況、時系列別、国別の充足状況を評価し、可視化する。

    Parameters:
        df (pd.DataFrame): 評価対象のデータフレーム。
        time_col (str): 四半期または年月を示す列名。
        country_col (str): 国コードを示す列名。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 欠損要約、四半期別充足率、国別充足率。
    """

    # ① 欠損値サマリー
    missing_summary = df.isnull().sum().to_frame('Missing Count')
    missing_summary['Total Rows'] = len(df)
    missing_summary['Missing Ratio (%)'] = 100 * missing_summary['Missing Count'] / missing_summary['Total Rows']
    missing_summary = missing_summary.sort_values(by='Missing Ratio (%)', ascending=False)

    # ② 四半期別の充足率
    quarter_summary = df.groupby(time_col).apply(lambda x: x.notnull().mean()).T

    # ③ 国別の充足率
    country_summary = df.groupby(country_col).apply(lambda x: x.notnull().mean()).T

    # ④ 可視化
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

