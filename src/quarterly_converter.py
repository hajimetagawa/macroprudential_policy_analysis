import pandas as pd

def convert_to_quarterly(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['Quarter'] = df['Month'].apply(lambda m: f"Q{((int(m)-1)//3)+1}")
    df['Year_Quarter'] = df['Year'].astype(str) + df['Quarter']

    df_agg = (
        df.groupby(['iso2', 'MaPP_Tool', 'Year_Quarter'], as_index=False)
        .agg({'OBS_Tightening': 'sum'})
        .rename(columns={'OBS_Tightening': 'OBS_Tightening_Aggregated'})
    )

    df_bin = (
        df.groupby(['iso2', 'MaPP_Tool', 'Year_Quarter'], as_index=False)
        .agg({'OBS_Tightening': 'max'})
        .rename(columns={'OBS_Tightening': 'OBS_Tightening_Binary'})
    )

    return df_agg, df_bin
