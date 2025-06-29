import pandas as pd

def extract_subcat_tools(df: pd.DataFrame) -> pd.DataFrame:
    suffixes = ['_HH_T', '_Corp_T', '_Gen_T', '_FX_T', '_FCD_T']
    subcat_cols = [col for col in df.columns if any(suffix in col for suffix in suffixes)]

    df_long = df.melt(
        id_vars=['iso2', 'Year', 'Month'],
        value_vars=subcat_cols,
        var_name='MaPP_Tool_Full',
        value_name='OBS_Tightening'
    )
    df_long[['MaPP_Tool', 'Subcategory']] = df_long['MaPP_Tool_Full'].str.extract(r'(\w+?)_((?:HH|Corp|Gen|FX|FCD))_T')

    return df_long[['Country_Code', 'Year', 'Month', 'MaPP_Tool', 'Subcategory', 'OBS_Tightening']]
