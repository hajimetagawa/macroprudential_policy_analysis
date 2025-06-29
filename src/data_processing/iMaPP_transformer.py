import pandas as pd
import os
import yaml
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def month_to_quarter(month: int) -> str:
    """
    Convert month to quarter (Q1~Q4)
    
    Parameters:
    - month: Month (1-12)
    
    Returns:
    - str: Quarter string (Q1, Q2, Q3, Q4, Q?)
    """
    if not isinstance(month, (int, float)):
        logger.warning(f"Invalid month data: {month} (type: {type(month)})")
        return "Q?"
        
    month = int(month)
    
    if month in [1, 2, 3]:
        return "Q1"
    elif month in [4, 5, 6]:
        return "Q2"
    elif month in [7, 8, 9]:
        return "Q3"
    elif month in [10, 11, 12]:
        return "Q4"
    else:
        logger.warning(f"Invalid month value: {month}")
        return "Q?"


def transform_imapp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform iMaPP data (Wide→Long, outlier processing, column renaming, quarter addition)

    Parameters:
    - df: Wide format input DataFrame

    Returns:
    - DataFrame formatted to Long format
    """

    # ===== Configuration definition =====
    rename_map = {
        "Country": "country",
        "iso2": "country_code",
        "iso3": "country_code_iso3",
        "ifscode": "ifs_code",
        "AE": "advanced",
        "EMDE": "emerging",
        "Year": "year",
        "Month": "month",
        "mapp_tool": "mapp_tool",
        "obs": "obs",
        "Quarter": "year_quarter"
    }

    output_columns = [
        "country", "country_code", "country_code_iso3", "ifs_code",
        "advanced", "emerging", "year", "month", "year_quarter",
        "mapp_tool", "obs"
    ]

    reserved_cols = list(rename_map.keys())
    reserved_cols_base = [col for col in reserved_cols if col not in ["mapp_tool", "obs", "Quarter"]]

    # ===== Column name cleansing (remove _T, _L) =====
    df.columns = [col.replace("_T", "").replace("_L", "") for col in df.columns]

    # ===== Extract required columns (+ indicator columns) =====
    df = df[[col for col in reserved_cols_base if col in df.columns] + 
            [col for col in df.columns if col not in reserved_cols_base]]

    # ===== Outlier cleaning (Wide format) =====
    for col in df.columns:
        if col not in reserved_cols_base:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ===== Wide → Long conversion (add mapp_tool column) =====
    df_long = df.melt(
        id_vars=reserved_cols_base,
        var_name="mapp_tool",
        value_name="obs"
    )

    # ===== Add quarter column =====
    df_long["Quarter"] = df_long["Year"].astype(str) + "-" + df_long["Month"].apply(month_to_quarter)

    # ===== Column name conversion =====
    df_long = df_long.rename(columns=rename_map)

    # ===== Column order formatting (existing columns only) =====
    df_long = df_long[[col for col in output_columns if col in df_long.columns]]

    return df_long


def validate_imapp_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate basic validity of iMaPP DataFrame
    
    Parameters:
    - df: DataFrame to validate
    
    Returns:
    - bool: Validity check result
    """
    try:
        if df.empty:
            logger.error("iMaPP DataFrame is empty")
            return False
            
        required_columns = ["country", "year", "month", "mapp_tool", "obs"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"iMaPP missing required columns: {missing_columns}")
            return False
            
        # Data type check
        if not pd.api.types.is_numeric_dtype(df["year"]):
            logger.warning("Year data is not numeric type")
            
        if not pd.api.types.is_numeric_dtype(df["month"]):
            logger.warning("Month data is not numeric type")
            
        if not pd.api.types.is_numeric_dtype(df["obs"]):
            logger.warning("Observation value data is not numeric type")
            
        logger.info(f"iMaPP data validity check passed ({len(df)} rows, {len(df.columns)} columns)")
        return True
        
    except Exception as e:
        logger.error(f"iMaPP data validity check error: {e}")
        return False

def prepare_imapp_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format iMaPP data for analysis
    
    Parameters:
    - df: Long format iMaPP DataFrame
    
    Returns:
    - pd.DataFrame: DataFrame aggregated for analysis
    
    Raises:
    - ValueError: Data validity error
    """
    try:
        # Data validity check
        if not validate_imapp_dataframe(df):
            raise ValueError("iMaPP data validity check failed")
            
        df = df.copy()

        # ===== 1. Exclude "SUM_17" records =====
        initial_count = len(df)
        df = df[df["mapp_tool"] != "SUM_17"]
        filtered_count = len(df)
        logger.info(f"Excluded SUM_17 records: {initial_count} → {filtered_count} rows")

        # ===== 2. Convert monthly to quarter end dates (add quarter_end column) =====
        try:
            df["quarter_end"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
            df["quarter_end"] = df["quarter_end"] + pd.offsets.QuarterEnd(0)
        except Exception as e:
            logger.error(f"Quarter date conversion error: {e}")
            raise

        # ===== 3. Quarter aggregation (Aggregate + Binary) =====
        try:
            grouped = df.groupby([
                "country", "country_code", "mapp_tool", 
                "advanced", "emerging", "year", "year_quarter"
            ]).agg(
                obs_agg=("obs", "sum"),    # Intensity (implementation count)
                obs_bin=("obs", "max")     # Whether implemented (1 if any implementation)
            ).reset_index()
            
            logger.info(f"Quarter aggregation complete: {len(grouped)} rows")
            return grouped
            
        except Exception as e:
            logger.error(f"Quarter aggregation error: {e}")
            raise
            
    except Exception as e:
        logger.error(f"iMaPP analysis data preparation error: {e}")
        raise
