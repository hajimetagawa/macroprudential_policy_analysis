import pandas as pd
import os
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def dataset_processor(df_dict: Dict[str, pd.DataFrame], output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Process multiple DataFrames by dataset name and save to processed folder.

    Parameters:
    - df_dict: dict[str, pd.DataFrame] → Raw DataFrame collection with names as keys
    - output_dir: Directory path to save processed CSV files

    Returns:
    - dict[str, pd.DataFrame]: Processed DataFrame collection for analysis
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== Dispatch processing for each dataset =====
    processed_dict = {}
    for name, df in df_dict.items():
        print(f"🔧 Processing {name}...")

        if name == "credit_gap":
            df_proc = process_credit_gap(df)

        elif name == "total_credit":
            df_proc = process_total_credit(df)

        elif name == "debt_service_ratio":
            df_proc = process_debt_service_ratio(df)

        elif name == "residential_property_price":
            df_proc = process_residential_property_price(df)

        elif name == "commercial_property_price":
            df_proc = process_commercial_property_price(df)

        elif name == "effective_exchange_rate":
            df_proc = process_effective_exchange_rate(df)

        elif name == "central_bank_policy_rate":
            df_proc = process_central_bank_policy_rate(df)
        

        else:
            print(f"⚠️ Unsupported dataset: {name} → Skipping")
            continue

        # Save processed data
        save_path = os.path.join(output_dir, f"{name}.csv")
        df_proc.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ Save complete: {save_path}")

        # Store results
        processed_dict[name] = df_proc

    return processed_dict



def validate_required_columns(df: pd.DataFrame, required_columns: list, dataset_name: str) -> bool:
    """Check for required column existence"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"{dataset_name}: Missing required columns - {missing_columns}")
        return False
    return True

def safe_numeric_conversion(series: pd.Series, column_name: str) -> pd.Series:
    """Safe numeric conversion"""
    try:
        converted = pd.to_numeric(series, errors="coerce")
        null_count = converted.isnull().sum()
        if null_count > 0:
            logger.warning(f"{column_name}: {null_count} values could not be converted to numeric")
        return converted.fillna(0)
    except Exception as e:
        logger.error(f"{column_name}: Numeric conversion error - {e}")
        return series.fillna(0)

def process_credit_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Process credit gap data"""
    try:
        # ===== Extract only necessary columns =====
        required_columns = ["BORROWERS_CTY", "CG_DTYPE", "TIME_PERIOD", "OBS_VALUE"]
        if not validate_required_columns(df, required_columns, "credit_gap"):
            raise ValueError("Credit gap: Missing required columns")
            
        df = df[required_columns].copy()

        # ===== Column name conversion =====
        df = df.rename(columns={
            "BORROWERS_CTY": "country",
            "CG_DTYPE": "data_type",
            "TIME_PERIOD": "year_quarter",
            "OBS_VALUE": "obs_value"
        })

        # ===== Value conversion (numeric) =====
        df["obs_value"] = safe_numeric_conversion(df["obs_value"], "credit_gap_obs_value")

        # ===== Data type label conversion =====
        dtype_map = {
            "A": "actual",
            "B": "trend_hp_filter",
            "C": "trend_actual"
        }
        df["data_type"] = df["data_type"].replace(dtype_map)

        # ===== Convert to wide format =====
        df = df.pivot(index=["country", "year_quarter"], columns="data_type", values="obs_value").reset_index()
        logger.info(f"Credit gap processing complete: {len(df)} rows")
        return df
            
    except Exception as e:
        logger.error(f"Credit gap processing error: {e}")
        raise

def process_total_credit(df: pd.DataFrame) -> pd.DataFrame:
    # ===== Extract only necessary columns =====
    keep_columns = ["BORROWERS_CTY", "TC_BORROWERS", "TC_LENDERS", "VALUATION", "UNIT_TYPE",
                     "TC_ADJUST", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== Column name conversion =====
    df = df.rename(columns={
        "BORROWERS_CTY": "country",
        "TC_BORROWERS": "borrowing_sector",
        "TC_LENDERS": "lending_sector",
        "VALUATION": "valuation_method",
        "UNIT_TYPE": "unit_type",
        "TC_ADJUST": "adjustment",
        "TIME_PERIOD": "year_quarter",
        "OBS_VALUE": "obs_value"
    })

    # ===== Value conversion (numeric) =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== Label mapping =====
    borrowing_sector_map = {
        "G": "general_government",
        "H": "households_and_npishs",
        "C": "non_financial_sector",
        "N": "non_financial_corporations",
        "P": "private_non_financial_sector"
    }
    df["borrowing_sector"] = df["borrowing_sector"].replace(borrowing_sector_map)

    lending_sector_map = {
        "A": "all_sectors",
        "B": "domestic_banks"
    }
    df["lending_sector"] = df["lending_sector"].replace(lending_sector_map)

    valuation_method_map = {
        "M": "market_value",
        "N": "nominal_value"
    }
    df["valuation_method"] = df["valuation_method"].replace(valuation_method_map)

    unit_type_map = {
        "770": "pct_of_gdp",
        "799": "pct_of_gdp_ppp",
        "USD": "usd",
        "XDC": "domestic_currency"
    }
    df["unit_type"] = df["unit_type"].replace(unit_type_map)

    adjustment_map = {
        "A": "adjusted_for_break",
        "U": "unadjusted"
    }
    df["adjustment"] = df["adjustment"].replace(adjustment_map)

    # ===== Conditional filtering =====
    df = df[
        (df["unit_type"] == "pct_of_gdp") &
        (df["valuation_method"] == "market_value") &
        (df["adjustment"] == "adjusted_for_break")
    ]

    df_all_lender = df[df["lending_sector"] == "all_sectors"]
    df_domestic_banks_lender = df[df["lending_sector"] == "domestic_banks"]

    # ===== Convert to wide format =====
    df_all_lender = df_all_lender.pivot(
        index=["country", "year_quarter"], 
        columns="borrowing_sector", 
        values="obs_value"
        ).reset_index()
    
    df_all_lender.columns = [
        col if col in ["country", "year_quarter"] else f"{col}_all_sector"
        for col in df_all_lender.columns
    ]

    df_domestic_banks_lender = df_domestic_banks_lender.pivot(
        index=["country", "year_quarter"], 
        columns="borrowing_sector", 
        values="obs_value"
        ).reset_index()

    df_domestic_banks_lender.columns = [
        col if col in ["country", "year_quarter"] else f"{col}_domestic_banks"
        for col in df_domestic_banks_lender.columns
    ]

    # ===== Horizontal merge (add columns) =====
    df_merged = pd.merge(
        df_all_lender,
        df_domestic_banks_lender,
        on=["country", "year_quarter"],
        how="outer"
    )

    return df_merged

def process_debt_service_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # ===== Extract only necessary columns =====
    keep_columns = ["BORROWERS_CTY", "DSR_BORROWERS", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== Column name conversion =====
    df = df.rename(columns={
        "BORROWERS_CTY": "country",
        "DSR_BORROWERS": "dsr_borrowers",
        "TIME_PERIOD": "year_quarter",
        "OBS_VALUE": "obs_value"
    })

    # ===== Value conversion (numeric) =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== Data type label conversion =====
    dsr_borrowers_map = {
        "H": "households_and_npishs",
        "N": "non_financial_corporations",
        "P": "private_non_financial_sector"
    }
    df["dsr_borrowers"] = df["dsr_borrowers"].replace(dsr_borrowers_map)

    # ===== Convert to wide format =====
    df = df.pivot(
        index=["country", "year_quarter"], 
        columns="dsr_borrowers", 
        values="obs_value"
        ).reset_index()

    return df

def process_residential_property_price(df: pd.DataFrame) -> pd.DataFrame:
    # ===== Extract only necessary columns =====
    keep_columns = ["REF_AREA", "VALUE", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    df["UNIT_MEASURE"] = df["UNIT_MEASURE"].astype("str")   

    # ===== Column name conversion =====
    df = df.rename(columns={
        "REF_AREA": "country",
        "VALUE": "value",
        "UNIT_MEASURE": "unit_of_measure",
        "TIME_PERIOD": "year_quarter",
        "OBS_VALUE": "obs_value"
    })

    # ===== Value conversion (numeric) =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== Data type label conversion =====
    value_map = {
        "N": "nominal",
        "R": "real",
    }
    df["value"] = df["value"].replace(value_map)

    unit_of_measure_map = {
        "628": "index_vs_2010",
        "771": "yoy_changes_pct",
    }
    df["unit_of_measure"] = df["unit_of_measure"].replace(unit_of_measure_map)

    # ===== Conditional filtering =====
    df = df[df["value"] == "real"]

    # ===== Convert to wide format =====
    df = df.pivot(
        index=["country", "year_quarter"], 
        columns="unit_of_measure", 
        values="obs_value"
        ).reset_index()

    return df

def process_commercial_property_price(df: pd.DataFrame) -> pd.DataFrame:
    # ===== Extract only necessary columns =====
    keep_columns = ["FREQ", "REF_AREA", "COVERED_AREA", "RE_TYPE", "COMPILING_ORG", "PRICED_UNIT", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== Column name conversion =====
    df = df.rename(columns={
        "FREQ": "frequency",
        "REF_AREA": "country",
        "COVERED_AREA": "covered_area",
        "RE_TYPE": "real_estate_type",
        "COMPILING_ORG": "compiling_agency",
        "PRICED_UNIT": "priced_unit",
        "UNIT_MEASURE": "unit_of_measure",
        "TIME_PERIOD": "year_quarter",
        "OBS_VALUE": "obs_value"
    })

    # ===== Value conversion (numeric) =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== Data type label conversion =====
    covered_area_map = {
        "0": "whole_country",
        "2": "capital",
        "3": "capital+suburbs",
        "4": "big_cities",
        "5": "a_big_city",
        "9": "urban_areas",
    }
    df["covered_area"] = df["covered_area"].replace(covered_area_map)

    real_estate_type_map = {
        "A": "commercial_property",
        "B": "commercial_property_office_premises",
        "C": "commercial_property_retail_premises",
        "D": "office_and_retail",
        "G": "industrial_properties",
        "I": "agricultural_properties",
        "M": "land_for_commercial",
        "O": "rented_dwellings"
    }
    df["real_estate_type"] = df["real_estate_type"].replace(real_estate_type_map)

    # ===== Conditional filtering =====
    df = df[
        (df["frequency"] == "Q") &
        (df["unit_of_measure"] != "USD") &
        (df["unit_of_measure"] != "AED") &
        (df["unit_of_measure"] != "PHP")
    ]

    # ===== Calculate average values by quarter and country =====
    df_grouped = df.groupby(["country", "year_quarter"], as_index=False)["obs_value"].mean()

    return df_grouped


def process_effective_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    # ===== Extract only necessary columns =====
    keep_columns = ["FREQ", "EER_TYPE", "EER_BASKET", "REF_AREA", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== Column name conversion =====
    df = df.rename(columns={
        "FREQ": "frequency",
        "EER_TYPE": "eer_type",
        "EER_BASKET": "eer_basket",
        "RE_TYPE": "real_estate_type",
        "REF_AREA": "country",
        "TIME_PERIOD": "year_month",
        "OBS_VALUE": "obs_value"
    })

    # ===== Value conversion (numeric) =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== Data type label conversion =====
    eer_type_map = {
        "N": "nominal",
        "R": "real"
    }
    df["eer_type"] = df["eer_type"].replace(eer_type_map)

    eer_basket_map = {
        "B": "broad",
        "N": "narrow"
    }
    df["eer_basket"] = df["eer_basket"].replace(eer_basket_map)

    # ===== Conditional filtering =====
    df = df[
        (df["frequency"] == "M") &
        (df["eer_basket"] == "broad")
    ]

    # ===== Convert to wide format =====
    df = df.pivot(
        index=["country", "year_month"], 
        columns="eer_type", 
        values="obs_value"
        ).reset_index()

    # ===== Generate quarterly information (YYYY-MM → YYYY-QX) =====
    df["year_quarter"] = pd.to_datetime(df["year_month"] + "-01").dt.to_period("Q").astype(str)
    df["year_quarter"] = df["year_quarter"].str.replace("Q", "-Q", regex=False)  # → "2020Q1" → "2020-Q1"

    # ===== Calculate quarterly averages =====
    df_quarterly = df.groupby(["country", "year_quarter"], as_index=False).mean(numeric_only=True)

    return df_quarterly


def process_central_bank_policy_rate(df: pd.DataFrame) -> pd.DataFrame:
    # ===== Extract only necessary columns =====
    keep_columns = ["FREQ", "REF_AREA", "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]
    df = df[keep_columns]

    # ===== Column name conversion =====
    df = df.rename(columns={
        "FREQ": "frequency",
        "REF_AREA": "country",
        "TIME_PERIOD": "year_month",
        "OBS_VALUE": "obs_value",
        "OBS_STATUS": "obs_status"
    })

    # ===== Value conversion (numeric) =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== Conditional filtering =====
    df = df[
        (df["frequency"] == "M") &
        (df["obs_status"] != "M")
    ]

    # ===== Generate quarterly information (YYYY-MM → YYYY-QX) =====
    df["year_quarter"] = pd.to_datetime(df["year_month"] + "-01").dt.to_period("Q").astype(str)
    df["year_quarter"] = df["year_quarter"].str.replace("Q", "-Q", regex=False)  # → "2020Q1" → "2020-Q1"

    # ===== Calculate quarterly maximum values ===== # Note: interpretation needs review
    df_quarterly = df.groupby(["country", "year_quarter"], as_index=False).max(numeric_only=True)
    
    return df_quarterly