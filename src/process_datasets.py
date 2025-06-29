import pandas as pd
import os
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def dataset_processor(df_dict: Dict[str, pd.DataFrame], output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    複数のDataFrameをデータセット名ごとに整形し、processedフォルダに保存。

    Parameters:
    - df_dict: dict[str, pd.DataFrame] → nameをキーとしたRaw DataFrame群
    - output_dir: 加工後のCSVファイル保存先ディレクトリ

    Returns:
    - dict[str, pd.DataFrame]: 加工済みのDataFrame群（分析用）
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== 各データセットの加工処理をディスパッチ =====
    processed_dict = {}
    for name, df in df_dict.items():
        print(f"🔧 {name} を整形中...")

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
            print(f"⚠️ 未対応のデータセット: {name} → スキップ")
            continue

        # 保存
        save_path = os.path.join(output_dir, f"{name}.csv")
        df_proc.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ 保存完了: {save_path}")

        # 結果格納
        processed_dict[name] = df_proc

    return processed_dict



def validate_required_columns(df: pd.DataFrame, required_columns: list, dataset_name: str) -> bool:
    """必須列の存在をチェック"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"{dataset_name}: 必須列が不足 - {missing_columns}")
        return False
    return True

def safe_numeric_conversion(series: pd.Series, column_name: str) -> pd.Series:
    """安全な数値変換"""
    try:
        converted = pd.to_numeric(series, errors="coerce")
        null_count = converted.isnull().sum()
        if null_count > 0:
            logger.warning(f"{column_name}: {null_count}個の値を数値変換できませんでした")
        return converted.fillna(0)
    except Exception as e:
        logger.error(f"{column_name}: 数値変換エラー - {e}")
        return series.fillna(0)

def process_credit_gap(df: pd.DataFrame) -> pd.DataFrame:
    """クレジットギャップデータの処理"""
    try:
        # ===== 必要なカラムのみ抽出 =====
        required_columns = ["BORROWERS_CTY", "CG_DTYPE", "TIME_PERIOD", "OBS_VALUE"]
        if not validate_required_columns(df, required_columns, "credit_gap"):
            raise ValueError("クレジットギャップ: 必須列が不足")
            
        df = df[required_columns].copy()

        # ===== 列名の変換 =====
        df = df.rename(columns={
            "BORROWERS_CTY": "country",
            "CG_DTYPE": "data_type",
            "TIME_PERIOD": "year_quarter",
            "OBS_VALUE": "obs_value"
        })

        # ===== 値の変換（数値化） =====
        df["obs_value"] = safe_numeric_conversion(df["obs_value"], "credit_gap_obs_value")

        # ===== データタイプのラベル変換 =====
        dtype_map = {
            "A": "actual",
            "B": "trend_hp_filter",
            "C": "trend_actual"
        }
        df["data_type"] = df["data_type"].replace(dtype_map)

        # ===== ワイド形式への変換 =====
        df = df.pivot(index=["country", "year_quarter"], columns="data_type", values="obs_value").reset_index()
        logger.info(f"クレジットギャップ処理完了: {len(df)}行")
        return df
            
    except Exception as e:
        logger.error(f"クレジットギャップ処理エラー: {e}")
        raise

def process_total_credit(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 必要なカラムのみ抽出 =====
    keep_columns = ["BORROWERS_CTY", "TC_BORROWERS", "TC_LENDERS", "VALUATION", "UNIT_TYPE",
                     "TC_ADJUST", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== 列名の変換 =====
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

    # ===== 値の変換（数値化） =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== ラベルマッピング =====
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

    # ===== 条件フィルター =====
    df = df[
        (df["unit_type"] == "pct_of_gdp") &
        (df["valuation_method"] == "market_value") &
        (df["adjustment"] == "adjusted_for_break")
    ]

    df_all_lender = df[df["lending_sector"] == "all_sectors"]
    df_domestic_banks_lender = df[df["lending_sector"] == "domestic_banks"]

    # ===== ワイド形式への変換 =====
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

    # ===== 横にマージ（列追加） =====
    df_merged = pd.merge(
        df_all_lender,
        df_domestic_banks_lender,
        on=["country", "year_quarter"],
        how="outer"
    )

    return df_merged

def process_debt_service_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 必要なカラムのみ抽出 =====
    keep_columns = ["BORROWERS_CTY", "DSR_BORROWERS", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== 列名の変換 =====
    df = df.rename(columns={
        "BORROWERS_CTY": "country",
        "DSR_BORROWERS": "dsr_borrowers",
        "TIME_PERIOD": "year_quarter",
        "OBS_VALUE": "obs_value"
    })

    # ===== 値の変換（数値化） =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== データタイプのラベル変換 =====
    dsr_borrowers_map = {
        "H": "households_and_npishs",
        "N": "non_financial_corporations",
        "P": "private_non_financial_sector"
    }
    df["dsr_borrowers"] = df["dsr_borrowers"].replace(dsr_borrowers_map)

    # ===== ワイド形式への変換 =====
    df = df.pivot(
        index=["country", "year_quarter"], 
        columns="dsr_borrowers", 
        values="obs_value"
        ).reset_index()

    return df

def process_residential_property_price(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 必要なカラムのみ抽出 =====
    keep_columns = ["REF_AREA", "VALUE", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    df["UNIT_MEASURE"] = df["UNIT_MEASURE"].astype("str")   

    # ===== 列名の変換 =====
    df = df.rename(columns={
        "REF_AREA": "country",
        "VALUE": "value",
        "UNIT_MEASURE": "unit_of_measure",
        "TIME_PERIOD": "year_quarter",
        "OBS_VALUE": "obs_value"
    })

    # ===== 値の変換（数値化） =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== データタイプのラベル変換 =====
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

    # ===== 条件フィルター =====
    df = df[df["value"] == "real"]

    # ===== ワイド形式への変換 =====
    df = df.pivot(
        index=["country", "year_quarter"], 
        columns="unit_of_measure", 
        values="obs_value"
        ).reset_index()

    return df

def process_commercial_property_price(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 必要なカラムのみ抽出 =====
    keep_columns = ["FREQ", "REF_AREA", "COVERED_AREA", "RE_TYPE", "COMPILING_ORG", "PRICED_UNIT", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== 列名の変換 =====
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

    # ===== 値の変換（数値化） =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== データタイプのラベル変換 =====
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

    # ===== 条件フィルター =====
    df = df[
        (df["frequency"] == "Q") &
        (df["unit_of_measure"] != "USD") &
        (df["unit_of_measure"] != "AED") &
        (df["unit_of_measure"] != "PHP")
    ]

    # ===== 四半期 × 国ごとの平均値を算出 =====
    df_grouped = df.groupby(["country", "year_quarter"], as_index=False)["obs_value"].mean()

    return df_grouped


def process_effective_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 必要なカラムのみ抽出 =====
    keep_columns = ["FREQ", "EER_TYPE", "EER_BASKET", "REF_AREA", "TIME_PERIOD", "OBS_VALUE"]
    df = df[keep_columns]

    # ===== 列名の変換 =====
    df = df.rename(columns={
        "FREQ": "frequency",
        "EER_TYPE": "eer_type",
        "EER_BASKET": "eer_basket",
        "RE_TYPE": "real_estate_type",
        "REF_AREA": "country",
        "TIME_PERIOD": "year_month",
        "OBS_VALUE": "obs_value"
    })

    # ===== 値の変換（数値化） =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== データタイプのラベル変換 =====
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

    # ===== 条件フィルター =====
    df = df[
        (df["frequency"] == "M") &
        (df["eer_basket"] == "broad")
    ]

    # ===== ワイド形式への変換 =====
    df = df.pivot(
        index=["country", "year_month"], 
        columns="eer_type", 
        values="obs_value"
        ).reset_index()

    # ===== 四半期情報を生成（YYYY-MM → YYYY-QX） =====
    df["year_quarter"] = pd.to_datetime(df["year_month"] + "-01").dt.to_period("Q").astype(str)
    df["year_quarter"] = df["year_quarter"].str.replace("Q", "-Q", regex=False)  # → "2020Q1" → "2020-Q1"

    # ===== 四半期単位で平均を計算 =====
    df_quarterly = df.groupby(["country", "year_quarter"], as_index=False).mean(numeric_only=True)

    return df_quarterly


def process_central_bank_policy_rate(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 必要なカラムのみ抽出 =====
    keep_columns = ["FREQ", "REF_AREA", "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]
    df = df[keep_columns]

    # ===== 列名の変換 =====
    df = df.rename(columns={
        "FREQ": "frequency",
        "REF_AREA": "country",
        "TIME_PERIOD": "year_month",
        "OBS_VALUE": "obs_value",
        "OBS_STATUS": "obs_status"
    })

    # ===== 値の変換（数値化） =====
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce").fillna(0)

    # ===== 条件フィルター =====
    df = df[
        (df["frequency"] == "M") &
        (df["obs_status"] != "M")
    ]

    # ===== 四半期情報を生成（YYYY-MM → YYYY-QX） =====
    df["year_quarter"] = pd.to_datetime(df["year_month"] + "-01").dt.to_period("Q").astype(str)
    df["year_quarter"] = df["year_quarter"].str.replace("Q", "-Q", regex=False)  # → "2020Q1" → "2020-Q1"

    # ===== 四半期単位で最大値を計算 =====  <=解釈については要検討
    df_quarterly = df.groupby(["country", "year_quarter"], as_index=False).max(numeric_only=True)
    
    return df_quarterly