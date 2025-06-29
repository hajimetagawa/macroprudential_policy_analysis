import pandas as pd
import os
import yaml
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def month_to_quarter(month: int) -> str:
    """
    月 → 四半期(Q1～Q4)への変換
    
    Parameters:
    - month: 月（1-12）
    
    Returns:
    - str: 四半期文字列（Q1, Q2, Q3, Q4, Q?）
    """
    if not isinstance(month, (int, float)):
        logger.warning(f"不正な月データ: {month} (type: {type(month)})")
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
        logger.warning(f"不正な月値: {month}")
        return "Q?"


def transform_imapp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    iMaPPデータを整形（Wide→Long、異常値処理、列名リネーム、四半期追加）

    Parameters:
    - df: Wide形式の入力DataFrame

    Returns:
    - Long形式に整形されたDataFrame
    """

    # ===== 設定定義 =====
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

    # ===== カラム名クレンジング（_T, _Lの除去） =====
    df.columns = [col.replace("_T", "").replace("_L", "") for col in df.columns]

    # ===== 必要カラムの抽出（+ 指標列） =====
    df = df[[col for col in reserved_cols_base if col in df.columns] + 
            [col for col in df.columns if col not in reserved_cols_base]]

    # ===== 異常値クリーニング（Wide形式） =====
    for col in df.columns:
        if col not in reserved_cols_base:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ===== Wide → Long変換（mapp_tool列を追加） =====
    df_long = df.melt(
        id_vars=reserved_cols_base,
        var_name="mapp_tool",
        value_name="obs"
    )

    # ===== 四半期列の追加 =====
    df_long["Quarter"] = df_long["Year"].astype(str) + "-" + df_long["Month"].apply(month_to_quarter)

    # ===== 列名変換 =====
    df_long = df_long.rename(columns=rename_map)

    # ===== 列順整形（存在する列のみ） =====
    df_long = df_long[[col for col in output_columns if col in df_long.columns]]

    return df_long


def validate_imapp_dataframe(df: pd.DataFrame) -> bool:
    """
    iMaPP DataFrameの基本的な妥当性を検証
    
    Parameters:
    - df: 検証対象のDataFrame
    
    Returns:
    - bool: 妥当性チェック結果
    """
    try:
        if df.empty:
            logger.error("iMaPP DataFrameが空です")
            return False
            
        required_columns = ["country", "year", "month", "mapp_tool", "obs"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"iMaPP必須列が不足: {missing_columns}")
            return False
            
        # データ型チェック
        if not pd.api.types.is_numeric_dtype(df["year"]):
            logger.warning("年データが数値型ではありません")
            
        if not pd.api.types.is_numeric_dtype(df["month"]):
            logger.warning("月データが数値型ではありません")
            
        if not pd.api.types.is_numeric_dtype(df["obs"]):
            logger.warning("観測値データが数値型ではありません")
            
        logger.info(f"iMaPPデータ妥当性チェック通過 ({len(df)}行, {len(df.columns)}列)")
        return True
        
    except Exception as e:
        logger.error(f"iMaPPデータ妥当性チェックエラー: {e}")
        return False

def prepare_imapp_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    iMaPPデータを分析用に整形する
    
    Parameters:
    - df: Long形式のiMaPP DataFrame
    
    Returns:
    - pd.DataFrame: 分析用に集約されたDataFrame
    
    Raises:
    - ValueError: データ妥当性エラー
    """
    try:
        # データ妥当性チェック
        if not validate_imapp_dataframe(df):
            raise ValueError("iMaPPデータの妥当性チェックに失敗")
            
        df = df.copy()

        # ===== 1. "SUM_17" のレコードを除外 =====
        initial_count = len(df)
        df = df[df["mapp_tool"] != "SUM_17"]
        filtered_count = len(df)
        logger.info(f"SUM_17レコードを除外: {initial_count} → {filtered_count}行")

        # ===== 2. 月次 → 四半期末日付に変換（quarter_end列を追加）=====
        try:
            df["quarter_end"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
            df["quarter_end"] = df["quarter_end"] + pd.offsets.QuarterEnd(0)
        except Exception as e:
            logger.error(f"四半期日付変換エラー: {e}")
            raise

        # ===== 3. 四半期集計（Aggregate + Binary）=====
        try:
            grouped = df.groupby([
                "country", "country_code", "mapp_tool", 
                "advanced", "emerging", "year", "year_quarter"
            ]).agg(
                obs_agg=("obs", "sum"),    # 強度（実施回数）
                obs_bin=("obs", "max")     # 実施されたかどうか（1つでもあれば1）
            ).reset_index()
            
            logger.info(f"四半期集計完了: {len(grouped)}行")
            return grouped
            
        except Exception as e:
            logger.error(f"四半期集計エラー: {e}")
            raise
            
    except Exception as e:
        logger.error(f"iMaPP分析データ準備エラー: {e}")
        raise
