import sys
import os
import yaml
import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from src.iMaPP_data_loader import load_mapp_excel
from src.fetch_api_data import fetch_bis_datasets
from src.iMaPP_transformer import transform_imapp_data, prepare_imapp_for_analysis
from src.yaml_loader import load_yaml
from src.process_datasets import dataset_processor
from src.build_dataset import merge_datasets_var
from src.dataset_assessment import assess_data_quality

def setup_logging(config: Dict) -> None:
    """Initialize logging configuration"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    log_dir = config.get("paths", {}).get("logs", "../logs")
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# ================== ステップ0: 各データの取り込み ==================

def load_config() -> Dict:
    """Load configuration file"""
    try:
        config = load_yaml("config")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise

def ensure_directories(config: Dict) -> None:
    """Create necessary directories"""
    paths = config.get("paths", {})
    directories = [
        paths.get("src_data", "../data/raw"),
        paths.get("processed_data", "../data/processed"),
        paths.get("dataset_output", "../data/dataset"),
        paths.get("logs", "../logs"),
        paths.get("results", "../results")
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Directory creation completed")

def load_imapp_data(config: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load iMaPP data"""
    try:
        logger.info("[1.1/4] Loading iMaPP data...")
        
        input_config = config.get("input", {})
        paths_config = config.get("paths", {})
        
        file_pattern = input_config.get("file_pattern", "iMaPP_database*.xlsx")
        sheet_t = input_config.get("sheet_name_tightening", "MaPP_T")
        sheet_l = input_config.get("sheet_name_loosening", "MaPP_L")
        save_dir = paths_config.get("src_data", "../data/raw")
        
        # Check file existence
        import glob
        files = glob.glob(f"../{file_pattern}")
        if not files:
            raise FileNotFoundError(f"iMaPP file not found: {file_pattern}")
        
        df_iMaPP_tightening = load_mapp_excel(f"../{file_pattern}", sheet_t, save_dir, "MaPP_T.csv")
        df_iMaPP_loosening = load_mapp_excel(f"../{file_pattern}", sheet_l, save_dir, "MaPP_L.csv")
        
        logger.info(f"iMaPP data loading completed: T={len(df_iMaPP_tightening)} rows, L={len(df_iMaPP_loosening)} rows")
        return df_iMaPP_tightening, df_iMaPP_loosening
        
    except Exception as e:
        logger.error(f"iMaPP data loading error: {e}")
        raise


def load_bis_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """BISデータを読み込む"""
    try:
        logger.info("[1.2/4] BIS APIデータをダウンロード中...")
        
        data_sources = load_yaml("data_sources")
        bis_apis = data_sources.get("bis_apis", [])
        if not bis_apis:
            raise ValueError("BIS API設定が見つかりません")
            
        output_dir = config.get("paths", {}).get("src_data", "../data/raw")
        
        df_dict_bis = fetch_bis_datasets(api_list=bis_apis, output_dir=output_dir, test_mode=True)
        
        if not df_dict_bis:
            raise ValueError("BISデータの取得に失敗しました")
            
        logger.info(f"BISデータ取得完了: {len(df_dict_bis)}データセット")
        return df_dict_bis
        
    except Exception as e:
        logger.error(f"BISデータ取得エラー: {e}")
        raise


#その他データを追加する場合は、このパートで取得プロセスを記述


def process_imapp_data(df_tightening: pd.DataFrame, df_loosening: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """iMaPPデータを処理する"""
    try:
        logger.info("[2.1/4] iMaPPデータセットの加工...")
        
        df_iMaPP_t_transformed = transform_imapp_data(df_tightening)
        df_iMaPP_l_transformed = transform_imapp_data(df_loosening)
        
        df_imapp_t_for_analysis = prepare_imapp_for_analysis(df_iMaPP_t_transformed)
        df_imapp_l_for_analysis = prepare_imapp_for_analysis(df_iMaPP_l_transformed)
        
        # データ保存
        output_config = config.get("output", {})
        dataset_dir = config.get("paths", {}).get("dataset_output", "../data/dataset")
        output_file = output_config.get("imapp_dataset_t", "imapp_dataset_t.csv")
        output_path = Path(dataset_dir) / output_file
        df_imapp_t_for_analysis.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        logger.info(f"iMaPPデータ処理完了: {len(df_imapp_t_for_analysis)}行")
        return df_imapp_t_for_analysis
        
    except Exception as e:
        logger.error(f"iMaPPデータ処理エラー: {e}")
        raise

def process_bis_data(df_dict_bis: Dict[str, pd.DataFrame], config: Dict) -> Dict[str, pd.DataFrame]:
    """BISデータを処理する"""
    try:
        logger.info("[2.2/4] BISデータセットの加工...")
        
        processed_dir = config.get("paths", {}).get("processed_data", "../data/processed/")
        processed_dict_bis = dataset_processor(df_dict=df_dict_bis, output_dir=processed_dir)
        
        logger.info(f"BISデータ処理完了: {len(processed_dict_bis)}データセット")
        return processed_dict_bis
        
    except Exception as e:
        logger.error(f"BISデータ処理エラー: {e}")
        raise


def build_analysis_dataset(df_imapp_t_for_analysis: pd.DataFrame, processed_dict_bis: Dict[str, pd.DataFrame], config: Dict) -> pd.DataFrame:
    """分析用データベースを構築する"""
    try:
        logger.info("[3/4] 分析用データベースの構築...")
        
        df_var = merge_datasets_var(processed_dict_bis, on=["country", "year_quarter"])
        
        # データマージ
        df_imapp_analysis_dataset = pd.merge(
            df_imapp_t_for_analysis,
            df_var,
            on=["country_code", "year_quarter"],
            how="left"
        )
        
        # データ保存
        output_config = config.get("output", {})
        dataset_dir = config.get("paths", {}).get("dataset_output", "../data/dataset")
        output_file = output_config.get("imapp_analysis_dataset", "imapp_analysis_dataset_.csv")
        output_path = Path(dataset_dir) / output_file
        df_imapp_analysis_dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        # データ品質評価
        assess_data_quality(df_imapp_analysis_dataset)
        
        logger.info(f"分析用データベース構築完了: {len(df_imapp_analysis_dataset)}行")
        return df_imapp_analysis_dataset
        
    except Exception as e:
        logger.error(f"分析用データベース構築エラー: {e}")
        raise

def apply_country_scope(df_imapp_analysis_dataset: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """国別スコーピングを適用する"""
    try:
        logger.info("[4/4] 国別スコーピング適用中...")
        
        # 国別設定読み込み
        with open("../config/countries.yaml", "r") as f:
            yaml_data = yaml.safe_load(f)
        
        country_scope = yaml_data.get('countries_scope', {})
        if not country_scope:
            raise ValueError("国別設定が見つかりません")
        
        # 属性情報DataFrame作成
        records = []
        for iso3, info in country_scope.items():
            row = {
                "iso2": info["iso2"],
                "country_name": info["name"],
                "classification": info["classification"],
                "region": info["region"],
                "eu_member": info["eu_member"],
                "g20_member": info["g20_member"]
            }
            records.append(row)
        
        attr_df = pd.DataFrame(records)
        
        # フィルタリングとマージ
        iso2_list = attr_df['iso2'].tolist()
        filtered_df = df_imapp_analysis_dataset[df_imapp_analysis_dataset['country_code'].isin(iso2_list)].copy()
        
        merged_df = filtered_df.merge(attr_df, left_on='country_code', right_on='iso2', how='left')
        
        # 結果保存
        output_config = config.get("output", {})
        dataset_dir = config.get("paths", {}).get("dataset_output", "../data/dataset")
        output_file = output_config.get("final_dataset", "final.csv")
        output_path = Path(dataset_dir) / output_file
        merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        logger.info(f"国別スコーピング完了: {len(merged_df)}行, {len(iso2_list)}カ国")
        logger.info(f"対象国: {merged_df[['country_code', 'country_name', 'classification', 'region']].drop_duplicates().to_string()}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"国別スコーピングエラー: {e}")
        raise


def main():
    """メイン処理"""
    try:
        # 設定読み込み
        config = load_config()
        
        # ログ設定
        setup_logging(config)
        
        logger.info("=== マクロプルーデンシャル政策分析パイプライン開始 ===")
        
        # ディレクトリ準備
        ensure_directories(config)
        
        # ステップ1: データ読み込み
        df_iMaPP_tightening, df_iMaPP_loosening = load_imapp_data(config)
        df_dict_bis = load_bis_data(config)
        
        # ステップ2: データ処理
        df_imapp_t_for_analysis = process_imapp_data(df_iMaPP_tightening, df_iMaPP_loosening, config)
        processed_dict_bis = process_bis_data(df_dict_bis, config)
        
        # ステップ3: 分析用データベース構築
        df_imapp_analysis_dataset = build_analysis_dataset(df_imapp_t_for_analysis, processed_dict_bis, config)
        
        # ステップ4: 国別スコーピング
        final_df = apply_country_scope(df_imapp_analysis_dataset, config)
        
        logger.info("=== パイプライン完了 ===")
        logger.info(f"最終データセット: {len(final_df)}行")
        
        return final_df
        
    except Exception as e:
        logger.error(f"パイプライン実行エラー: {e}")
        raise

if __name__ == "__main__":
    main()

