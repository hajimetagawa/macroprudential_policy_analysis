# 🔧 src/data_loader.py
import pandas as pd
import os
import glob
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_mapp_excel(
    src_path: str, 
    sheet_name: str, 
    save_dir: Optional[str] = None, 
    save_name: Optional[str] = None
) -> pd.DataFrame:
    """
    iMaPP Excelファイルを読み込み、CSVとして保存する
    
    Parameters:
    - src_path: ファイルパスパターン（glob形式）
    - sheet_name: Excelシート名
    - save_dir: 保存先ディレクトリ（オプション）
    - save_name: 保存ファイル名（オプション）
    
    Returns:
    - pd.DataFrame: 読み込み済みデータ
    
    Raises:
    - FileNotFoundError: ファイルが見つからない場合
    - ValueError: シートが存在しない場合
    """
    try:
        # ファイル検索
        files = glob.glob(src_path)
        if not files:
            raise FileNotFoundError(f"ファイルが見つかりません: {src_path}")
        
        file_path = files[0]
        logger.info(f"iMaPPファイルを読み込み中: {file_path}, シート: {sheet_name}")
        
        # Excel読み込み
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError as e:
            if "Worksheet" in str(e):
                raise ValueError(f"シート '{sheet_name}' が存在しません: {file_path}")
            raise
        
        if df.empty:
            logger.warning(f"空のデータが読み込まれました: {sheet_name}")
        
        # 欠损値を0で埋める
        df = df.fillna(0)
        
        # CSV保存
        if save_dir and save_name:
            save_path = Path(save_dir) / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            logger.info(f"✔ データ保存完了: {save_path} ({len(df)}行)")
        
        return df
        
    except Exception as e:
        logger.error(f"iMaPPデータ読み込みエラー: {e}")
        raise
