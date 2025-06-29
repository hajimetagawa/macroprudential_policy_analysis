import pandas as pd
from functools import reduce
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def merge_datasets_var(
    df_dict: Dict[str, pd.DataFrame], 
    on: List[str], 
    how: str = 'outer'
) -> pd.DataFrame:
    """
    複数のデータセットをマージして分析用データセットを構築
    
    Parameters:
    - df_dict: マージ対象のDataFrame辞書
    - on: マージキーの列名リスト
    - how: マージ方法（'outer', 'inner', 'left', 'right'）
    
    Returns:
    - pd.DataFrame: マージされたデータセット
    
    Raises:
    - ValueError: マージキーが不正な場合
    """
    try:
        if not df_dict:
            raise ValueError("マージ対象のデータセットが空です")
            
        if not on:
            raise ValueError("マージキーが指定されていません")
        
        logger.info(f"{len(df_dict)}個のデータセットをマージ中... (キー: {on}, 方法: {how})")
        
        df_list = []
        total_memory_usage = 0

        for name, df in df_dict.items():
            if df.empty:
                logger.warning(f"{name}: 空のデータセットをスキップ")
                continue
                
            # メモリ使用量チェック
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            total_memory_usage += memory_mb
            
            # 必須キーの存在チェック
            missing_keys = [key for key in on if key not in df.columns]
            if missing_keys:
                logger.error(f"{name}: マージキーが不足 - {missing_keys}")
                continue
            
            df_copy = df.copy()

            # マージキー以外にのみプレフィックスを付与
            rename_dict = {
                col: f"{name}_{col}" for col in df_copy.columns if col not in on
            }
            df_copy = df_copy.rename(columns=rename_dict)
            df_list.append(df_copy)
            
            logger.debug(f"{name}: {len(df_copy)}行, {memory_mb:.1f}MB")
        
        if not df_list:
            raise ValueError("マージ可能なデータセットがありません")
            
        logger.info(f"総メモリ使用量: {total_memory_usage:.1f}MB")
        
        # 効率的なマージ処理
        if len(df_list) == 1:
            df_var = df_list[0]
        else:
            df_var = reduce(
                lambda left, right: pd.merge(left, right, on=on, how=how, suffixes=('', '_dup')), 
                df_list
            )
        
        # 重複列の削除
        dup_columns = [col for col in df_var.columns if col.endswith('_dup')]
        if dup_columns:
            logger.warning(f"重複列を削除: {dup_columns}")
            df_var = df_var.drop(columns=dup_columns)

        # 列名の正規化
        if "country" in df_var.columns and "country_code" not in df_var.columns:
            df_var = df_var.rename(columns={"country": "country_code"})
        
        logger.info(f"マージ完了: {len(df_var)}行, {len(df_var.columns)}列")
        return df_var
        
    except Exception as e:
        logger.error(f"データセットマージエラー: {e}")
        raise



