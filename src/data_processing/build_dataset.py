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
    Merge multiple datasets to build analytical dataset
    
    Parameters:
    - df_dict: DataFrame dictionary for merging
    - on: List of merge key column names
    - how: Merge method ('outer', 'inner', 'left', 'right')
    
    Returns:
    - pd.DataFrame: Merged dataset
    
    Raises:
    - ValueError: When merge keys are invalid
    """
    try:
        if not df_dict:
            raise ValueError("Dataset dictionary for merging is empty")
            
        if not on:
            raise ValueError("Merge keys not specified")
        
        logger.info(f"Merging {len(df_dict)} datasets... (keys: {on}, method: {how})")
        
        df_list = []
        total_memory_usage = 0

        for name, df in df_dict.items():
            if df.empty:
                logger.warning(f"{name}: Skipping empty dataset")
                continue
                
            # Memory usage check
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            total_memory_usage += memory_mb
            
            # Required key existence check
            missing_keys = [key for key in on if key not in df.columns]
            if missing_keys:
                logger.error(f"{name}: Missing merge keys - {missing_keys}")
                continue
            
            df_copy = df.copy()

            # Add prefix only to non-merge key columns
            rename_dict = {
                col: f"{name}_{col}" for col in df_copy.columns if col not in on
            }
            df_copy = df_copy.rename(columns=rename_dict)
            df_list.append(df_copy)
            
            logger.debug(f"{name}: {len(df_copy)}行, {memory_mb:.1f}MB")
        
        if not df_list:
            raise ValueError("No datasets available for merging")
            
        logger.info(f"Total memory usage: {total_memory_usage:.1f}MB")
        
        # Efficient merge processing
        if len(df_list) == 1:
            df_var = df_list[0]
        else:
            df_var = reduce(
                lambda left, right: pd.merge(left, right, on=on, how=how, suffixes=('', '_dup')), 
                df_list
            )
        
        # Remove duplicate columns
        dup_columns = [col for col in df_var.columns if col.endswith('_dup')]
        if dup_columns:
            logger.warning(f"Removing duplicate columns: {dup_columns}")
            df_var = df_var.drop(columns=dup_columns)

        # Column name normalization
        if "country" in df_var.columns and "country_code" not in df_var.columns:
            df_var = df_var.rename(columns={"country": "country_code"})
        
        logger.info(f"Merge complete: {len(df_var)} rows, {len(df_var.columns)} columns")
        return df_var
        
    except Exception as e:
        logger.error(f"Dataset merge error: {e}")
        raise



