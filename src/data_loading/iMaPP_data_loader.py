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
    Load iMaPP Excel file and save as CSV
    
    Parameters:
    - src_path: File path pattern (glob format)
    - sheet_name: Excel sheet name
    - save_dir: Save directory (optional)
    - save_name: Save file name (optional)
    
    Returns:
    - pd.DataFrame: Loaded data
    
    Raises:
    - FileNotFoundError: When file is not found
    - ValueError: When sheet does not exist
    """
    try:
        # File search
        files = glob.glob(src_path)
        if not files:
            raise FileNotFoundError(f"File not found: {src_path}")
        
        file_path = files[0]
        logger.info(f"Loading iMaPP file: {file_path}, sheet: {sheet_name}")
        
        # Excel loading
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError as e:
            if "Worksheet" in str(e):
                raise ValueError(f"Sheet '{sheet_name}' does not exist: {file_path}")
            raise
        
        if df.empty:
            logger.warning(f"Empty data loaded: {sheet_name}")
        
        # Fill missing values with 0
        df = df.fillna(0)
        
        # CSV save
        if save_dir and save_name:
            save_path = Path(save_dir) / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Data saved: {save_path} ({len(df)} rows)")
        
        return df
        
    except Exception as e:
        logger.error(f"iMaPP data loading error: {e}")
        raise
