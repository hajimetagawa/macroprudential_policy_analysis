# src/config_loader.py
import yaml
import os
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def load_yaml(yaml_name: str, config_dir: str = "../config") -> Dict[str, Any]:
    """
    YAML設定ファイルを読み込む
    
    Parameters:
    - yaml_name: YAMLファイル名（拡張子なし）
    - config_dir: 設定ファイルディレクトリ
    
    Returns:
    - Dict[str, Any]: YAMLデータ
    
    Raises:
    - FileNotFoundError: ファイルが存在しない場合
    - yaml.YAMLError: YAMLパースエラー
    """
    try:
        file_path = Path(config_dir) / f"{yaml_name}.yaml"
        
        if not file_path.exists():
            raise FileNotFoundError(f"YAMLファイルが存在しません: {file_path}")
        
        logger.debug(f"YAMLファイル読み込み中: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if data is None:
            logger.warning(f"YAMLファイルが空です: {file_path}")
            return {}
            
        logger.debug(f"YAML読み込み完了: {yaml_name}")
        return data
        
    except yaml.YAMLError as e:
        logger.error(f"YAMLパースエラー: {file_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"YAML読み込みエラー: {file_path} - {e}")
        raise
