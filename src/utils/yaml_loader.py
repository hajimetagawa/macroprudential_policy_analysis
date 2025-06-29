# src/config_loader.py
import yaml
import os
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def load_yaml(yaml_name: str, config_dir: str = "../config") -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Parameters:
    - yaml_name: YAML file name (without extension)
    - config_dir: Configuration file directory
    
    Returns:
    - Dict[str, Any]: YAML data
    
    Raises:
    - FileNotFoundError: When file does not exist
    - yaml.YAMLError: YAML parsing error
    """
    try:
        file_path = Path(config_dir) / f"{yaml_name}.yaml"
        
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file does not exist: {file_path}")
        
        logger.debug(f"Loading YAML file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if data is None:
            logger.warning(f"YAML file is empty: {file_path}")
            return {}
            
        logger.debug(f"YAML loading completed: {yaml_name}")
        return data
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {file_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"YAML loading error: {file_path} - {e}")
        raise
