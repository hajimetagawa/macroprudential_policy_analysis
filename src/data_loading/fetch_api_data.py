import pandas as pd
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

def create_session_with_retry() -> requests.Session:
    """Create HTTP session with retry functionality"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def validate_api_config(api: Dict) -> bool:
    """Validate API configuration validity"""
    required_fields = ["name", "url", "filename"]
    for field in required_fields:
        if not api.get(field):
            logger.error(f"Required field '{field}' missing in API configuration: {api}")
            return False
    return True

def fetch_from_api(url: str, output_path: str, name: str) -> Optional[pd.DataFrame]:
    """Fetch data from API and save to CSV"""
    try:
        session = create_session_with_retry()
        logger.info(f"🌐 Fetching {name} data (API access)...")
        
        # API access with timeout setting
        response = session.get(url, timeout=60)
        response.raise_for_status()
        
        # Read as CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), low_memory=False)
        
        if df.empty:
            logger.warning(f"{name}: Empty data returned")
            return None
            
        # Save file
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ Save complete: {output_path} ({len(df)} rows)")
        
        return df
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ {name}: API access timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ {name}: API connection failed")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ {name}: HTTP error {e.response.status_code}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"❌ {name}: CSV data is empty")
        return None
    except Exception as e:
        logger.error(f"❌ {name}: Unexpected error: {e}")
        return None

def load_from_csv(output_path: str, name: str) -> Optional[pd.DataFrame]:
    """Load data from CSV file"""
    try:
        if not Path(output_path).exists():
            logger.error(f"❌ {name}: File does not exist: {output_path}")
            return None
            
        logger.info(f"🧪 [TEST] Loading {name} from CSV...")
        df = pd.read_csv(output_path, low_memory=False)
        
        if df.empty:
            logger.warning(f"⚠️ {name}: CSV file is empty")
            return None
            
        logger.info(f"✅ Load complete: {name} ({len(df)} rows)")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"❌ {name}: CSV file is empty or invalid")
        return None
    except Exception as e:
        logger.error(f"❌ {name}: CSV loading error: {e}")
        return None

def fetch_bis_datasets(api_list: List[Dict], output_dir: str, test_mode: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch BIS data and return in dictionary format.
    In test mode, load from existing CSV files.

    Parameters:
    - api_list: List[Dict] ← BIS API configuration defined in YAML
    - output_dir: Save/load destination folder
    - test_mode: If True, load from CSV files (no API access)

    Returns:
    - Dict[str, pd.DataFrame]: {name: DataFrame}
    
    Raises:
    - ValueError: When API configuration is invalid
    - OSError: When directory creation fails
    """
    if not api_list:
        raise ValueError("API configuration list is empty")
        
    # Create output directory
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Directory creation failed: {output_dir}")
        raise
    
    df_dict = {}
    successful_downloads = 0
    
    for api in api_list:
        # API configuration validation
        if not validate_api_config(api):
            continue
            
        name = api["name"]
        url = api["url"]
        filename = api["filename"]
        output_path = Path(output_dir) / filename
        
        try:
            if test_mode:
                df = load_from_csv(str(output_path), name)
            else:
                df = fetch_from_api(url, str(output_path), name)
            
            if df is not None:
                df_dict[name] = df
                successful_downloads += 1
            else:
                logger.warning(f"⚠️ {name}: Data fetch/load failed")
                
        except Exception as e:
            logger.error(f"❌ {name}: Error occurred during processing: {e}")
            continue
    
    logger.info(f"Data fetch complete: {successful_downloads}/{len(api_list)} successful")
    
    if not df_dict:
        raise ValueError("Failed to fetch all BIS data")
    
    return df_dict


def validate_bis_dataframe(df: pd.DataFrame, name: str) -> bool:
    """
    Validate basic validity of BIS data
    
    Parameters:
    - df: DataFrame to validate
    - name: Dataset name
    
    Returns:
    - bool: Validity check result
    """
    try:
        # Basic check
        if df.empty:
            logger.warning(f"{name}: Data is empty")
            return False
            
        # Minimum required column count check
        if len(df.columns) < 3:
            logger.warning(f"{name}: Too few columns ({len(df.columns)} columns)")
            return False
            
        # General BIS data required column check
        expected_columns = ["TIME_PERIOD", "OBS_VALUE"]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"{name}: Missing required columns: {missing_columns}")
            return False
            
        # Data type check
        if "OBS_VALUE" in df.columns:
            numeric_count = pd.to_numeric(df["OBS_VALUE"], errors="coerce").notna().sum()
            if numeric_count == 0:
                logger.warning(f"{name}: No numeric data in OBS_VALUE column")
                return False
                
        logger.info(f"✅ {name}: Data validity check passed ({len(df)} rows, {len(df.columns)} columns)")
        return True
        
    except Exception as e:
        logger.error(f"❌ {name}: Error during validity check: {e}")
        return False
