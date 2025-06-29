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
    """リトライ機能付きのHTTPセッションを作成"""
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
    """API設定の妥当性を検証"""
    required_fields = ["name", "url", "filename"]
    for field in required_fields:
        if not api.get(field):
            logger.error(f"API設定に必須フィールド '{field}' がありません: {api}")
            return False
    return True

def fetch_from_api(url: str, output_path: str, name: str) -> Optional[pd.DataFrame]:
    """APIからデータを取得してCSVに保存"""
    try:
        session = create_session_with_retry()
        logger.info(f"🌐 {name} データ取得中（APIアクセス）...")
        
        # タイムアウト設定でAPIアクセス
        response = session.get(url, timeout=60)
        response.raise_for_status()
        
        # CSVデータとして読み込み
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), low_memory=False)
        
        if df.empty:
            logger.warning(f"{name}: 空のデータが返されました")
            return None
            
        # ファイル保存
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 保存完了: {output_path} ({len(df)}行)")
        
        return df
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ {name}: APIアクセスがタイムアウトしました")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ {name}: API接続に失敗しました")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ {name}: HTTPエラー {e.response.status_code}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"❌ {name}: CSVデータが空です")
        return None
    except Exception as e:
        logger.error(f"❌ {name}: 予期しないエラー: {e}")
        return None

def load_from_csv(output_path: str, name: str) -> Optional[pd.DataFrame]:
    """CSVファイルからデータを読み込み"""
    try:
        if not Path(output_path).exists():
            logger.error(f"❌ {name}: ファイルが存在しません: {output_path}")
            return None
            
        logger.info(f"🧪 [TEST] {name} をCSVから読み込み中...")
        df = pd.read_csv(output_path, low_memory=False)
        
        if df.empty:
            logger.warning(f"⚠️ {name}: CSVファイルが空です")
            return None
            
        logger.info(f"✅ 読み込み完了: {name} ({len(df)}行)")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"❌ {name}: CSVファイルが空または不正です")
        return None
    except Exception as e:
        logger.error(f"❌ {name}: CSV読み込みエラー: {e}")
        return None

def fetch_bis_datasets(api_list: List[Dict], output_dir: str, test_mode: bool = False) -> Dict[str, pd.DataFrame]:
    """
    BISデータを取得して辞書形式で返す。
    テストモードでは、既存のCSVファイルから読み込む。

    Parameters:
    - api_list: List[Dict] ← YAMLで定義されたBIS API設定
    - output_dir: 保存・読み込み先フォルダ
    - test_mode: TrueならCSVファイルから読み込み（APIアクセスなし）

    Returns:
    - Dict[str, pd.DataFrame]: {name: DataFrame}
    
    Raises:
    - ValueError: API設定が不正な場合
    - OSError: ディレクトリ作成に失敗した場合
    """
    if not api_list:
        raise ValueError("API設定リストが空です")
        
    # 出力ディレクトリ作成
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"ディレクトリ作成に失敗: {output_dir}")
        raise
    
    df_dict = {}
    successful_downloads = 0
    
    for api in api_list:
        # API設定検証
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
                logger.warning(f"⚠️ {name}: データの取得/読み込みに失敗")
                
        except Exception as e:
            logger.error(f"❌ {name}: 処理中にエラーが発生: {e}")
            continue
    
    logger.info(f"データ取得完了: {successful_downloads}/{len(api_list)} 成功")
    
    if not df_dict:
        raise ValueError("すべてのBISデータの取得に失敗しました")
    
    return df_dict


def validate_bis_dataframe(df: pd.DataFrame, name: str) -> bool:
    """
    BISデータの基本的な妥当性を検証
    
    Parameters:
    - df: 検証対象のDataFrame
    - name: データセット名
    
    Returns:
    - bool: 妥当性チェック結果
    """
    try:
        # 基本チェック
        if df.empty:
            logger.warning(f"{name}: データが空です")
            return False
            
        # 必要最小限の列数チェック
        if len(df.columns) < 3:
            logger.warning(f"{name}: 列数が少なすぎます ({len(df.columns)}列)")
            return False
            
        # 一般的なBISデータの必須列チェック
        expected_columns = ["TIME_PERIOD", "OBS_VALUE"]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"{name}: 必須列が不足: {missing_columns}")
            return False
            
        # データ型チェック
        if "OBS_VALUE" in df.columns:
            numeric_count = pd.to_numeric(df["OBS_VALUE"], errors="coerce").notna().sum()
            if numeric_count == 0:
                logger.warning(f"{name}: OBS_VALUE列に数値データがありません")
                return False
                
        logger.info(f"✅ {name}: データ妥当性チェック通過 ({len(df)}行, {len(df.columns)}列)")
        return True
        
    except Exception as e:
        logger.error(f"❌ {name}: 妥当性チェック中にエラー: {e}")
        return False
