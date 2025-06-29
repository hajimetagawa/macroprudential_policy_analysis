"""
Pipeline Orchestrator
Coordinates the entire data processing pipeline for macroprudential policy analysis
"""
import pandas as pd
import yaml
import glob
import logging
from typing import Dict, Tuple
from pathlib import Path

from .data_loading.iMaPP_data_loader import load_mapp_excel
from .data_loading.fetch_api_data import fetch_bis_datasets
from .data_processing.iMaPP_transformer import transform_imapp_data, prepare_imapp_for_analysis
from .utils.yaml_loader import load_yaml
from .data_processing.process_datasets import dataset_processor
from .data_processing.build_dataset import merge_datasets_var
from .analysis.dataset_assessment import assess_data_quality

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the complete data processing pipeline"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories"""
        paths = self.config.get("paths", {})
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
    
    def load_imapp_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load iMaPP data from Excel files"""
        try:
            logger.info("[1.1/4] Loading iMaPP data...")
            
            input_config = self.config.get("input", {})
            paths_config = self.config.get("paths", {})
            
            file_pattern = input_config.get("file_pattern", "iMaPP_database*.xlsx")
            sheet_t = input_config.get("sheet_name_tightening", "MaPP_T")
            sheet_l = input_config.get("sheet_name_loosening", "MaPP_L")
            save_dir = paths_config.get("src_data", "../data/raw")
            
            # Check file existence
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
    
    def load_bis_data(self) -> Dict[str, pd.DataFrame]:
        """Load BIS data from APIs"""
        try:
            logger.info("[1.2/4] Downloading BIS API data...")
            
            data_sources = load_yaml("data_sources")
            bis_apis = data_sources.get("bis_apis", [])
            if not bis_apis:
                raise ValueError("BIS API configuration not found")
                
            output_dir = self.config.get("paths", {}).get("src_data", "../data/raw")
            
            df_dict_bis = fetch_bis_datasets(api_list=bis_apis, output_dir=output_dir, test_mode=True)
            
            if not df_dict_bis:
                raise ValueError("Failed to fetch BIS data")
                
            logger.info(f"BIS data fetching completed: {len(df_dict_bis)} datasets")
            return df_dict_bis
            
        except Exception as e:
            logger.error(f"BIS data fetching error: {e}")
            raise
    
    def process_imapp_data(self, df_tightening: pd.DataFrame, df_loosening: pd.DataFrame) -> pd.DataFrame:
        """Process iMaPP data for analysis"""
        try:
            logger.info("[2.1/4] Processing iMaPP dataset...")
            
            df_iMaPP_t_transformed = transform_imapp_data(df_tightening)
            df_iMaPP_l_transformed = transform_imapp_data(df_loosening)
            
            df_imapp_t_for_analysis = prepare_imapp_for_analysis(df_iMaPP_t_transformed)
            df_imapp_l_for_analysis = prepare_imapp_for_analysis(df_iMaPP_l_transformed)
            
            # Save data
            output_config = self.config.get("output", {})
            dataset_dir = self.config.get("paths", {}).get("dataset_output", "../data/dataset")
            output_file = output_config.get("imapp_dataset_t", "imapp_dataset_t.csv")
            output_path = Path(dataset_dir) / output_file
            df_imapp_t_for_analysis.to_csv(output_path, index=False, encoding="utf-8-sig")
            
            logger.info(f"iMaPP data processing completed: {len(df_imapp_t_for_analysis)} rows")
            return df_imapp_t_for_analysis
            
        except Exception as e:
            logger.error(f"iMaPP data processing error: {e}")
            raise
    
    def process_bis_data(self, df_dict_bis: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process BIS data for analysis"""
        try:
            logger.info("[2.2/4] Processing BIS datasets...")
            
            processed_dir = self.config.get("paths", {}).get("processed_data", "../data/processed/")
            processed_dict_bis = dataset_processor(df_dict=df_dict_bis, output_dir=processed_dir)
            
            logger.info(f"BIS data processing completed: {len(processed_dict_bis)} datasets")
            return processed_dict_bis
            
        except Exception as e:
            logger.error(f"BIS data processing error: {e}")
            raise
    
    def build_analysis_dataset(self, df_imapp_t_for_analysis: pd.DataFrame, 
                             processed_dict_bis: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build comprehensive analysis dataset"""
        try:
            logger.info("[3/4] Building analysis database...")
            
            df_var = merge_datasets_var(processed_dict_bis, on=["country", "year_quarter"])
            
            # Data merge
            df_imapp_analysis_dataset = pd.merge(
                df_imapp_t_for_analysis,
                df_var,
                on=["country_code", "year_quarter"],
                how="left"
            )
            
            # Save data
            output_config = self.config.get("output", {})
            dataset_dir = self.config.get("paths", {}).get("dataset_output", "../data/dataset")
            output_file = output_config.get("imapp_analysis_dataset", "imapp_analysis_dataset_.csv")
            output_path = Path(dataset_dir) / output_file
            df_imapp_analysis_dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
            
            # Data quality assessment
            assess_data_quality(df_imapp_analysis_dataset)
            
            logger.info(f"Analysis database construction completed: {len(df_imapp_analysis_dataset)} rows")
            return df_imapp_analysis_dataset
            
        except Exception as e:
            logger.error(f"Analysis database construction error: {e}")
            raise
    
    def apply_country_scope(self, df_imapp_analysis_dataset: pd.DataFrame) -> pd.DataFrame:
        """Apply country scoping to filter target countries"""
        try:
            logger.info("[4/4] Applying country scoping...")
            
            # Load country configuration
            with open("../config/countries.yaml", "r") as f:
                yaml_data = yaml.safe_load(f)
            
            country_scope = yaml_data.get('countries_scope', {})
            if not country_scope:
                raise ValueError("Country configuration not found")
            
            # Create attributes DataFrame
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
            
            # Filtering and merge
            iso2_list = attr_df['iso2'].tolist()
            filtered_df = df_imapp_analysis_dataset[df_imapp_analysis_dataset['country_code'].isin(iso2_list)].copy()
            
            merged_df = filtered_df.merge(attr_df, left_on='country_code', right_on='iso2', how='left')
            
            # Save results
            output_config = self.config.get("output", {})
            dataset_dir = self.config.get("paths", {}).get("dataset_output", "../data/dataset")
            output_file = output_config.get("final_dataset", "final.csv")
            output_path = Path(dataset_dir) / output_file
            merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            
            logger.info(f"Country scoping completed: {len(merged_df)} rows, {len(iso2_list)} countries")
            logger.info(f"Target countries: {merged_df[['country_code', 'country_name', 'classification', 'region']].drop_duplicates().to_string()}")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Country scoping error: {e}")
            raise
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """Execute the complete data processing pipeline"""
        try:
            # Step 1: Data loading
            df_iMaPP_tightening, df_iMaPP_loosening = self.load_imapp_data()
            df_dict_bis = self.load_bis_data()
            
            # Step 2: Data processing
            df_imapp_t_for_analysis = self.process_imapp_data(df_iMaPP_tightening, df_iMaPP_loosening)
            processed_dict_bis = self.process_bis_data(df_dict_bis)
            
            # Step 3: Analysis database construction
            df_imapp_analysis_dataset = self.build_analysis_dataset(df_imapp_t_for_analysis, processed_dict_bis)
            
            # Step 4: Country scoping
            final_df = self.apply_country_scope(df_imapp_analysis_dataset)
            
            return final_df
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            raise