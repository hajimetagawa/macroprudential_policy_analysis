"""
Dataset Preprocessor
Handles dataset loading, variable selection, renaming, and preprocessing for analysis
"""

import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """
    Preprocesses analysis dataset based on YAML configuration
    """
    
    def __init__(self, config_path: str = "../config/analysis_variables.yaml"):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path: Path to analysis variables configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load analysis variables configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Analysis variables configuration loaded: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def load_and_preprocess_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset and apply preprocessing steps
        
        Args:
            dataset_path: Path to the analysis dataset
            
        Returns:
            Preprocessed DataFrame ready for analysis
        """
        # Load dataset
        df = self._load_dataset(dataset_path)
        
        # Apply variable mapping and selection
        df_processed = self._apply_variable_mapping(df)
        df_selected = self._select_analysis_variables(df_processed)
        
        # Apply data filters
        df_filtered = self._apply_data_filters(df_selected)
        
        # Validate dataset
        self._validate_dataset(df_filtered)
        
        logger.info(f"Dataset preprocessing completed: {len(df_filtered)} rows, {len(df_filtered.columns)} columns")
        return df_filtered
    
    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load the analysis dataset"""
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns from {dataset_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _apply_variable_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply variable mapping to rename columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with renamed columns
        """
        variable_mapping = self.config.get('variable_mapping', {})
        
        # Find existing columns that match the mapping
        columns_to_rename = {}
        for original_col, new_col in variable_mapping.items():
            if original_col in df.columns:
                columns_to_rename[original_col] = new_col
        
        # Apply renaming
        df_renamed = df.rename(columns=columns_to_rename)
        
        logger.info(f"Variable mapping applied: {len(columns_to_rename)} columns renamed")
        if columns_to_rename:
            logger.debug(f"Renamed columns: {columns_to_rename}")
        
        return df_renamed
    
    def _select_analysis_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the variables needed for analysis
        
        Args:
            df: DataFrame with mapped column names
            
        Returns:
            DataFrame with selected columns only
        """
        analysis_vars = self.config.get('analysis_variables', {})
        
        # Collect all variables to keep
        vars_to_keep = []
        vars_to_keep.extend(analysis_vars.get('dependent_vars', []))
        vars_to_keep.extend(analysis_vars.get('independent_vars', []))
        vars_to_keep.extend(analysis_vars.get('control_vars', []))
        
        # Filter to existing columns
        existing_vars = [var for var in vars_to_keep if var in df.columns]
        missing_vars = [var for var in vars_to_keep if var not in df.columns]
        
        if missing_vars:
            logger.warning(f"Missing variables in dataset: {missing_vars}")
        
        # Select columns
        df_selected = df[existing_vars].copy()
        
        logger.info(f"Variable selection completed: {len(existing_vars)} variables selected")
        return df_selected
    
    def _apply_data_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality filters
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        original_rows = len(df_filtered)
        
        data_filters = self.config.get('data_filters', {})
        
        # Apply date range filter if period column exists
        if 'period' in df_filtered.columns:
            start_period = data_filters.get('start_period')
            end_period = data_filters.get('end_period')
            
            if start_period:
                df_filtered = df_filtered[df_filtered['period'] >= start_period]
            if end_period:
                df_filtered = df_filtered[df_filtered['period'] <= end_period]
        
        # Apply minimum observations per country filter
        if 'country_code' in df_filtered.columns:
            min_obs = data_filters.get('min_observations_per_country', 0)
            if min_obs > 0:
                country_counts = df_filtered['country_code'].value_counts()
                valid_countries = country_counts[country_counts >= min_obs].index
                df_filtered = df_filtered[df_filtered['country_code'].isin(valid_countries)]
        
        # Apply missing data threshold filter
        max_missing_ratio = data_filters.get('max_missing_ratio', 1.0)
        if max_missing_ratio < 1.0:
            # Calculate missing ratio for each row
            missing_ratios = df_filtered.isnull().sum(axis=1) / len(df_filtered.columns)
            df_filtered = df_filtered[missing_ratios <= max_missing_ratio]
        
        rows_removed = original_rows - len(df_filtered)
        logger.info(f"Data filters applied: {rows_removed} rows removed, {len(df_filtered)} rows remaining")
        
        return df_filtered
    
    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate the preprocessed dataset
        
        Args:
            df: Preprocessed DataFrame
        """
        if df.empty:
            raise ValueError("Dataset is empty after preprocessing")
        
        # Check for required variable categories
        analysis_vars = self.config.get('analysis_variables', {})
        
        dependent_vars = analysis_vars.get('dependent_vars', [])
        existing_dependent = [var for var in dependent_vars if var in df.columns]
        if not existing_dependent:
            logger.warning("No dependent variables found in dataset")
        
        independent_vars = analysis_vars.get('independent_vars', [])
        existing_independent = [var for var in independent_vars if var in df.columns]
        if len(existing_independent) < 2:
            logger.warning("Fewer than 2 independent variables found in dataset")
        
        logger.info("Dataset validation completed successfully")
    
    def get_variable_info(self) -> Dict:
        """
        Get information about configured variables
        
        Returns:
            Dictionary with variable information
        """
        analysis_vars = self.config.get('analysis_variables', {})
        return {
            'dependent_variables': analysis_vars.get('dependent_vars', []),
            'independent_variables': analysis_vars.get('independent_vars', []),
            'control_variables': analysis_vars.get('control_vars', []),
            'variable_mapping': self.config.get('variable_mapping', {}),
            'data_filters': self.config.get('data_filters', {})
        }
    
    def print_dataset_summary(self, df: pd.DataFrame) -> None:
        """
        Print summary of the preprocessed dataset
        
        Args:
            df: Preprocessed DataFrame
        """
        print("\n=== Dataset Summary ===")
        print(f"Shape: {df.shape}")
        print(f"Period range: {df['period'].min()} - {df['period'].max()}" if 'period' in df.columns else "Period: N/A")
        print(f"Countries: {df['country_code'].nunique()}" if 'country_code' in df.columns else "Countries: N/A")
        
        # Missing data summary
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            print("\nMissing data:")
            for col, missing_count in missing_summary[missing_summary > 0].items():
                missing_pct = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
        else:
            print("\nNo missing data")
        
        print("======================\n")