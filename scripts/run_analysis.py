#!/usr/bin/env python3
"""
Macroprudential Policy Analysis Script
Main script for running policy reaction function analysis
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis_engine import AnalysisEngine
from src.data_processing.dataset_preprocessor import DatasetPreprocessor

def main():
    """Main analysis execution"""
    try:
        print("Starting macroprudential policy reaction function analysis...")
        
        # Load and preprocess final dataset
        data_path = Path("../data/dataset/macroprudential_analysis_dataset.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        # Initialize preprocessor and load dataset
        preprocessor = DatasetPreprocessor()
        df = preprocessor.load_and_preprocess_dataset(str(data_path))
        
        # Print dataset summary
        preprocessor.print_dataset_summary(df)
        
        # Print variable information
        var_info = preprocessor.get_variable_info()
        print("=== Analysis Variables ===")
        print(f"Dependent variables: {var_info['dependent_variables']}")
        print(f"Independent variables: {var_info['independent_variables']}")
        print(f"Control variables: {var_info['control_variables']}")
        print("==========================\n")
        
        # Initialize and run analysis
        analyzer = AnalysisEngine()
        results = analyzer.run_comprehensive_analysis(df)
        
        # Generate visualizations
        analyzer.create_visualizations(results, df)
        
        # Output results summary
        analyzer.print_analysis_summary(results)
        
        print("Analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Analysis execution error: {e}")
        raise

if __name__ == "__main__":
    results = main()