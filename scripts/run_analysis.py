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

def main():
    """Main analysis execution"""
    try:
        print("Starting macroprudential policy reaction function analysis...")
        
        # Load final dataset
        data_path = Path("../data/dataset/final.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {len(df)} rows")
        
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