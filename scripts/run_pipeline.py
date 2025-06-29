#!/usr/bin/env python3
"""
Macroprudential Policy Analysis Pipeline
Main orchestration script for data processing pipeline
"""
import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline_orchestrator import PipelineOrchestrator
from src.utils.yaml_loader import load_yaml

def setup_logging(config: dict) -> None:
    """Initialize logging configuration"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    log_dir = config.get("paths", {}).get("logs", "../logs")
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    """Main pipeline execution"""
    try:
        # Load configuration
        config = load_yaml("config")
        
        # Setup logging
        setup_logging(config)
        
        logger = logging.getLogger(__name__)
        logger.info("=== Macroprudential Policy Analysis Pipeline Started ===")
        
        # Initialize and run pipeline
        orchestrator = PipelineOrchestrator(config)
        final_dataset = orchestrator.run_full_pipeline()
        
        logger.info("=== Pipeline Completed ===")
        logger.info(f"Final dataset: {len(final_dataset)} rows")
        
        return final_dataset
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Pipeline execution error: {e}")
        raise

if __name__ == "__main__":
    main()