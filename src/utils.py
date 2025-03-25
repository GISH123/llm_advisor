# src/utils.py
import logging
import os
from datetime import datetime

def setup_logging(log_directory="logs"):
    """Set up logging to file and console."""
    os.makedirs(log_directory, exist_ok=True)
    log_filename = os.path.join(log_directory, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )