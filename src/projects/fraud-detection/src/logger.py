import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    # logger with file and console handlers
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # file handler
    log_file = log_dir / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
