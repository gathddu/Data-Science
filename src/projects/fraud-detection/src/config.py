from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class Paths:
    # data paths configuration
    project_root: Path = Path(__file__).parent.parent
    raw_data: Path = field(init=False)
    processed_data: Path = field(init=False)
    gold_data: Path = field(init=False)
    output: Path = field(init=False)
    logs: Path = field(init=False)
    
    def __post_init__(self):
        self.raw_data = self.project_root / 'data' / 'raw'
        self.processed_data = self.project_root / 'data' / 'processed'
        self.gold_data = self.project_root / 'data' / 'gold'
        self.output = self.project_root / 'output'
        self.logs = self.output / 'logs'
        
        # create directories
        for path in [self.processed_data, self.gold_data, self.logs]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class ETLConfig:
    # ETL pipeline configuration
    paths: Paths = field(default_factory=Paths)
    test_size: float = 0.2
    random_state: int = 42
    
    # feature selection
    categorical_features: List[str] = field(default_factory=lambda: [
        'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 
        'DeviceType', 'DeviceInfo', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    ])
    
    numerical_features: List[str] = field(default_factory=lambda: [
        'TransactionAmt', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14'
    ])
    
    # missing value thresholds
    missing_threshold: float = 0.5  # Drop columns with >50% missing
    
    # outlier detection
    outlier_method: str = 'iqr'  # 'iqr' or 'zscore'
    iqr_multiplier: float = 1.5

def get_config() -> ETLConfig:
    # get ETL configuration
    return ETLConfig()
