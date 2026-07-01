import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import json
from config import get_config
from logger import setup_logger

class FraudDetectionETL:
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logger(__name__, self.config.paths.logs)
        self.train_tx = None
        self.train_id = None
        self.merged_data = None
        self.quality_report = {}

    def _convert_to_serializable(self, obj):
    
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    
    def run(self) -> None:

        self.logger.info("="*60)
        self.logger.info("STARTING FRAUD DETECTION ETL PIPELINE")
        self.logger.info("="*60)
        
        try:
            self.extract()
            self.transform()
            self.load()
            self.logger.info("ETL Pipeline completed.")
        except Exception as e:
            self.logger.error(f"ETL Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def extract(self) -> None:

        self.logger.info("\n--- EXTRACTION PHASE ---")
        
        try:
            train_tx_path = self.config.paths.raw_data / 'train_transaction.csv'
            train_id_path = self.config.paths.raw_data / 'train_identity.csv'
            
            self.logger.info(f"Loading transactions from {train_tx_path}")
            self.train_tx = pd.read_csv(train_tx_path)
            self.logger.info(f"Loaded {len(self.train_tx)} transactions with {len(self.train_tx.columns)} columns")
            
            self.logger.info(f"Loading identity data from {train_id_path}")
            self.train_id = pd.read_csv(train_id_path)
            self.logger.info(f"Loaded {len(self.train_id)} identity records with {len(self.train_id.columns)} columns")
            
            self.quality_report['extraction'] = {
                'transactions_count': len(self.train_tx),
                'identity_count': len(self.train_id),
                'fraud_count': int(self.train_tx['isFraud'].sum()),
                'fraud_rate': float(self.train_tx['isFraud'].mean())
            }
            
        except FileNotFoundError as e:
            self.logger.error(f"Data file not found: {e}")
            raise
    
    def transform(self) -> None:

        self.logger.info("\n--- TRANSFORMATION PHASE ---")
        
        # merge data
        self.logger.info("Merging transaction and identity data")
        self.merged_data = self.train_tx.merge(self.train_id, on='TransactionID', how='left')
        self.logger.info(f"Merged data shape: {self.merged_data.shape}")
        
        # handle missing values
        self.logger.info("Handling missing values")
        initial_missing = self.merged_data.isnull().sum().sum()
        
        # drop columns with >50% missing
        missing_pct = self.merged_data.isnull().sum() / len(self.merged_data)
        cols_to_drop = missing_pct[missing_pct > self.config.missing_threshold].index.tolist()
        self.merged_data = self.merged_data.drop(columns=cols_to_drop)
        self.logger.info(f"Dropped {len(cols_to_drop)} columns with >{self.config.missing_threshold*100}% missing")
        
        # fill remaining missing values
        fill_values = {}
        for col in self.merged_data.columns:
            if self.merged_data[col].isnull().sum() > 0:
                if self.merged_data[col].dtype in ['float64', 'int64']:
                    fill_values[col] = self.merged_data[col].median()
                else:
                    fill_values[col] = 'unknown'

        self.merged_data = self.merged_data.fillna(fill_values)
        
        self.logger.info(f"Filled missing values")
        
        # detect outliers
        self.logger.info("Detecting outliers")
        outlier_count = self._detect_outliers()
        self.logger.info(f"Detected {outlier_count} outliers")
        
        self.quality_report['transformation'] = {
            'initial_missing_values': int(initial_missing),
            'columns_dropped': len(cols_to_drop),
            'outliers_detected': outlier_count,
            'final_shape': list(self.merged_data.shape)
        }
    
    def _detect_outliers(self) -> int:

        numeric_cols = self.merged_data.select_dtypes(include=[np.number]).columns
        outlier_count = 0
    
        for col in numeric_cols:
            Q1 = self.merged_data[col].quantile(0.25)
            Q3 = self.merged_data[col].quantile(0.75)
            IQR = Q3 - Q1
        
            if IQR == 0:
                continue
            
        lower_bound = Q1 - self.config.iqr_multiplier * IQR
        upper_bound = Q3 + self.config.iqr_multiplier * IQR
        outliers = ((self.merged_data[col] < lower_bound) | (self.merged_data[col] > upper_bound)).sum()
        outlier_count += outliers
    
        return outlier_count

    
    def load(self) -> None:

        self.logger.info("\n--- LOADING PHASE ---")
        
        # processed data
        processed_path = self.config.paths.processed_data / 'fraud_detection_processed.csv'
        self.merged_data.to_csv(processed_path, index=False)
        self.logger.info(f"Saved processed data to {processed_path}")
        
        # quality report
        report_path = self.config.paths.output / 'quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(self._convert_to_serializable(self.quality_report), f, indent=2)
        self.logger.info(f"Saved quality report to {report_path}")
        
        self.quality_report['loading'] = {
            'processed_file': str(processed_path),
            'report_file': str(report_path)
        }

if __name__ == '__main__':
    pipeline = FraudDetectionETL()
    pipeline.run()
