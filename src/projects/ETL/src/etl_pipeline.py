import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from abc import ABC, abstractmethod

from config import config, DataSource, ETLConfig
from validators import DataValidator, QualityReport, ValidationResult


# logging
def setup_logging(log_level: str = 'INFO') -> logging.Logger:

    logger = logging.getLogger('ETL')
    logger.setLevel(log_level)
    
    # file handler
    log_file = config.paths.LOGS_PATH / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging(config.LOG_LEVEL)


class DataTransformer(ABC):
    
    def __init__(self, config):

        self.config = config
        self.validator = DataValidator(config)
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        pass
    
    def _parse_date(self, date_str: str) -> Optional[pd.Timestamp]:

        if pd.isna(date_str):
            return pd.NaT
        
        date_str = str(date_str).strip()
        
        for fmt in self.config.DATE_FORMATS:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue
        
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except (ValueError, TypeError):
            return pd.NaT
    
    def _standardize_value(self, value: str, mapping: Dict[str, str]) -> str:

        if pd.isna(value):
            return 'unknown'
        
        value_lower = str(value).lower().strip()
        return mapping.get(value_lower, 'unknown')
    
    def _clean_phone(self, phone: str) -> Optional[str]:

        if pd.isna(phone):
            return np.nan
        
        phone_str = str(phone)
        cleaned = re.sub(r'[^0-9]', '', phone_str)
        return cleaned if cleaned else np.nan
    
    def _clean_identifier(self, value: str) -> Optional[str]:

        if pd.isna(value):
            return np.nan
        
        value_str = str(value)
        cleaned = re.sub(r'[^0-9]', '', value_str)
        return cleaned if cleaned else np.nan


class TransactionTransformer(DataTransformer):

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Transforming transactions...")
        
        df = df.copy()
        
        df.rename(columns=self.config.column_mappings.TRANSACTIONS, inplace=True)
        
        initial_count = len(df)
        df = df.drop_duplicates(keep='first')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate transactions")
        
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['amount'].fillna(df['amount'].median(), inplace=True)
        df['status'].fillna('unknown', inplace=True)
        df['payment_method'].fillna('unknown', inplace=True)
        
        df['status'] = df['status'].apply(
            lambda x: self._standardize_value(x, self.config.rules.STATUS_MAP)
        )
        df['payment_method'] = df['payment_method'].apply(
            lambda x: self._standardize_value(x, self.config.rules.PAYMENT_METHOD_MAP)
        )
        
        df['transaction_date'] = df['transaction_date'].apply(self._parse_date)
        
        invalid_dates = df['transaction_date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removed {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['transaction_date'])
        
        outlier_info = self.validator.detect_outliers_iqr(
            df['amount'].dropna(), 'amount'
        )
        if outlier_info['outlier_count'] > 0:
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in amount")
        
        logger.info(f"Transactions transformed: {len(df)} rows")
        return df


class ClientTransformer(DataTransformer):
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Transforming clients...")
        
        df = df.copy()
        
        df.rename(columns=self.config.column_mappings.CLIENTS, inplace=True)
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['client_id'], keep='first')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate clients")
        
        df['name'].fillna('Unknown', inplace=True)
        df['email'].fillna('unknown@unknown.com', inplace=True)
        
        df['phone'] = df['phone'].apply(self._clean_phone)
        df['cpf_cnpj'] = df['cpf_cnpj'].apply(self._clean_identifier)
        df['email'] = df['email'].str.lower().str.strip()
        
        df['registration_date'] = df['registration_date'].apply(self._parse_date)
        
        logger.info(f"Clients transformed: {len(df)} rows")
        return df


class VehicleTransformer(DataTransformer):
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Transforming vehicles...")
        
        df = df.copy()
        
        df.rename(columns=self.config.column_mappings.VEHICLES, inplace=True)
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['vehicle_id'], keep='first')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate vehicles")
        
        df['brand'].fillna('Unknown', inplace=True)
        df['model'].fillna('Unknown', inplace=True)
        df['category'].fillna('Unknown', inplace=True)
        
        df['cost_price'] = pd.to_numeric(df['cost_price'], errors='coerce')
        df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
        df['year_manufactured'] = pd.to_numeric(df['year_manufactured'], errors='coerce')
        df['kilometers'] = pd.to_numeric(df['kilometers'], errors='coerce')
        
        price_anomalies = (df['cost_price'] > df['sale_price']).sum()
        if price_anomalies > 0:
            logger.warning(f"Found {price_anomalies} vehicles with cost > sale price")
        
        outlier_info = self.validator.detect_outliers_iqr(
            df['sale_price'].dropna(), 'sale_price'
        )
        if outlier_info['outlier_count'] > 0:
            logger.info(f"Detected {outlier_info['outlier_count']} outliers in sale_price")
        
        logger.info(f"Vehicles transformed: {len(df)} rows")
        return df


class ETLPipeline:
    
    def __init__(self, config):

        self.config = config
        self.validator = DataValidator(config)
        self.quality_report = QualityReport()
        
        self.transformers = {
            'transactions': TransactionTransformer(config),
            'clients': ClientTransformer(config),
            'vehicles': VehicleTransformer(config)
        }
        
        self.raw_data: Dict[str, pd.DataFrame] = {}
        
        self.silver_data: Dict[str, pd.DataFrame] = {}
        
        self.gold_data: Optional[pd.DataFrame] = None
    
    def extract(self) -> bool:

        logger.info("="*60)
        logger.info("EXTRACT PHASE")
        logger.info("="*60)
        
        try:
            self.raw_data['transactions'] = pd.read_csv(
                self.config.paths.RAW_DATA_PATH / 'transactions.csv'
            )
            logger.info(f"Loaded transactions: {len(self.raw_data['transactions'])} rows")
            
            self.raw_data['clients'] = pd.read_csv(
                self.config.paths.RAW_DATA_PATH / 'clients.csv'
            )
            logger.info(f"Loaded clients: {len(self.raw_data['clients'])} rows")
            
            self.raw_data['vehicles'] = pd.read_csv(
                self.config.paths.RAW_DATA_PATH / 'vehicles.csv'
            )
            logger.info(f"Loaded vehicles: {len(self.raw_data['vehicles'])} rows")
            
            logger.info("EXTRACT phase completed successfully")
            return True
        
        except FileNotFoundError as e:
            logger.error(f"File not found during EXTRACT: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during EXTRACT: {e}")
            return False
    
    def transform(self) -> bool:

        logger.info("="*60)
        logger.info("TRANSFORM PHASE")
        logger.info("="*60)
        
        try:
            self.silver_data['transactions'] = self.transformers['transactions'].transform(
                self.raw_data['transactions']
            )
            self.silver_data['clients'] = self.transformers['clients'].transform(
                self.raw_data['clients']
            )
            self.silver_data['vehicles'] = self.transformers['vehicles'].transform(
                self.raw_data['vehicles']
            )
            
            logger.info("TRANSFORM phase completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error during TRANSFORM: {e}")
            return False
    
    def integrate(self) -> bool:

        logger.info("="*60)
        logger.info("INTEGRATION PHASE")
        logger.info("="*60)
        
        try:
            df_gold = self.silver_data['transactions'].copy()
            
            df_gold = df_gold.merge(
                self.silver_data['clients'],
                on='client_id',
                how='left',
                suffixes=('_transaction', '_client')
            )
            
            df_gold = df_gold.merge(
                self.silver_data['vehicles'],
                on='vehicle_id',
                how='left',
                suffixes=('', '_vehicle')
            )
            
            orphan_clients = df_gold['name'].isna().sum()
            orphan_vehicles = df_gold['brand'].isna().sum()
            
            if orphan_clients > 0:
                logger.warning(f"Found {orphan_clients} transactions with missing client data")
            if orphan_vehicles > 0:
                logger.warning(f"Found {orphan_vehicles} transactions with missing vehicle data")
            
            self.gold_data = df_gold
            logger.info(f"Data integrated: {len(df_gold)} rows")
            logger.info("INTEGRATION phase completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error during INTEGRATION: {e}")
            return False
    
    def load(self) -> bool:

        logger.info("="*60)
        logger.info("LOAD PHASE")
        logger.info("="*60)
        
        try:
            self.silver_data['transactions'].to_csv(
                self.config.paths.PROCESSED_DATA_PATH / 'transactions_silver.csv',
                index=False
            )
            logger.info("Saved transactions_silver.csv")
            
            self.silver_data['clients'].to_csv(
                self.config.paths.PROCESSED_DATA_PATH / 'clients_silver.csv',
                index=False
            )
            logger.info("Saved clients_silver.csv")
            
            self.silver_data['vehicles'].to_csv(
                self.config.paths.PROCESSED_DATA_PATH / 'vehicles_silver.csv',
                index=False
            )
            logger.info("Saved vehicles_silver.csv")
            
            self.gold_data.to_csv(
                self.config.paths.GOLD_DATA_PATH / 'integrated_data_gold.csv',
                index=False
            )
            logger.info("Saved integrated_data_gold.csv")
            
            logger.info("LOAD phase completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error during LOAD: {e}")
            return False
    
    def validate(self) -> None:

        logger.info("="*60)
        logger.info("VALIDATION PHASE")
        logger.info("="*60)
        
        result = self.validator.validate_primary_key(
            self.silver_data['transactions'], 'transaction_id'
        )
        self.quality_report.add_check('transactions_primary_key', result)
        
        result = self.validator.validate_primary_key(
            self.silver_data['clients'], 'client_id'
        )
        self.quality_report.add_check('clients_primary_key', result)
        
        result = self.validator.validate_primary_key(
            self.silver_data['vehicles'], 'vehicle_id'
        )
        self.quality_report.add_check('vehicles_primary_key', result)
        
        result = self.validator.validate_referential_integrity(
            self.silver_data['clients'],
            self.silver_data['transactions'],
            'client_id',
            'client_id'
        )
        self.quality_report.add_check('referential_integrity_clients', result)
        
        logger.info(self.quality_report.summary())
    
    def run(self) -> bool:

        logger.info("\n" + "="*60)
        logger.info("STARTING ETL PIPELINE")
        logger.info("="*60 + "\n")
        
        if not self.extract():
            logger.error("Pipeline failed at EXTRACT phase")
            return False
        
        if not self.transform():
            logger.error("Pipeline failed at TRANSFORM phase")
            return False
        
        if not self.integrate():
            logger.error("Pipeline failed at INTEGRATION phase")
            return False
        
        if not self.load():
            logger.error("Pipeline failed at LOAD phase")
            return False
        
        self.validate()
        
        logger.info("\n" + "="*60)
        logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
        return True


if __name__ == '__main__':
    pipeline = ETLPipeline(config)
    success = pipeline.run()
    
    if success:
        print("ETL Pipeline completed successfully!")
    else:
        print("ETL Pipeline failed. Check logs for details.")
