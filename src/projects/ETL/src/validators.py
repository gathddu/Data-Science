from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:

    is_valid: bool
    issues: List[str]
    warnings: List[str]
    stats: Dict


class DataValidator:
    
    def __init__(self, config):

        self.config = config
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> ValidationResult:

        issues = []
        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)
        
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        warnings = []
        if extra_columns:
            warnings.append(f"Extra columns (will be kept): {extra_columns}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            stats={'total_columns': len(df.columns), 'expected_columns': len(expected_columns)}
        )
    
    def validate_no_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> ValidationResult:

        duplicates = df.duplicated(subset=subset, keep=False).sum()
        
        issues = []
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=[],
            stats={'duplicates_found': duplicates}
        )
    
    def validate_primary_key(self, df: pd.DataFrame, key_column: str) -> ValidationResult:

        if key_column not in df.columns:
            return ValidationResult(
                is_valid=False,
                issues=[f"Key column '{key_column}' not found"],
                warnings=[],
                stats={}
            )
        
        duplicates = df[key_column].duplicated().sum()
        nulls = df[key_column].isna().sum()
        
        issues = []
        if duplicates > 0:
            issues.append(f"Primary key '{key_column}' has {duplicates} duplicates")
        if nulls > 0:
            issues.append(f"Primary key '{key_column}' has {nulls} null values")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=[],
            stats={'duplicates': duplicates, 'nulls': nulls}
        )
    
    def validate_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> ValidationResult:

        missing_pct = df.isnull().sum() / len(df)
        problematic = missing_pct[missing_pct > threshold]
        
        issues = []
        warnings = []
        
        if len(problematic) > 0:
            for col, pct in problematic.items():
                msg = f"Column '{col}' has {pct*100:.1f}% missing values"
                if pct > 0.8:
                    issues.append(msg)
                else:
                    warnings.append(msg)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            stats={'columns_with_missing': len(missing_pct[missing_pct > 0])}
        )
    
    def validate_numeric_range(self, df: pd.DataFrame, column: str, 
                              min_val: Optional[float] = None, 
                              max_val: Optional[float] = None) -> ValidationResult:

        if column not in df.columns:
            return ValidationResult(
                is_valid=False,
                issues=[f"Column '{column}' not found"],
                warnings=[],
                stats={}
            )
        
        issues = []
        out_of_range = 0
        
        if min_val is not None:
            out_of_range += (df[column] < min_val).sum()
        if max_val is not None:
            out_of_range += (df[column] > max_val).sum()
        
        if out_of_range > 0:
            issues.append(f"Column '{column}' has {out_of_range} values outside range [{min_val}, {max_val}]")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=[],
            stats={'out_of_range': out_of_range}
        )
    
    def validate_referential_integrity(self, parent_df: pd.DataFrame, child_df: pd.DataFrame,
                                      parent_key: str, child_key: str) -> ValidationResult:

        orphan_records = (~child_df[child_key].isin(parent_df[parent_key])).sum()
        
        warnings = []
        if orphan_records > 0:
            warnings.append(f"Found {orphan_records} orphan records in child table")
        
        return ValidationResult(
            is_valid=True,  # orphans are warnings, not errors
            issues=[],
            warnings=warnings,
            stats={'orphan_records': orphan_records}
        )
    
    def detect_outliers_iqr(self, series: pd.Series, column_name: str) -> Dict:

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        multiplier = self.config.IQR_MULTIPLIER
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers.sum()
        
        return {
            'column': column_name,
            'outlier_count': outlier_count,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': outliers[outliers].index.tolist()
        }
    
    def detect_anomalies(self, df: pd.DataFrame, cost_col: str, sale_col: str) -> Dict:

        if cost_col not in df.columns or sale_col not in df.columns:
            return {'error': f'Columns {cost_col} or {sale_col} not found'}
        
        anomalies = df[cost_col] > df[sale_col]
        anomaly_count = anomalies.sum()
        
        return {
            'anomaly_count': anomaly_count,
            'anomaly_indices': anomalies[anomalies].index.tolist(),
            'anomaly_percentage': (anomaly_count / len(df)) * 100
        }


class QualityReport:
    
    def __init__(self):

        self.checks: Dict[str, ValidationResult] = {}
        self.outliers: Dict[str, List[Dict]] = {}
        self.anomalies: Dict[str, Dict] = {}
    
    def add_check(self, check_name: str, result: ValidationResult) -> None:

        self.checks[check_name] = result
    
    def add_outliers(self, dataset_name: str, outlier_list: List[Dict]) -> None:

        self.outliers[dataset_name] = outlier_list
    
    def add_anomalies(self, dataset_name: str, anomaly_dict: Dict) -> None:

        self.anomalies[dataset_name] = anomaly_dict
    
    def to_dict(self) -> Dict:

        return {
            'validation_checks': {
                name: {
                    'is_valid': result.is_valid,
                    'issues': result.issues,
                    'warnings': result.warnings,
                    'stats': result.stats
                }
                for name, result in self.checks.items()
            },
            'outliers': self.outliers,
            'anomalies': self.anomalies
        }
    
    def summary(self) -> str:

        total_checks = len(self.checks)
        passed_checks = sum(1 for r in self.checks.values() if r.is_valid)
        
        summary = f"\n{'='*60}\nQUALITY REPORT SUMMARY\n{'='*60}\n"
        summary += f"Validation Checks: {passed_checks}/{total_checks} passed\n"
        
        issues_count = sum(len(r.issues) for r in self.checks.values())
        warnings_count = sum(len(r.warnings) for r in self.checks.values())
        
        if issues_count > 0:
            summary += f"Issues: {issues_count}\n"
        if warnings_count > 0:
            summary += f"Warnings: {warnings_count}\n"
        
        return summary
