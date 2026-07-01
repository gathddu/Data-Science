import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, recall_score, precision_score
)
import pickle
from config import get_config
from logger import setup_logger


class FraudDetectionModels:
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logger('ml_models', self.config.paths.logs)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    def run(self) -> None:

        self.logger.info("="*60)
        self.logger.info("STARTING ML MODEL TRAINING")
        self.logger.info("="*60)
        
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.save_results()
        
        self.logger.info("ML Pipeline completed")
    
    def prepare_data(self) -> None:

        self.logger.info("\n--- DATA PREPARATION ---")
        
        data_path = self.config.paths.processed_data / 'fraud_detection_processed.csv'
        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(df)} records from processed data")
        
        # separate target
        y = df['isFraud']
        X = df.drop(columns=['isFraud', 'TransactionID'])
        
        # encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'str']).columns
        self.logger.info(f"Encoding {len(categorical_cols)} categorical columns")
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # scale numerical features
        scaler = StandardScaler()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=y
        )
        
        self.logger.info(f"Train set: {len(self.X_train)} | Test set: {len(self.X_test)}")
        self.logger.info(f"Fraud rate (train): {self.y_train.mean()*100:.2f}%")
        self.logger.info(f"Fraud rate (test): {self.y_test.mean()*100:.2f}%")
    
    def train_models(self) -> None:

        self.logger.info("\n--- MODEL TRAINING ---")
        
        model_configs = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, random_state=self.config.random_state, class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.config.random_state,
                class_weight='balanced', n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.config.random_state,
                max_depth=5, learning_rate=0.1
            )
        }
        
        for name, model in model_configs.items():
            self.logger.info(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            self.logger.info(f"{name} trained")
    
    def evaluate_models(self) -> None:

        self.logger.info("\n--- MODEL EVALUATION ---")
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            metrics = {
                'accuracy': float((y_pred == self.y_test).mean()),
                'precision': float(precision_score(self.y_test, y_pred)),
                'recall': float(recall_score(self.y_test, y_pred)),
                'f1_score': float(f1_score(self.y_test, y_pred)),
                'roc_auc': float(roc_auc_score(self.y_test, y_proba))
            }
            
            self.results[name] = metrics
            
            self.logger.info(f"\n{name}:")
            self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
            self.logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    def save_results(self) -> None:

        self.logger.info("\n--- SAVING RESULTS ---")
        
        # save models
        models_dir = self.config.paths.output / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_dir / f'{name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Saved {name} to {model_path}")
        
        # save metrics
        reports_dir = self.config.paths.output / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = reports_dir / 'model_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Saved results to {results_path}")


if __name__ == '__main__':
    pipeline = FraudDetectionModels()
    pipeline.run()
