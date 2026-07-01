import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from config import get_config
from logger import setup_logger


class FraudVisualizations:
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logger('visualizations', self.config.paths.logs)
        self.viz_dir = self.config.paths.output / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.results = None
    
    def run(self) -> None:

        self.logger.info("="*60)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("="*60)
        
        self.load_data()
        self.plot_fraud_distribution()
        self.plot_amount_analysis()
        self.plot_temporal_patterns()
        self.plot_feature_importance()
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        
        self.logger.info("All visualizations generated")
    
    def load_data(self) -> None:

        data_path = self.config.paths.processed_data / 'fraud_detection_processed.csv'
        self.df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(self.df)} records")
        
        results_path = self.config.paths.output / 'reports' / 'model_results.json'
        with open(results_path, 'r') as f:
            self.results = json.load(f)
    
    def plot_fraud_distribution(self) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # count plot
        fraud_counts = self.df['isFraud'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(['Legitimate', 'Fraud'], fraud_counts.values, color=colors)
        axes[0].set_title('Transaction Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(fraud_counts.values):
            axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')
        
        # percentage pie
        axes[1].pie(fraud_counts.values, labels=['Legitimate', 'Fraud'],
                   autopct='%1.2f%%', colors=colors, startangle=90,
                   explode=(0, 0.1))
        axes[1].set_title('Fraud Rate', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '01_fraud_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("Fraud distribution plot saved")
    
    def plot_amount_analysis(self) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # box plot
        fraud_data = self.df[self.df['isFraud'] == 1]['TransactionAmt']
        legit_data = self.df[self.df['isFraud'] == 0]['TransactionAmt']
        
        axes[0].boxplot([legit_data.clip(upper=1000), fraud_data.clip(upper=1000)],
                       tick_labels=['Legitimate', 'Fraud'])
        axes[0].set_title('Transaction Amount Distribution (capped at $1000)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amount ($)')
        
        # histogram overlay
        axes[1].hist(legit_data.clip(upper=500), bins=50, alpha=0.7, label='Legitimate', color='#2ecc71', density=True)
        axes[1].hist(fraud_data.clip(upper=500), bins=50, alpha=0.7, label='Fraud', color='#e74c3c', density=True)
        axes[1].set_title('Amount Distribution Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Amount ($)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '02_amount_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("Amount analysis plot saved")
    
    def plot_temporal_patterns(self) -> None:

        fig, ax = plt.subplots(figsize=(14, 5))
        
        # group by time bins
        self.df['time_bin'] = pd.cut(self.df['TransactionDT'], bins=50)
        time_fraud = self.df.groupby('time_bin', observed=True)['isFraud'].mean() * 100
        
        ax.plot(range(len(time_fraud)), time_fraud.values, color='#e74c3c', linewidth=2)
        ax.fill_between(range(len(time_fraud)), time_fraud.values, alpha=0.3, color='#e74c3c')
        ax.set_title('Fraud Rate Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Fraud Rate (%)')
        ax.axhline(y=self.df['isFraud'].mean()*100, color='gray', linestyle='--', label=f'Average: {self.df["isFraud"].mean()*100:.2f}%')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '03_temporal_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("Temporal patterns plot saved")
    
    def plot_feature_importance(self) -> None:

        import pickle
        
        model_path = self.config.paths.output / 'models' / 'random_forest.pkl'
        with open(model_path, 'rb') as f:
            rf_model = pickle.load(f)
        
        # get feature names (exclude target and ID)
        feature_cols = [c for c in self.df.columns if c not in ['isFraud', 'TransactionID', 'time_bin']]
        importances = rf_model.feature_importances_
        
        indices = np.argsort(importances)[-20:]
        top_features = [feature_cols[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(20), top_importances, color='#3498db')
        ax.set_yticks(range(20))
        ax.set_yticklabels(top_features)
        ax.set_title('Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '04_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("Feature importance plot saved")
    
    def plot_model_comparison(self) -> None:
    
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        x = np.arange(len(models))
        width = 0.15
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            values = [self.results[m][metric] for m in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models])
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '05_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("Model comparison plot saved")
    
    def plot_confusion_matrices(self) -> None:

        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        import pickle
        
        # prepare data same way as training
        y = self.df['isFraud']
        X = self.df.drop(columns=['isFraud', 'TransactionID', 'time_bin'], errors='ignore')
        
        categorical_cols = X.select_dtypes(include=['object', 'str']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        scaler = StandardScaler()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        model_names = ['logistic_regression', 'random_forest', 'gradient_boosting']
        
        for idx, name in enumerate(model_names):
            model_path = self.config.paths.output / 'models' / f'{name}.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / '06_confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("Confusion matrices plot saved")


if __name__ == '__main__':
    viz = FraudVisualizations()
    viz.run()
