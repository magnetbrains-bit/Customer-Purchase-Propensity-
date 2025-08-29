"""
Model Evaluation and Visualization Script
========================================

This script generates comprehensive evaluation metrics and visualizations
for the Customer Propensity Analysis model.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    roc_auc_score, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_and_data():
    """Load the trained model and processed data."""
    try:
        model = joblib.load('propensity_model.pkl')
        print("✅ Model loaded successfully")
        
        # Load processed data
        visitor_df = pd.read_csv('visitor_df_final.csv')
        print(f"✅ Data loaded: {visitor_df.shape}")
        
        return model, visitor_df
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")
        return None, None

def prepare_features_and_target(visitor_df):
    """Prepare features and target for evaluation."""
    # Define feature columns - these match the final model features
    feature_columns = [
        'view_count', 'addtocart_count', 'unique_items_viewed',
        'add_to_cart_rate', 'recency_days', 'num_sessions',
        'avg_events_per_session'
    ]
    
    # Check which features exist in the data
    available_features = [col for col in feature_columns if col in visitor_df.columns]
    print(f"📊 Available features: {available_features}")
    
    # Prepare X and y
    X = visitor_df[available_features].fillna(0)
    y = visitor_df['purchased']
    
    print(f"📈 Features shape: {X.shape}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, available_features

def generate_confusion_matrix_plot(y_true, y_pred, save_path='visualizations/confusion_matrix.png'):
    """Generate and save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Purchase', 'Predicted Purchase'],
                yticklabels=['Actual No Purchase', 'Actual Purchase'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Customer Purchase Prediction', fontsize=16, pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.text(0.5, -0.15, 
             f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}',
             ha='center', transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Confusion matrix saved to {save_path}")

def generate_roc_curve_plot(y_true, y_pred_proba, save_path='visualizations/roc_curve.png'):
    """Generate and save ROC curve visualization."""
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add AUC score annotation
    plt.text(0.6, 0.2, f'AUC = {auc_score:.4f}', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📈 ROC curve saved to {save_path}")

def generate_precision_recall_plot(y_true, y_pred_proba, save_path='visualizations/precision_recall.png'):
    """Generate and save Precision-Recall curve visualization."""
    plt.figure(figsize=(10, 8))
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Plot precision-recall curve
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'Precision-Recall Curve (AP = {avg_precision:.4f})')
    
    # Add baseline (random classifier)
    baseline = len(y_true[y_true == 1]) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                label=f'Random Classifier (AP = {baseline:.4f})')
    
    # Customize plot
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add AP score annotation
    plt.text(0.6, 0.8, f'Average Precision = {avg_precision:.4f}', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Precision-Recall curve saved to {save_path}")

def generate_feature_importance_plot(model, feature_names, save_path='visualizations/feature_importance.png'):
    """Generate and save feature importance visualization."""
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            print("⚠️ Model doesn't have feature_importances_ or coef_ attribute")
            return
        
        # Create DataFrame for plotting
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(feature_imp_df)), feature_imp_df['importance'])
        
        # Customize plot
        plt.yticks(range(len(feature_imp_df)), feature_imp_df['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Feature Importance Analysis', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"🔍 Feature importance plot saved to {save_path}")
        
    except Exception as e:
        print(f"⚠️ Error generating feature importance plot: {e}")

def generate_threshold_analysis_plot(y_true, y_pred_proba, save_path='visualizations/threshold_analysis.png'):
    """Generate and save threshold analysis visualization."""
    plt.figure(figsize=(12, 8))
    
    # Calculate metrics for different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': []
    }
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot each metric
    ax1.plot(thresholds, metrics['precision'], 'o-', color='blue', linewidth=2, markersize=6)
    ax1.set_title('Precision vs Threshold', fontsize=14)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Precision')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(thresholds, metrics['recall'], 'o-', color='green', linewidth=2, markersize=6)
    ax2.set_title('Recall vs Threshold', fontsize=14)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Recall')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(thresholds, metrics['f1'], 'o-', color='red', linewidth=2, markersize=6)
    ax3.set_title('F1-Score vs Threshold', fontsize=14)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1-Score')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(thresholds, metrics['accuracy'], 'o-', color='purple', linewidth=2, markersize=6)
    ax4.set_title('Accuracy vs Threshold', fontsize=14)
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance vs Classification Threshold', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Threshold analysis saved to {save_path}")

def generate_class_distribution_plot(y_true, save_path='visualizations/class_distribution.png'):
    """Generate and save class distribution visualization."""
    plt.figure(figsize=(12, 5))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    sns.countplot(x=y_true, ax=ax1, palette=['lightcoral', 'lightgreen'])
    ax1.set_title('Class Distribution', fontsize=14)
    ax1.set_xlabel('Purchase Status (0=No, 1=Yes)')
    ax1.set_ylabel('Count')
    
    # Add count labels on bars
    for i, p in enumerate(ax1.patches):
        ax1.annotate(f'{int(p.get_height()):,}', 
                     (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=12)
    
    # Pie chart
    class_counts = y_true.value_counts()
    colors = ['lightcoral', 'lightgreen']
    ax2.pie(class_counts.values, labels=['No Purchase', 'Purchase'], 
             autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Class Distribution (%)', fontsize=14)
    
    plt.suptitle('Target Variable Distribution Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Class distribution plot saved to {save_path}")

def generate_correlation_heatmap(X, save_path='visualizations/correlation_heatmap.png'):
    """Generate and save correlation heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"🔥 Correlation heatmap saved to {save_path}")

def generate_comprehensive_report(y_true, y_pred, y_pred_proba, feature_names):
    """Generate comprehensive evaluation report."""
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*80)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Advanced metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\n📊 BASIC METRICS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\n🎯 ADVANCED METRICS:")
    print(f"   ROC AUC:           {auc_score:.4f}")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Matthews Corr:     {mcc:.4f}")
    print(f"   Cohen's Kappa:     {kappa:.4f}")
    
    print(f"\n📈 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['No Purchase', 'Purchase']))
    
    print(f"\n🔍 CONFUSION MATRIX:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"   True Negatives:  {cm[0,0]:,}")
    print(f"   False Positives: {cm[0,1]:,}")
    print(f"   False Negatives: {cm[1,0]:,}")
    print(f"   True Positives:  {cm[1,1]:,}")
    
    print(f"\n📋 FEATURE ANALYSIS:")
    print(f"   Number of features: {len(feature_names)}")
    print(f"   Features: {', '.join(feature_names)}")
    
    print("\n" + "="*80)

def main():
    """Main execution function."""
    print("🚀 Starting Customer Propensity Model Evaluation...")
    
    # Load model and data
    model, visitor_df = load_model_and_data()
    if model is None or visitor_df is None:
        return
    
    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(visitor_df)
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Generate all visualizations
    print("\n🎨 Generating visualizations...")
    
    # 1. Confusion Matrix
    generate_confusion_matrix_plot(y, y_pred, 'confusion_matrix.png')
    
    # 2. ROC Curve
    generate_roc_curve_plot(y, y_pred_proba, 'roc_curve.png')
    
    # 3. Precision-Recall Curve
    generate_precision_recall_plot(y, y_pred_proba, 'precision_recall.png')
    
    # 4. Feature Importance
    generate_feature_importance_plot(model, feature_names, 'feature_importance.png')
    
    # 5. Threshold Analysis
    generate_threshold_analysis_plot(y, y_pred_proba, 'threshold_analysis.png')
    
    # 6. Class Distribution
    generate_class_distribution_plot(y, 'class_distribution.png')
    
    # 7. Correlation Heatmap
    generate_correlation_heatmap(X, 'correlation_heatmap.png')
    
    # Generate comprehensive report
    generate_comprehensive_report(y, y_pred, y_pred_proba, feature_names)
    
    print("\n✅ Model evaluation completed successfully!")
    print("📁 All visualizations have been saved to the 'visualizations/' folder.")
    print("\n📊 Generated files:")
    print("   📂 visualizations/")
    print("   ├── confusion_matrix.png")
    print("   ├── roc_curve.png")
    print("   ├── precision_recall.png")
    print("   ├── feature_importance.png")
    print("   ├── threshold_analysis.png")
    print("   ├── class_distribution.png")
    print("   └── correlation_heatmap.png")

if __name__ == "__main__":
    main()
