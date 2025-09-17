#!/usr/bin/env python3
"""
Streamlined Lending Club Analysis for Hiring Team
Focuses on $50,000 budget backtest and loan recommendations for 2017Q1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gzip
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Set random seeds for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore')

def load_and_process_data():
    """Load and process the data efficiently."""
    print("Loading data for hiring team analysis...")
    
    # Load only necessary quarters
    quarters = ['2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1']
    data_path = Path("data/archive/")
    
    raw_data = {}
    for quarter in quarters:
        file_path = data_path / f"{quarter}.csv.gz"
        if file_path.exists():
            print(f"Loading {quarter}...")
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, low_memory=False)
                raw_data[quarter] = df
                print(f"  {quarter}: {len(df):,} loans")
    
    # Process data
    processed_data = {}
    for quarter, df in raw_data.items():
        print(f"Processing {quarter}...")
        
        # Create target variable
        default_statuses = [
            'Charged Off', 'Default', 'Late (31-120 days)', 
            'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off'
        ]
        df = df.copy()
        df['default'] = df['loan_status'].isin(default_statuses).astype(int)
        
        # Clean key columns
        df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y', errors='coerce')
        
        # Clean percentage columns
        for col in ['int_rate', 'revol_util']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').replace('', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean employment length
        if 'emp_length' in df.columns:
            emp_length_map = {
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
                '10+ years': 10, 'n/a': np.nan
            }
            df['emp_length'] = df['emp_length'].map(emp_length_map)
        
        # Clean term
        if 'term' in df.columns:
            df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
        
        # FICO average
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        
        # Cap outliers
        if 'annual_inc' in df.columns:
            inc_cap = df['annual_inc'].quantile(0.99)
            df['annual_inc'] = df['annual_inc'].clip(upper=inc_cap)
        
        processed_data[quarter] = df
    
    return processed_data

def prepare_modeling_data(processed_data):
    """Prepare data for modeling with time-ordered split."""
    print("\nPreparing modeling data...")
    
    # Combine training data (2016Q1-Q3)
    train_dfs = []
    for quarter in ['2016Q1', '2016Q2', '2016Q3']:
        if quarter in processed_data:
            df = processed_data[quarter].copy()
            # Remove current loans
            df = df[df['loan_status'] != 'Current'].copy()
            train_dfs.append(df)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = processed_data['2016Q4'].copy()
    val_df = val_df[val_df['loan_status'] != 'Current'].copy()
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    print(f"Training default rate: {train_df['default'].mean():.3f}")
    print(f"Validation default rate: {val_df['default'].mean():.3f}")
    
    # Select key features
    feature_cols = [
        'funded_amnt', 'int_rate', 'term', 'installment', 'sub_grade', 'annual_inc',
        'dti', 'fico_avg', 'emp_length', 'home_ownership', 'purpose', 'verification_status',
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
        'total_acc', 'addr_state'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"Using {len(available_features)} features")
    
    # Prepare feature matrices
    X_train = train_df[available_features].copy()
    X_val = val_df[available_features].copy()
    y_train = train_df['default']
    y_val = val_df['default']
    
    # Handle categorical variables
    categorical_features = X_train.select_dtypes(include=['object']).columns
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = X_train[col].astype(str).fillna('missing')
        le.fit(X_train[col])
        
        if 'missing' not in le.classes_:
            classes = list(le.classes_)
            classes.append('missing')
            le.classes_ = np.array(classes)
        
        X_train[col] = le.transform(X_train[col])
        
        X_val[col] = X_val[col].astype(str).fillna('missing')
        X_val[col] = X_val[col].apply(lambda x: x if x in le.classes_ else 'missing')
        X_val[col] = le.transform(X_val[col])
    
    # Handle missing values and scale
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
    
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    
    return X_train, X_val, y_train, y_val, scaler, available_features

def train_model(X_train, y_train):
    """Train calibrated logistic regression model."""
    print("\nTraining model...")
    
    base_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance."""
    print("\nEvaluating model...")
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    brier_score = brier_score_loss(y_val, y_pred_proba)
    
    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    
    return auc_score, brier_score, y_pred_proba

def generate_loan_recommendations(model, scaler, available_features, processed_data, budget=50000):
    """Generate loan recommendations for 2017Q1."""
    print(f"\n=== GENERATING LOAN RECOMMENDATIONS FOR 2017Q1 (${budget:,} budget) ===")
    
    # Get 2017Q1 data
    test_data = processed_data['2017Q1'].copy()
    test_data = test_data[test_data['loan_status'] != 'Current'].copy()
    
    # Prepare features
    X_test = test_data[available_features].copy()
    
    # Handle categorical variables
    categorical_features = X_test.select_dtypes(include=['object']).columns
    for col in categorical_features:
        X_test[col] = X_test[col].fillna('missing')
        unique_vals = X_test[col].unique()
        val_map = {val: i for i, val in enumerate(unique_vals)}
        X_test[col] = X_test[col].map(val_map)
    
    # Handle missing values and scale
    imputer = SimpleImputer(strategy='median')
    X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns, index=X_test.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Get predictions
    risk_proba = model.predict_proba(X_test)[:, 1]
    
    # Create decisions DataFrame
    decisions = test_data[['id', 'funded_amnt', 'int_rate', 'term', 'installment', 
                           'sub_grade', 'annual_inc', 'dti', 'fico_avg', 'emp_length',
                           'home_ownership', 'purpose', 'verification_status', 'default']].copy()
    decisions['predicted_default_prob'] = risk_proba
    
    # Sort by lowest risk
    decisions = decisions.sort_values('predicted_default_prob')
    
    # Apply budget constraint
    cumulative_investment = decisions['funded_amnt'].cumsum()
    selected_mask = cumulative_investment <= budget
    selected_loans = decisions[selected_mask].copy()
    
    # Calculate results
    total_investment = selected_loans['funded_amnt'].sum()
    selected_default_rate = selected_loans['default'].mean()
    overall_default_rate = test_data['default'].mean()
    
    print(f"Available loans: {len(test_data):,}")
    print(f"Selected loans: {len(selected_loans):,}")
    print(f"Total investment: ${total_investment:,.2f}")
    print(f"Budget utilization: {total_investment / budget:.1%}")
    print(f"Selected default rate: {selected_default_rate:.3f}")
    print(f"Overall default rate: {overall_default_rate:.3f}")
    print(f"Improvement: {overall_default_rate - selected_default_rate:.3f}")
    
    # Add ranking
    selected_loans['rank'] = range(1, len(selected_loans) + 1)
    
    return selected_loans, total_investment, selected_default_rate, overall_default_rate

def create_calibration_plot(model, X_val, y_val, save_path='calibration_plot_hiring.png'):
    """Create calibration plot."""
    print(f"\nCreating calibration plot...")
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(12, 5))
    
    # Calibration plot
    plt.subplot(1, 2, 1)
    plt.plot(mean_pred, fraction_pos, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot - 2016Q4 Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Probability distribution
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_proba, bins=30, alpha=0.7, density=True, edgecolor='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Calibration plot saved to {save_path}")
    plt.close()  # Close plot instead of showing it

def main():
    """Main execution function."""
    print("=" * 60)
    print("HIRING TEAM ANALYSIS - $50,000 BUDGET")
    print("=" * 60)
    
    # Load and process data
    processed_data = load_and_process_data()
    
    # Prepare modeling data
    X_train, X_val, y_train, y_val, scaler, available_features = prepare_modeling_data(processed_data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    auc_score, brier_score, y_pred_proba = evaluate_model(model, X_val, y_val)
    
    # Generate loan recommendations
    recommendations, total_investment, selected_default_rate, overall_default_rate = generate_loan_recommendations(
        model, scaler, available_features, processed_data, budget=50000
    )
    
    # Create calibration plot
    create_calibration_plot(model, X_val, y_val)
    
    # Save recommendations
    recommendations.to_csv('loan_recommendations_2017Q1_50k.csv', index=False)
    print(f"\nLoan recommendations saved to 'loan_recommendations_2017Q1_50k.csv'")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS FOR HIRING TEAM")
    print("=" * 60)
    print(f"Model Performance (2016Q4 validation):")
    print(f"  ROC-AUC: {auc_score:.4f}")
    print(f"  Brier Score: {brier_score:.4f}")
    print(f"\n2017Q1 Investment Results ($50,000 budget):")
    print(f"  Selected loans: {len(recommendations):,}")
    print(f"  Total investment: ${total_investment:,.2f}")
    print(f"  Budget utilization: {total_investment / 50000:.1%}")
    print(f"  Selected default rate: {selected_default_rate:.3f}")
    print(f"  Market default rate: {overall_default_rate:.3f}")
    print(f"  Risk improvement: {overall_default_rate - selected_default_rate:.3f}")
    
    print(f"\nTop 5 Recommended Loans:")
    for i, row in recommendations.head().iterrows():
        print(f"  {row['rank']}. Loan ID {row['id']}: ${row['funded_amnt']:,.0f} at {row['int_rate']:.1f}% (Risk: {row['predicted_default_prob']:.3f})")
    
    return recommendations

if __name__ == "__main__":
    recommendations = main()
