"""
Escalation Prediction Pipeline

This script loads a dataset, processes features, and generates escalation predictions
using a pre-trained LightGBM model.

Pipeline Steps:
1. Load dataset from CSV
2. Create derived features (IS_ND, IS_MnI, IS_PFQ, IS_OSI, etc.)
3. Handle missing values
4. Convert data types for LightGBM
5. Apply log transformations to skewed features
6. Load the pre-trained model
7. Generate escalation predictions
8. Export results to CSV

Usage:
    python escalation_prediction_pipeline.py --input <input_csv> --output <output_csv>
    python escalation_prediction_pipeline.py --input <input_csv> --output <output_csv> --model <model_path>

Author: Negotiation Agent Team
"""

import argparse
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = 'model/v1_lgb_1125_fold3.pkl'

# Target column name
TARGET_COLUMN = 'IS_ESCALATED'

# Features expected by the model (from training notebook)
FINAL_FEATURES = [
    'SH_CNR', 'SH_IS_CREDITS', 'DEFECT_CATEGORY', 'SH_IS_REFUND', 'SH_IS_REDELIVERY',
    'IS_MnI', 'SH_FIRST_REPORT_ISSUE', 'SH_LATEST_REPORT_ISSUE', 'MTO_ORDER_COUNT_L90D',
    'FRAUD_CNR_REQUEST_RATIO_L60D', 'MTO_ORDER_COUNT_L12M', 'SH_IS_REJET',
    'FRAUD_CNR_APPROVED_REQUESTS_COUNT_L60D', 'AVG_VP_LIFETIME', 'IS_OSI',
    'MTO_ORDER_COUNT_L28D', 'FRAUD_CNR_AMOUNT_L60D', 'ACTUAL_VP_RAW_AMT_L12M',
    'MTO_ORDER_COUNT_LIFETIME', 'CREDIT_REFUND_ORDER_COUNT_L90D', 'IS_ND',
    'IS_20_MIN_LATE', 'MOST_FREQ_MTO_COUNT', 'FRAUD_CNR_REQUEST_RATIO_L180D',
    'NEVER_DELIVERED_COUNT_L90D', 'DEFECT_DELIVERY_COUNT_L12M',
    'HIGH_QUALITY_DELIVERY_COUNT_L12M', 'NEVER_DELIVERED_COUNT_L12M',
    'ORDER_COUNT_L12M', 'IS_PFQ', 'ACTUAL_DELIVERIES_COUNT_L12M',
    'FRAUD_CNR_AMOUNT_L180D', 'AVG_VP_LIFETIME_CATEGORY', 'ORDER_COUNT_L90D',
    'MOST_FREQ_MTO_ISSUE', 'FRAUD_CNR_REQUESTED_DELIVERIES_COUNT_L180D',
    'MTO_ORDER_COUNT_L7D', 'FRAUD_CNR_APPROVED_REQUESTS_COUNT_L180D',
    'CREDIT_REFUND_ORDER_COUNT_L12M', 'ORDER_COUNT_LIFETIME',
    'ML_CX_CNR_RISK_V1_SCORE', 'LATEST_MTO_ISSUE', 'SUBTOTAL',
    'HIGH_QUALITY_DELIVERY_COUNT_L90D', 'NEVER_DELIVERED_COUNT_L28D',
    'TOTAL_ITEM_COUNT', 'IS_TOP_95_PERCENT_VP', 'FRAUD_CNR_ND_ORDERS_COUNT_L60D',
    'NEVER_DELIVERED_COUNT_LIFETIME', 'ORDER_COUNT_L28D', 'PROMOTIONS',
    'AVG_SPEND_LIFETIME', 'DEFECT_DELIVERY_COUNT_L90D', 'DEFAULT_ZIP_CODE',
    'CREDIT_REFUND_ORDER_COUNT_L28D', 'AVG_GOV_LIFETIME',
    'CREDIT_REFUND_ORDER_COUNT_LIFETIME', 'TOTAL_MAIN_VISITOR_COUNT_L90D',
    'NEVER_DELIVERED_COUNT_L7D', 'FRAUD_CNR_ISSUANCE_AMOUNT_LIFETIME',
    'DEFECT_DELIVERY_COUNT_LIFETIME', 'HIGH_QUALITY_DELIVERY_COUNT_L28D',
    'FRAUD_CNR_GOV_AMOUNT_LIFETIME', 'SUBMIT_PLATFORM', 'AVG_SPEND_LIFETIME_CATEGORY',
    'IS_ELITE_CX', 'HOMEPAGE_SESSION_COUNT_L90D', 'EARLY_MORNING_COUNT_RATIO_LIFETIME',
    'CANCEL_COUNT_LIFETIME', 'HIGH_QUALITY_DELIVERY_COUNT_LIFETIME'
]

# Categorical features
CATEGORICAL_FEATURES = [
    'DEFECT_CATEGORY', 'SH_FIRST_REPORT_ISSUE', 'SH_LATEST_REPORT_ISSUE',
    'DEFAULT_ZIP_CODE', 'IS_TOP_95_PERCENT_VP', 'AVG_SPEND_LIFETIME_CATEGORY',
    'AVG_VP_LIFETIME_CATEGORY', 'MOST_FREQ_MTO_ISSUE', 'LATEST_MTO_ISSUE', 
    'SUBMIT_PLATFORM', 'PAYMENT_METHOD', 'FREQ_CATEGORY', 'ORDER_TIME_OF_DAY'
]

# Numeric features (derived from FINAL_FEATURES minus categorical)
NUMERIC_FEATURES = [f for f in FINAL_FEATURES if f not in CATEGORICAL_FEATURES]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load dataset from CSV file with appropriate dtype handling.
    
    Args:
        path: Path to the CSV file
        verbose: If True, print progress information
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if verbose:
        print(f"Loading dataset from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    df = pd.read_csv(path, low_memory=False)
    
    # Convert DELIVERY_ID to float64 for consistency
    if 'DELIVERY_ID' in df.columns:
        df['DELIVERY_ID'] = df['DELIVERY_ID'].astype(np.float64)
    
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
    
    return df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def create_defect_category_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Create binary indicator features from DEFECT_CATEGORY.
    
    Features created:
    - IS_ND: Never Delivered
    - IS_MnI: Missing or Incorrect Items
    - IS_PFQ: Order Quality Issue (Poor Food Quality)
    - IS_OSI: Order Status Inquiry
    - IS_LATE: Delivery Too Late / Early
    - IS_WOD: Wrong Order Received
    
    Args:
        df: Input DataFrame
        verbose: If True, print progress information
        
    Returns:
        pd.DataFrame: DataFrame with new features
    """
    df = df.copy()
    
    if 'DEFECT_CATEGORY' not in df.columns:
        if verbose:
            print("Warning: DEFECT_CATEGORY column not found. Skipping defect feature creation.")
        return df
    
    # IS_ND: Never Delivered
    df['IS_ND'] = (df['DEFECT_CATEGORY'] == 'Never Delivered').astype(int)
    
    # IS_MnI: Missing or Incorrect Items
    df['IS_MnI'] = (df['DEFECT_CATEGORY'] == 'Missing or Incorrect Items').astype(int)
    
    # IS_PFQ: Order Quality Issue (Poor Food Quality)
    df['IS_PFQ'] = (df['DEFECT_CATEGORY'] == 'Order Quality Issue').astype(int)
    
    # IS_OSI: Order Status Inquiry
    df['IS_OSI'] = (df['DEFECT_CATEGORY'] == 'Order Status Inquiry').astype(int)
    
    # IS_LATE: Delivery Too Late / Early
    df['IS_LATE'] = (df['DEFECT_CATEGORY'] == 'Delivery Too Late / Early').astype(int)
    
    # IS_WOD: Wrong Order Received
    df['IS_WOD'] = (df['DEFECT_CATEGORY'] == 'Wrong Order Received').astype(int)
    
    if verbose:
        print("Created defect category features:")
        print(f"  - IS_ND (Never Delivered): {df['IS_ND'].sum():,} cases")
        print(f"  - IS_MnI (Missing/Incorrect): {df['IS_MnI'].sum():,} cases")
        print(f"  - IS_PFQ (Quality Issue): {df['IS_PFQ'].sum():,} cases")
        print(f"  - IS_OSI (Status Inquiry): {df['IS_OSI'].sum():,} cases")
        print(f"  - IS_LATE (Late/Early): {df['IS_LATE'].sum():,} cases")
        print(f"  - IS_WOD (Wrong Order): {df['IS_WOD'].sum():,} cases")
    
    return df


def handle_missing_values(df: pd.DataFrame, features: list, 
                          categorical_features: list, verbose: bool = True) -> pd.DataFrame:
    """
    Handle missing values for the specified features.
    
    Strategy:
    - Numeric features: Fill with 0
    - Categorical features: Fill with 'Unknown'
    
    Args:
        df: Input DataFrame
        features: List of feature columns to process
        categorical_features: List of categorical feature names
        verbose: If True, print progress information
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df = df.copy()
    
    # Get available features
    available_features = [f for f in features if f in df.columns]
    
    if verbose:
        print(f"\nHandling missing values for {len(available_features)} features...")
    
    null_filled_count = 0
    for col in available_features:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_filled_count += 1
            
            if col in categorical_features:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(0)
    
    if verbose:
        if null_filled_count == 0:
            print("  No missing values found in the feature columns!")
        else:
            print(f"  Filled missing values for {null_filled_count} columns")
    
    return df


def convert_feature_dtypes(df: pd.DataFrame, categorical_features: list, 
                           numeric_features: list, verbose: bool = True) -> pd.DataFrame:
    """
    Convert features to appropriate data types for LightGBM.
    
    Args:
        df: Input DataFrame
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        verbose: If True, print progress information
        
    Returns:
        pd.DataFrame: DataFrame with converted dtypes
    """
    df = df.copy()
    
    # Convert categorical features to string
    available_cat = [f for f in categorical_features if f in df.columns]
    for col in available_cat:
        df[col] = df[col].astype(str)
    
    if verbose:
        print(f"Converted {len(available_cat)} categorical features to 'str' dtype")
    
    # Convert numeric features
    available_num = [f for f in numeric_features if f in df.columns]
    for col in available_num:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')
    
    if verbose:
        print(f"Converted {len(available_num)} numeric features to 'float64' dtype")
    
    return df


def identify_skewed_features(df: pd.DataFrame, numeric_features: list, 
                             skew_threshold: float = 1.0) -> tuple:
    """
    Identify highly skewed numeric features.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature names
        skew_threshold: Threshold for considering a feature as skewed
        
    Returns:
        tuple: (skewed_cols, negative_cols)
    """
    available_num = [f for f in numeric_features if f in df.columns]
    
    # Calculate skewness
    skew_values = df[available_num].skew()
    
    # Identify skewed columns
    skewed_cols = skew_values[skew_values > skew_threshold].index.tolist()
    
    # Identify columns with negative values (need special handling)
    negative_cols = []
    for col in skewed_cols:
        if (df[col] < 0).any():
            negative_cols.append(col)
    
    # Remove negative columns from skewed_cols for standard log1p
    skewed_cols = [col for col in skewed_cols if col not in negative_cols]
    
    return skewed_cols, negative_cols


def apply_log_transformations(df: pd.DataFrame, skewed_cols: list, 
                              negative_cols: list, verbose: bool = True) -> pd.DataFrame:
    """
    Apply log transformations to skewed features.
    
    - For non-negative skewed columns: log1p transformation
    - For columns with negative values: sign-preserving log transformation
    
    Args:
        df: Input DataFrame
        skewed_cols: Columns for standard log1p
        negative_cols: Columns needing sign-preserving transformation
        verbose: If True, print progress information
        
    Returns:
        pd.DataFrame: Transformed DataFrame
    """
    df = df.copy()
    
    if verbose:
        print("Applying log transformations...")
    
    # Standard log1p for non-negative skewed columns
    transformed_count = 0
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].astype(float))
            transformed_count += 1
    
    if verbose:
        print(f"  Applied log1p to {transformed_count} features")
    
    # Sign-preserving log for columns with negative values
    sign_transformed_count = 0
    for col in negative_cols:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col].astype(float)))
            sign_transformed_count += 1
    
    if verbose:
        print(f"  Applied sign-preserving log to {sign_transformed_count} features")
    
    return df


# ==============================================================================
# MODEL LOADING AND PREDICTION
# ==============================================================================

def load_model(model_path: str, verbose: bool = True):
    """
    Load pre-trained LightGBM model from pickle file.
    
    Args:
        model_path: Path to the pickle file
        verbose: If True, print progress information
        
    Returns:
        Loaded model object
    """
    if verbose:
        print(f"\nLoading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if verbose:
        print(f"Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
    
    return model


def generate_predictions(model, df: pd.DataFrame, features: list, 
                         categorical_features: list, verbose: bool = True) -> np.ndarray:
    """
    Generate predictions using the trained LightGBM Booster model.
    
    Args:
        model: Trained LightGBM model (Booster object)
        df: DataFrame with features
        features: List of feature names to use (in exact order expected by model)
        categorical_features: List of categorical feature names
        verbose: If True, print progress information
        
    Returns:
        np.array: Prediction probabilities
    """
    # Check which features are available
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    
    if verbose:
        print(f"\nRequired features: {len(features)}")
        print(f"Available features: {len(available)}")
    
    if missing:
        if verbose:
            print(f"\nWARNING: Missing {len(missing)} features!")
            print("Creating missing features with default values...")
        for f in missing:
            if f in categorical_features:
                df[f] = 'Unknown'
            else:
                df[f] = 0.0
    
    # Prepare feature matrix - maintain exact column order from model
    X = df[features].copy()
    
    # Encode categorical features as numeric (required for numpy array approach)
    label_encoders = {}
    
    for col in features:
        if col in categorical_features:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown').astype(str)
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(np.float64)
    
    if verbose:
        print(f"\nGenerating predictions for {len(X):,} samples...")
        print(f"Using {len(features)} features in model's expected order")
        print(f"Encoded {len(label_encoders)} categorical features as numeric")
    
    # Convert to numpy array - bypasses pandas categorical mismatch issues
    X_array = X.values.astype(np.float64)
    
    # Generate predictions
    pred_probs = model.predict(X_array)
    
    if verbose:
        print(f"\nPredictions generated successfully!")
        print(f"Prediction range: [{pred_probs.min():.4f}, {pred_probs.max():.4f}]")
        print(f"Mean prediction: {pred_probs.mean():.4f}")
    
    return pred_probs


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline(input_path: str, output_path: str, model_path: str, 
                 threshold: float = 0.5, verbose: bool = True) -> pd.DataFrame:
    """
    Run the complete escalation prediction pipeline.
    
    Args:
        input_path: Path to input CSV dataset
        output_path: Path to output CSV file
        model_path: Path to the pre-trained model pickle file
        threshold: Classification threshold for escalation prediction
        verbose: If True, print progress information
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    if verbose:
        print("=" * 60)
        print("ESCALATION PREDICTION PIPELINE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load dataset
    if verbose:
        print("\n" + "-" * 60)
        print("Step 1: Loading Dataset")
        print("-" * 60)
    df = load_dataset(input_path, verbose=verbose)
    
    # Store delivery IDs for final output
    delivery_ids = df['DELIVERY_ID'].unique().tolist() if 'DELIVERY_ID' in df.columns else None
    
    # Step 2: Create defect category features
    if verbose:
        print("\n" + "-" * 60)
        print("Step 2: Feature Engineering")
        print("-" * 60)
    df = create_defect_category_features(df, verbose=verbose)
    
    # Step 3: Handle missing values
    if verbose:
        print("\n" + "-" * 60)
        print("Step 3: Handling Missing Values")
        print("-" * 60)
    df = handle_missing_values(df, FINAL_FEATURES, CATEGORICAL_FEATURES, verbose=verbose)
    
    # Step 4: Convert data types
    if verbose:
        print("\n" + "-" * 60)
        print("Step 4: Converting Data Types")
        print("-" * 60)
    df = convert_feature_dtypes(df, CATEGORICAL_FEATURES, NUMERIC_FEATURES, verbose=verbose)
    
    # Step 5: Apply log transformations
    if verbose:
        print("\n" + "-" * 60)
        print("Step 5: Applying Log Transformations")
        print("-" * 60)
    skewed_cols, negative_cols = identify_skewed_features(df, NUMERIC_FEATURES)
    if verbose:
        print(f"Identified {len(skewed_cols)} skewed features for log1p transformation")
        print(f"Identified {len(negative_cols)} features with negative values")
    df_transformed = apply_log_transformations(df, skewed_cols, negative_cols, verbose=verbose)
    
    # Step 6: Load model
    if verbose:
        print("\n" + "-" * 60)
        print("Step 6: Loading Model")
        print("-" * 60)
    model = load_model(model_path, verbose=verbose)
    
    # Get model's expected features
    try:
        model_features = model.feature_name()
        if verbose:
            print(f"Model expects {len(model_features)} features")
    except AttributeError:
        model_features = FINAL_FEATURES
        if verbose:
            print("Using FINAL_FEATURES configuration (model doesn't expose feature names)")
    
    # Step 7: Generate predictions
    if verbose:
        print("\n" + "-" * 60)
        print("Step 7: Generating Predictions")
        print("-" * 60)
    predictions = generate_predictions(
        model, df_transformed, model_features, CATEGORICAL_FEATURES, verbose=verbose
    )
    
    # Add predictions to DataFrame
    df_transformed['PREDICTED_ESCALATION_PROB'] = predictions
    df_transformed['PREDICTED_ESCALATION'] = (predictions >= threshold).astype(int)
    
    # Restore delivery IDs
    if delivery_ids is not None:
        df_transformed['DELIVERY_ID'] = delivery_ids
    
    # Step 8: Export results
    if verbose:
        print("\n" + "-" * 60)
        print("Step 8: Exporting Results")
        print("-" * 60)
    
    # Determine output columns
    output_cols = ['DELIVERY_ID'] if 'DELIVERY_ID' in df_transformed.columns else []
    output_cols.extend([f for f in FINAL_FEATURES if f in df_transformed.columns])
    output_cols.extend(['PREDICTED_ESCALATION_PROB', 'PREDICTED_ESCALATION'])
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Export to CSV
    df_output = df_transformed[output_cols]
    df_output.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Results exported to: {output_path}")
        print(f"Output shape: {df_output.shape}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total samples processed: {len(df_transformed):,}")
        print(f"Features used: {len(model_features)}")
        print(f"\nPrediction Statistics:")
        print(f"  - Mean probability: {predictions.mean():.4f}")
        print(f"  - Median probability: {np.median(predictions):.4f}")
        print(f"  - Min: {predictions.min():.4f}")
        print(f"  - Max: {predictions.max():.4f}")
        print(f"\nPredicted Escalations (threshold={threshold}):")
        print(f"  - Escalated: {(df_transformed['PREDICTED_ESCALATION'] == 1).sum():,}")
        print(f"  - Not Escalated: {(df_transformed['PREDICTED_ESCALATION'] == 0).sum():,}")
        print(f"  - Predicted escalation rate: {df_transformed['PREDICTED_ESCALATION'].mean():.2%}")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df_output


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Escalation Prediction Pipeline - Generate escalation predictions from dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default model path
    python escalation_prediction_pipeline.py --input data.csv --output predictions.csv

    # Run with custom model path
    python escalation_prediction_pipeline.py --input data.csv --output predictions.csv --model custom_model.pkl

    # Run with custom threshold
    python escalation_prediction_pipeline.py --input data.csv --output predictions.csv --threshold 0.3

    # Run in quiet mode
    python escalation_prediction_pipeline.py --input data.csv --output predictions.csv --quiet
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV dataset'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output CSV file for predictions'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help=f'Path to pre-trained model pickle file (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Classification threshold for escalation prediction (default: 0.5)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Run in quiet mode (minimal output)'
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Try to find model relative to script location
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent  # src/scripts/pipelines -> project root
        model_path = project_root / DEFAULT_MODEL_PATH
        
        if not model_path.exists():
            print(f"Error: Default model not found at {model_path}")
            print("Please specify the model path using --model argument")
            sys.exit(1)
        
        model_path = str(model_path)
    
    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            model_path=model_path,
            threshold=args.threshold,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

