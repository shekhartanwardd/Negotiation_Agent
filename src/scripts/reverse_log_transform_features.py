"""
Reverse Log Transform Script for Escalated Cases Payload

This script reverses the log transformations applied to features in the 
escalated_cases_payload.json file.

Based on analysis of the (Clone) 3-v2-training.ipynb and escalation_model_ml_pipeline.ipynb:

Features with log1p transformation (non-negative):
- IS_VIP_CUSTOMER: log1p transformed → binary (0 or 1)
- ORDER_SUBTOTAL: log1p transformed → dollar amount
- ISSUE_COUNT_LAST_10_ORDERS: log1p transformed → integer count
- ISSUE_COUNT_LAST_10_DAYS: log1p transformed → integer count

Features with sign-preserving log transformation (can have negative values):
- SH_CNR: sign-preserving log transformed → dollar amount

Features NOT transformed:
- IS_CNR_ABUSER: raw risk score (0-1)

Transformation applied in training:
    # For non-negative columns:
    df_transformed[col] = np.log1p(train_df[col])  # i.e., ln(1 + x)
    
    # For columns with negative values (sign-preserving):
    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col].astype(float)))
    
Reverse transformation:
    # For log1p:
    original = np.expm1(transformed)  # i.e., e^x - 1
    
    # For sign-preserving log:
    original = np.sign(transformed) * np.expm1(np.abs(transformed))

Usage:
    python reverse_log_transform_features.py [--file <path>] [--all]
    
    --file: Specific file to transform
    --all: Transform all known files

Output:
    Modifies the payload file in place
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Any, List, Dict


# Configuration - Default files to process
DEFAULT_FILES = [
    Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/train_sampled_payload_original_values.json"),
    Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/evaluation_escalated_cases_payload_original_values.json"),
]

# Features and their transformation types
LOG1P_FEATURES = [
    'IS_VIP_CUSTOMER',       # Binary after reversal (0 or 1)
    'ORDER_SUBTOTAL',        # Dollar amount
    'ISSUE_COUNT_LAST_10_ORDERS',  # Integer count
    'ISSUE_COUNT_LAST_10_DAYS',    # Integer count
]

SIGN_PRESERVING_LOG_FEATURES = [
    'SH_CNR',  # Dollar amount (can be negative in transformed form)
]

# Features that should be integers after reversal
INTEGER_FEATURES = [
    'IS_VIP_CUSTOMER',
    'ISSUE_COUNT_LAST_10_ORDERS',
    'ISSUE_COUNT_LAST_10_DAYS',
]

# Features that should be rounded to 2 decimal places (money)
MONEY_FEATURES = [
    'ORDER_SUBTOTAL',
    'SH_CNR',
]


def reverse_log1p(value: float) -> float:
    """
    Reverse the log1p transformation.
    
    log1p(x) = ln(1 + x)
    Reverse: expm1(y) = e^y - 1
    
    Args:
        value: The log-transformed value
        
    Returns:
        The original value before log transformation
    """
    if value is None:
        return 0.0
    try:
        if np.isnan(value):
        return 0.0
    return np.expm1(value)
    except (TypeError, ValueError):
        return 0.0


def reverse_sign_preserving_log(value: float) -> float:
    """
    Reverse the sign-preserving log transformation.
    
    Forward: sign(x) * log1p(|x|)
    Reverse: sign(y) * expm1(|y|)
    
    Args:
        value: The sign-preserving log-transformed value
        
    Returns:
        The original value before transformation
    """
    if value is None:
        return 0.0
    try:
        if np.isnan(value):
            return 0.0
        return np.sign(value) * np.expm1(np.abs(value))
    except (TypeError, ValueError):
        return 0.0


def is_likely_log_transformed(value: float, feature: str) -> bool:
    """
    Heuristically check if a value appears to be log-transformed.
    
    Log-transformed values typically have specific patterns:
    - IS_VIP_CUSTOMER: 0.693... (ln(2)) for VIP=1, 0 for VIP=0
    - Count features: Small decimals like 1.609... (ln(5)), etc.
    - Money features: Values that look like ln(dollar_amount)
    
    Args:
        value: The value to check
        feature: The feature name
        
    Returns:
        True if the value appears to be log-transformed
    """
    if value is None:
        return False
    
    # IS_VIP_CUSTOMER: If it's already 0 or 1 (integer), it's not transformed
    if feature == 'IS_VIP_CUSTOMER':
        return not (value == 0 or value == 1)
    
    # For other integer features, if it's a whole number >= 0, likely already reversed
    if feature in INTEGER_FEATURES:
        return not (isinstance(value, int) or (isinstance(value, float) and value == int(value) and value >= 0))
    
    # For money features, check if value looks like a log value (typically < 15 for reasonable amounts)
    # log1p(1000000) ≈ 13.8, so values > 15 are unlikely to be log-transformed
    if feature in MONEY_FEATURES:
        # If value has many decimal places and is in typical log range, likely transformed
        if isinstance(value, float) and 0 < abs(value) < 15:
            # Check if it has the characteristic decimal pattern of log values
            str_val = str(value)
            if '.' in str_val and len(str_val.split('.')[1]) > 2:
                return True
        return False
    
    return True  # Default: assume transformed


def transform_record(record: dict, skip_already_transformed: bool = True) -> dict:
    """
    Transform a single record by reversing log transformations.
    
    Args:
        record: A dictionary containing the payload data
        skip_already_transformed: If True, skip features that appear already reversed
        
    Returns:
        The record with reversed transformations
    """
    transformed = record.copy()
    
    # Process log1p features
    for feature in LOG1P_FEATURES:
        if feature in transformed and transformed[feature] is not None:
            value = transformed[feature]
            
            # Skip if already appears to be in original form
            if skip_already_transformed and not is_likely_log_transformed(value, feature):
                continue
                
            original_value = reverse_log1p(value)
            
            # Round appropriately based on feature type
            if feature in INTEGER_FEATURES:
                transformed[feature] = int(round(original_value))
            elif feature in MONEY_FEATURES:
                transformed[feature] = round(original_value, 2)
            else:
                transformed[feature] = original_value
    
    # Process sign-preserving log features
    for feature in SIGN_PRESERVING_LOG_FEATURES:
        if feature in transformed and transformed[feature] is not None:
            value = transformed[feature]
            
            # Skip if already appears to be in original form
            if skip_already_transformed and not is_likely_log_transformed(value, feature):
                continue
                
            original_value = reverse_sign_preserving_log(value)
            
            # Round appropriately
            if feature in MONEY_FEATURES:
                transformed[feature] = round(original_value, 2)
            else:
                transformed[feature] = original_value
    
    return transformed


def transform_payload(input_path: Path, in_place: bool = True) -> None:
    """
    Transform the entire payload file.
    
    Args:
        input_path: Path to the input JSON file
        in_place: If True, overwrite the input file; otherwise create a new file
    """
    print(f"\nLoading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records")
    
    # Show sample before transformation
    print("\n" + "="*70)
    print("Sample BEFORE transformation (first 3 records):")
    print("="*70)
    for i in range(min(3, len(data))):
        print(f"  Record {i+1}:")
        for feature in LOG1P_FEATURES + SIGN_PRESERVING_LOG_FEATURES:
            if feature in data[i]:
                val = data[i][feature]
                print(f"    {feature}: {val}")
    
    # Transform each record
    print("\nReversing log transformations...")
    transformed_data = [transform_record(record) for record in data]
    
    # Show sample after transformation
    print("\n" + "="*70)
    print("Sample AFTER transformation (first 3 records):")
    print("="*70)
    for i in range(min(3, len(transformed_data))):
        print(f"  Record {i+1}:")
        for feature in LOG1P_FEATURES + SIGN_PRESERVING_LOG_FEATURES:
            if feature in transformed_data[i]:
                val = transformed_data[i][feature]
                print(f"    {feature}: {val}")
    
    # Validate transformations
    print("\n" + "="*70)
    print("Validation Summary:")
    print("="*70)
    
    for feature in LOG1P_FEATURES + SIGN_PRESERVING_LOG_FEATURES:
        values = [r.get(feature) for r in transformed_data if r.get(feature) is not None]
        if values:
            print(f"  {feature}:")
            print(f"    Range: [{min(values)}, {max(values)}]")
            if feature in INTEGER_FEATURES:
                unique_vals = sorted(set(values))
                if len(unique_vals) <= 10:
                    print(f"    Unique values: {unique_vals}")
                else:
                    print(f"    Unique values count: {len(unique_vals)}")
    
    # Save transformed data
    output_path = input_path  # In-place update
    print(f"\nSaving transformed data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully saved {len(transformed_data)} records with reversed transformations")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Reverse log transformations in payload files'
    )
    parser.add_argument(
        '--file', 
        type=str, 
        help='Specific file to transform'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Transform all default files'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Transform specific file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
        transform_payload(file_path)
    elif args.all or True:  # Default behavior: transform all files
        print("="*70)
        print("REVERSE LOG TRANSFORM - Processing All Files")
        print("="*70)
        
        for file_path in DEFAULT_FILES:
            if file_path.exists():
                transform_payload(file_path)
            else:
                print(f"\nWarning: File not found, skipping: {file_path}")
        
        print("\n" + "="*70)
        print("ALL FILES PROCESSED")
        print("="*70)


if __name__ == "__main__":
    main()
