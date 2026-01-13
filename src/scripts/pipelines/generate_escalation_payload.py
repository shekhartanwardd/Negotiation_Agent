"""
Escalation Payload Generation Pipeline

This script creates JSON payloads for cases predicted to escalate to human agents.
The payloads are used to call GPT for automated negotiation/resolution.

Pipeline Steps:
1. Load datasets (predictions, original, non-log)
2. Merge datasets to create a unified dataset with all required features
3. Extract chatbot conversation portion (CONVERSATION_CB)
4. Extract apology credits from conversations (Extracted_AC)
5. Filter cases based on escalation probability threshold
6. Generate JSON payloads for escalated cases
7. Optionally generate JSON payloads for non-escalated cases

Input Dependencies:
- dataset_predictions.csv: Output of escalation_prediction_pipeline.py (log-transformed features + predictions)
- dataset_non_log.csv: Output of escalation_prediction_pipeline.py (original feature scale)
- dataset.csv: Original dataset with CONVERSATION and Parsed_AC columns

Output:
- escalated_cases_payload.json: Payloads for cases predicted to escalate
- non_escalated_cases_payload.json: Payloads for cases not predicted to escalate (optional)

The escalation threshold (default 0.5) determines which cases are predicted to escalate
to a human agent. Only these cases are sent to GPT for automated resolution.

Usage:
    python generate_escalation_payload.py --threshold 0.5
    python generate_escalation_payload.py --threshold 0.3 --include-non-escalated

Author: Negotiation Agent Team
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default paths (relative to project root)
DEFAULT_PREDICTIONS_PATH = 'dataset/processed_dataset/dataset_predictions.csv'
DEFAULT_NON_LOG_PATH = 'dataset/processed_dataset/dataset_non_log.csv'
DEFAULT_ORIGINAL_PATH = 'dataset/processed_dataset/dataset.csv'
DEFAULT_OUTPUT_DIR = 'dataset/escalated_payload'
DEFAULT_NON_ESCALATED_OUTPUT_DIR = 'dataset/non_escalated_payload'

# Feature mapping from dataset columns to payload feature names
FEATURE_MAPPING = {
    "CONVERSATION_CB": ["CONVERSATION_CB", "CONVERSATION"],
    "IS_CNR_ABUSER": ["IS_CNR_ABUSER", "ML_CX_CNR_RISK_V1_SCORE"],
    "Parsed_AC": ["Parsed_AC"],
    "Extracted_AC": ["Extracted_AC"],
    "ORDER_SUBTOTAL": ["ORDER_SUBTOTAL", "SUBTOTAL"],
    "IS_VIP_CUSTOMER": ["IS_VIP_CUSTOMER", "IS_ELITE_CX"],
    "ISSUE_COUNT_LAST_10_ORDERS": ["ISSUE_COUNT_LAST_10_ORDERS", "CREDIT_REFUND_ORDER_COUNT_L28D"],
    "ISSUE_COUNT_LAST_10_DAYS": ["ISSUE_COUNT_LAST_10_DAYS", "CREDIT_REFUND_ORDER_COUNT_L12M"],
    "PREDICTED_ESCALATION_PROB": ["PREDICTED_ESCALATION_PROB"],
    "SH_CNR": ["SH_CNR"],
    "DELIVERY_ID": ["DELIVERY_ID"]
}


# ==============================================================================
# CONVERSATION PARSING FUNCTIONS
# ==============================================================================

def parse_conversation_before_agent_transfer(conversation: str) -> str:
    """
    Parse the CONVERSATION column and filter to keep only the chatbot conversation
    portion - everything before the customer asks for a human agent.
    
    The function looks for the "System: Connecting you with an agent..." message
    and returns everything before that point.
    
    Args:
        conversation: String containing the full conversation
        
    Returns:
        str: Parsed conversation up to (but not including) the agent transfer message
             Returns the original conversation if no transfer message is found
             Returns empty string if conversation is None/NaN
    """
    if pd.isna(conversation) or conversation is None:
        return ""
    
    conversation = str(conversation)
    
    transfer_patterns = [
        r"System:\s*Connecting you with an agent\.*",
        r"System:\s*You are now connected to our support agent",
        r"System:\s*Transferring you to an agent",
    ]
    
    earliest_pos = len(conversation)
    for pattern in transfer_patterns:
        match = re.search(pattern, conversation, re.IGNORECASE)
        if match:
            earliest_pos = min(earliest_pos, match.start())
    
    if earliest_pos == len(conversation):
        return conversation.strip()
    
    parsed = conversation[:earliest_pos].strip()
    return parsed


def add_conversation_cb_column(df: pd.DataFrame, conversation_col: str = 'CONVERSATION',
                                new_col: str = 'CONVERSATION_CB', verbose: bool = True) -> pd.DataFrame:
    """
    Add a new column with parsed conversations (chatbot portion only).
    
    Args:
        df: DataFrame containing the conversation column
        conversation_col: Name of the conversation column
        new_col: Name of the new column to create
        verbose: If True, print progress information
        
    Returns:
        pd.DataFrame: DataFrame with the new column added
    """
    df = df.copy()
    
    if verbose:
        print(f"Parsing {len(df):,} conversations for chatbot portion...")
    
    df[new_col] = df[conversation_col].apply(parse_conversation_before_agent_transfer)
    
    non_empty = (df[new_col].str.len() > 0).sum()
    truncated = (df[new_col].str.len() < df[conversation_col].fillna('').str.len()).sum()
    
    if verbose:
        print(f"  Created column '{new_col}'")
        print(f"  - Non-empty conversations: {non_empty:,}")
        print(f"  - Conversations truncated (had agent transfer): {truncated:,}")
    
    return df


# ==============================================================================
# APOLOGY CREDIT EXTRACTION FUNCTIONS
# ==============================================================================

def extract_dollar_amount(text: str) -> Optional[float]:
    """Extract a dollar amount from text like '$10.00', '$5', 'A$3.00'."""
    cleaned = re.sub(r'^A?\$', '', text.strip())
    cleaned = cleaned.replace(',', '')
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned)
    
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_apology_credit_context(context: str) -> bool:
    """Check if the context indicates this is an apology/additional credit."""
    context_lower = context.lower()
    
    apology_keywords = [
        'apology', 'apologize', 'sorry',
        'additional', 'extra', 'more',
        'inconvenience', 'trouble', 'delay',
        'token of', 'gesture',
        'on top of', 'also give', 'also add', 'also process',
        'for your time', 'for the wait',
        'compensation', 'goodwill'
    ]
    
    has_apology_keyword = any(kw in context_lower for kw in apology_keywords)
    
    refund_keywords = [
        'full refund', 'refund of', 'total refund',
        'order amount', 'order total',
        'processed credits of',
    ]
    
    is_likely_refund = any(kw in context_lower for kw in refund_keywords)
    
    if 'additional' in context_lower and 'credit' in context_lower:
        return True
    
    if has_apology_keyword and not is_likely_refund:
        return True
    
    if 'with additional credits' in context_lower:
        return True
    
    return False


def find_apology_credits(conversation: str) -> List[Dict]:
    """Find all apology credit mentions in a conversation."""
    if not conversation or not isinstance(conversation, str):
        return []
    
    matches = []
    
    patterns = [
        (r'(?:A?\$)([\d,.]+)\s+additional\s+(?:DoorDash\s+)?credits?', "additional_credits"),
        (r'(?:A?\$)([\d,.]+)\s+apology\s+credits?', "apology_credits"),
        (r'(?:with\s+)?additional\s+credits?\s+(?:of\s+)?(?:A?\$)([\d,.]+)', "additional_credits_of"),
        (r'(?:A?\$)([\d,.]+)\s+(?:credits?\s+)?(?:for\s+)?(?:the\s+)?inconvenience', "for_inconvenience"),
        (r'token\s+of\s+apology[^$]{0,50}(?:A?\$)([\d,.]+)', "token_of_apology"),
        (r'as\s+(?:an?\s+)?apology[^$]{0,50}(?:A?\$)([\d,.]+)', "as_apology"),
        (r'(?:A?\$)([\d,.]+)\s+(?:more\s+)?as\s+(?:an?\s+)?apology', "dollar_as_apology"),
        (r'(?:A?\$)([\d,.]+)\s+(?:in\s+)?(?:DoorDash\s+)?credits?\s+for\s+(?:the\s+)?(?:delay|wait|trouble)', "credits_for_delay"),
    ]
    
    for pattern, pattern_type in patterns:
        for match in re.finditer(pattern, conversation, re.IGNORECASE):
            amount = extract_dollar_amount('$' + match.group(1))
            if amount:
                context = conversation[max(0, match.start()-50):min(len(conversation), match.end()+50)]
                matches.append({
                    "amount": amount,
                    "pattern": pattern_type,
                    "context": context[:100]
                })
    
    return matches


def deduplicate_matches(matches: List[Dict]) -> List[Dict]:
    """Remove duplicate matches that refer to the same credit."""
    if not matches:
        return []
    
    amount_to_matches = {}
    for match in matches:
        amount = match["amount"]
        if amount not in amount_to_matches:
            amount_to_matches[amount] = []
        amount_to_matches[amount].append(match)
    
    unique_matches = []
    for amount, amount_matches in amount_to_matches.items():
        unique_matches.append(amount_matches[0])
    
    return unique_matches


def extract_apology_credit(conversation: str) -> float:
    """Extract total apology credit amount from a single conversation string."""
    matches = find_apology_credits(conversation)
    unique_matches = deduplicate_matches(matches)
    total = sum(m["amount"] for m in unique_matches)
    return round(total, 2)


def add_extracted_ac_column(df: pd.DataFrame, conversation_col: str = 'CONVERSATION',
                            output_col: str = 'Extracted_AC', verbose: bool = True) -> pd.DataFrame:
    """
    Add extracted apology credit column to a DataFrame.
    
    Args:
        df: DataFrame containing conversation data
        conversation_col: Name of the column containing conversation text
        output_col: Name of the column to store extracted amounts
        verbose: If True, print progress information
    
    Returns:
        DataFrame with new column added
    """
    df = df.copy()
    
    if verbose:
        print(f"Extracting apology credits from {len(df):,} conversations...")
    
    amounts = []
    for idx, row in df.iterrows():
        conversation = row.get(conversation_col, '')
        if pd.isna(conversation):
            conversation = ''
        total = extract_apology_credit(str(conversation))
        amounts.append(total)
    
    df[output_col] = amounts
    
    if verbose:
        with_ac = sum(1 for a in amounts if a > 0)
        total_amount = sum(amounts)
        print(f"  Created column '{output_col}'")
        print(f"  - Rows with extracted AC > 0: {with_ac:,} ({100*with_ac/len(df):.1f}%)")
        print(f"  - Total extracted amount: ${total_amount:,.2f}")
    
    return df


# ==============================================================================
# DATA LOADING AND MERGING
# ==============================================================================

def load_data(path: str, verbose: bool = True) -> pd.DataFrame:
    """Load a CSV file and skip the first unnamed column if present."""
    if verbose:
        print(f"Loading: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path, low_memory=False)
    
    # Skip first column if it's an unnamed index column
    if df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
    
    if verbose:
        print(f"  Shape: {df.shape}")
    
    return df


def load_and_merge_datasets(predictions_path: str, non_log_path: str, original_path: str,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Load and merge the three required datasets.
    
    Args:
        predictions_path: Path to dataset_predictions.csv
        non_log_path: Path to dataset_non_log.csv
        original_path: Path to dataset.csv (original with CONVERSATION)
        verbose: If True, print progress information
        
    Returns:
        Merged DataFrame with all required features
    """
    if verbose:
        print("=" * 60)
        print("Loading Datasets")
        print("=" * 60)
    
    # Load datasets
    dataset_predictions = load_data(predictions_path, verbose=verbose)
    dataset_non_log = load_data(non_log_path, verbose=verbose)
    dataset_original = load_data(original_path, verbose=verbose)
    
    # Ensure DELIVERY_ID is float64 for consistent merging
    for df in [dataset_predictions, dataset_non_log, dataset_original]:
        if 'DELIVERY_ID' in df.columns:
            df['DELIVERY_ID'] = df['DELIVERY_ID'].astype(np.float64)
    
    if verbose:
        print("\n" + "-" * 60)
        print("Merging Datasets")
        print("-" * 60)
    
    # Filter columns from each dataset
    predictions_cols = ['DELIVERY_ID', 'PREDICTED_ESCALATION_PROB', 'PREDICTED_ESCALATION']
    available_pred_cols = [c for c in predictions_cols if c in dataset_predictions.columns]
    dataset_predictions_filtered = dataset_predictions[available_pred_cols]
    
    original_cols = ['DELIVERY_ID', 'CONVERSATION', 'Parsed_AC', 'CONVERSATION_HUMAN_AGENT']
    available_orig_cols = [c for c in original_cols if c in dataset_original.columns]
    dataset_original_filtered = dataset_original[available_orig_cols]
    
    non_log_cols = ['DELIVERY_ID', 'ML_CX_CNR_RISK_V1_SCORE', 'SUBTOTAL', 'IS_ELITE_CX',
                    'CREDIT_REFUND_ORDER_COUNT_L28D', 'CREDIT_REFUND_ORDER_COUNT_L12M', 'SH_CNR']
    available_non_log_cols = [c for c in non_log_cols if c in dataset_non_log.columns]
    dataset_non_log_filtered = dataset_non_log[available_non_log_cols]
    
    # Merge datasets
    temp1 = pd.merge(dataset_non_log_filtered, dataset_original_filtered,
                     how='inner', on='DELIVERY_ID')
    dataset = pd.merge(temp1, dataset_predictions_filtered, how='inner', on='DELIVERY_ID')
    
    if verbose:
        print(f"  Merged dataset shape: {dataset.shape}")
    
    # Rename columns for consistency
    column_rename = {
        'ML_CX_CNR_RISK_V1_SCORE': 'IS_CNR_ABUSER',
        'SUBTOTAL': 'ORDER_SUBTOTAL',
        'IS_ELITE_CX': 'IS_VIP_CUSTOMER',
        'CREDIT_REFUND_ORDER_COUNT_L28D': 'ISSUE_COUNT_LAST_10_ORDERS',
        'CREDIT_REFUND_ORDER_COUNT_L12M': 'ISSUE_COUNT_LAST_10_DAYS'
    }
    
    for old_name, new_name in column_rename.items():
        if old_name in dataset.columns:
            dataset = dataset.rename(columns={old_name: new_name})
    
    # Add CONVERSATION_HUMAN_AGENT flag if not present
    if 'CONVERSATION_HUMAN_AGENT' not in dataset.columns:
        if verbose:
            print("  Adding CONVERSATION_HUMAN_AGENT flag...")
        human_agent_cases = []
        for conversation in dataset['CONVERSATION']:
            if pd.notna(conversation) and 'Human Agent' in str(conversation):
                human_agent_cases.append(1)
            else:
                human_agent_cases.append(0)
        dataset['CONVERSATION_HUMAN_AGENT'] = human_agent_cases
    
    if verbose:
        print(f"  Final dataset columns: {len(dataset.columns)}")
    
    return dataset


# ==============================================================================
# PAYLOAD GENERATION
# ==============================================================================

def get_feature_value(row: pd.Series, feature_name: str, aliases: List[str]) -> Any:
    """Get the value for a feature, trying aliases in order."""
    for alias in aliases:
        if alias in row.index and pd.notna(row[alias]):
            return row[alias]
    return None


def create_payload(df: pd.DataFrame, verbose: bool = True) -> List[Dict]:
    """
    Create JSON payloads for cases using the defined feature schema.
    
    Args:
        df: DataFrame with cases
        verbose: If True, print progress information
        
    Returns:
        List of JSON payload dictionaries
    """
    payloads = []
    
    if verbose:
        print(f"Creating payloads for {len(df):,} cases...")
    
    for idx, row in df.iterrows():
        payload = {}
        
        for feature_name, aliases in FEATURE_MAPPING.items():
            value = get_feature_value(row, feature_name, aliases)
            
            if value is not None:
                # Convert numpy types to Python native types for JSON serialization
                if hasattr(value, 'item'):
                    value = value.item()
                # Handle special float values
                if isinstance(value, float):
                    if value != value:  # NaN check
                        value = None
                    elif value == float('inf') or value == float('-inf'):
                        value = None
            
            payload[feature_name] = value
        
        payloads.append(payload)
    
    if verbose:
        print(f"  Created {len(payloads):,} payloads")
    
    return payloads


def save_payloads(payloads: List[Dict], output_path: str, verbose: bool = True) -> None:
    """Save payloads to a JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(payloads, f, indent=2, default=str)
    
    if verbose:
        print(f"  Saved to: {output_path}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline(predictions_path: str, non_log_path: str, original_path: str,
                 output_dir: str, threshold: float = 0.5,
                 include_non_escalated: bool = False,
                 non_escalated_output_dir: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete escalation payload generation pipeline.
    
    Args:
        predictions_path: Path to dataset_predictions.csv
        non_log_path: Path to dataset_non_log.csv
        original_path: Path to dataset.csv
        output_dir: Directory to save escalated payloads
        threshold: Escalation probability threshold (default 0.5)
        include_non_escalated: If True, also generate non-escalated payloads
        non_escalated_output_dir: Directory for non-escalated payloads
        verbose: If True, print progress information
        
    Returns:
        Dictionary with statistics about the run
    """
    stats = {}
    
    if verbose:
        print("=" * 60)
        print("ESCALATION PAYLOAD GENERATION PIPELINE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Escalation threshold: {threshold}")
        print()
    
    # Step 1: Load and merge datasets
    dataset = load_and_merge_datasets(predictions_path, non_log_path, original_path, verbose=verbose)
    stats['total_records'] = len(dataset)
    
    # Step 2: Add CONVERSATION_CB column (chatbot portion only)
    if verbose:
        print("\n" + "-" * 60)
        print("Extracting Chatbot Conversation")
        print("-" * 60)
    dataset = add_conversation_cb_column(dataset, conversation_col='CONVERSATION',
                                          new_col='CONVERSATION_CB', verbose=verbose)
    
    # Step 3: Add Extracted_AC column
    if verbose:
        print("\n" + "-" * 60)
        print("Extracting Apology Credits")
        print("-" * 60)
    dataset = add_extracted_ac_column(dataset, conversation_col='CONVERSATION',
                                       output_col='Extracted_AC', verbose=verbose)
    
    # Step 4: Filter escalated cases based on threshold
    if verbose:
        print("\n" + "-" * 60)
        print("Filtering Escalated Cases")
        print("-" * 60)
        print(f"Using threshold: PREDICTED_ESCALATION_PROB >= {threshold}")
    
    escalated_df = dataset[dataset['PREDICTED_ESCALATION_PROB'] >= threshold].copy()
    stats['escalated_count'] = len(escalated_df)
    stats['escalation_rate'] = len(escalated_df) / len(dataset) * 100
    
    if verbose:
        print(f"  Escalated cases: {len(escalated_df):,} ({stats['escalation_rate']:.2f}%)")
        print(f"  Non-escalated cases: {len(dataset) - len(escalated_df):,}")
    
    # Step 5: Create payloads for escalated cases
    if verbose:
        print("\n" + "-" * 60)
        print("Generating Escalated Payloads")
        print("-" * 60)
    
    escalated_payloads = create_payload(escalated_df, verbose=verbose)
    
    # Save escalated payloads
    escalated_output_path = os.path.join(output_dir, 'escalated_cases_payload.json')
    save_payloads(escalated_payloads, escalated_output_path, verbose=verbose)
    stats['escalated_payload_path'] = escalated_output_path
    
    # Step 6: Optionally create payloads for non-escalated cases
    if include_non_escalated:
        if verbose:
            print("\n" + "-" * 60)
            print("Generating Non-Escalated Payloads")
            print("-" * 60)
        
        non_escalated_df = dataset[dataset['PREDICTED_ESCALATION_PROB'] < threshold].copy()
        stats['non_escalated_count'] = len(non_escalated_df)
        
        non_escalated_payloads = create_payload(non_escalated_df, verbose=verbose)
        
        if non_escalated_output_dir is None:
            non_escalated_output_dir = output_dir.replace('escalated', 'non_escalated')
        
        non_escalated_output_path = os.path.join(non_escalated_output_dir, 'non_escalated_cases_payload.json')
        save_payloads(non_escalated_payloads, non_escalated_output_path, verbose=verbose)
        stats['non_escalated_payload_path'] = non_escalated_output_path
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total records processed: {stats['total_records']:,}")
        print(f"Escalation threshold: {threshold}")
        print(f"Escalated cases: {stats['escalated_count']:,} ({stats['escalation_rate']:.2f}%)")
        print(f"\nOutput Files:")
        print(f"  - Escalated payloads: {stats['escalated_payload_path']}")
        if include_non_escalated:
            print(f"  - Non-escalated payloads: {stats['non_escalated_payload_path']}")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate JSON payloads for escalated cases based on prediction threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
    This script filters cases that are predicted to escalate to a human agent
    (based on PREDICTED_ESCALATION_PROB >= threshold) and generates JSON payloads
    for those cases. These payloads are then used to call GPT for automated
    negotiation and resolution.
    
    The threshold (default 0.5) determines which cases are considered likely
    to escalate. Only these cases are processed by GPT, optimizing API costs
    while focusing on high-impact cases.

Examples:
    # Run with default threshold (0.5)
    python generate_escalation_payload.py

    # Run with custom threshold (lower = more cases, higher recall)
    python generate_escalation_payload.py --threshold 0.3

    # Run with custom threshold and also generate non-escalated payloads
    python generate_escalation_payload.py --threshold 0.5 --include-non-escalated

    # Run with custom paths
    python generate_escalation_payload.py \\
        --predictions dataset/predictions.csv \\
        --non-log dataset/non_log.csv \\
        --original dataset/original.csv \\
        --output-dir results/payloads
        """
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        default=None,
        help=f'Path to dataset_predictions.csv (default: {DEFAULT_PREDICTIONS_PATH})'
    )
    
    parser.add_argument(
        '--non-log',
        type=str,
        default=None,
        help=f'Path to dataset_non_log.csv (default: {DEFAULT_NON_LOG_PATH})'
    )
    
    parser.add_argument(
        '--original',
        type=str,
        default=None,
        help=f'Path to original dataset.csv (default: {DEFAULT_ORIGINAL_PATH})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory for escalated payloads (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Escalation probability threshold (default: 0.5). Cases with PREDICTED_ESCALATION_PROB >= threshold are considered escalated.'
    )
    
    parser.add_argument(
        '--include-non-escalated',
        action='store_true',
        help='Also generate payloads for non-escalated cases'
    )
    
    parser.add_argument(
        '--non-escalated-output-dir',
        type=str,
        default=None,
        help=f'Output directory for non-escalated payloads (default: {DEFAULT_NON_ESCALATED_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Run in quiet mode (minimal output)'
    )
    
    args = parser.parse_args()
    
    # Determine paths relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent  # src/scripts/pipelines -> project root
    
    predictions_path = args.predictions or str(project_root / DEFAULT_PREDICTIONS_PATH)
    non_log_path = args.non_log or str(project_root / DEFAULT_NON_LOG_PATH)
    original_path = args.original or str(project_root / DEFAULT_ORIGINAL_PATH)
    output_dir = args.output_dir or str(project_root / DEFAULT_OUTPUT_DIR)
    non_escalated_output_dir = args.non_escalated_output_dir or str(project_root / DEFAULT_NON_ESCALATED_OUTPUT_DIR)
    
    try:
        run_pipeline(
            predictions_path=predictions_path,
            non_log_path=non_log_path,
            original_path=original_path,
            output_dir=output_dir,
            threshold=args.threshold,
            include_non_escalated=args.include_non_escalated,
            non_escalated_output_dir=non_escalated_output_dir,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

