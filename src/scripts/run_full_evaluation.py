"""
Full Dataset Evaluation Script

This script runs the apology credit engine on the full escalated cases dataset
and calculates the total apology credit with ceiling logic.

Ceiling logic: $0-5 -> $5, $6-10 -> $10, $11-15 -> $15, etc.
"""

import os
import sys
import json
import re
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from apology_credit_engine import (
    ApologyCreditEngine,
    load_system_prompt,
    load_config,
    setup_output_directories,
    save_results,
    generate_summary
)


def csv_to_payloads(csv_path: str) -> list:
    """
    Convert CSV dataset to list of payload dictionaries.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of payload dictionaries
    """
    df = pd.read_csv(csv_path)
    
    # Map CSV columns to expected payload format
    payloads = []
    for _, row in df.iterrows():
        payload = {
            'DELIVERY_ID': row.get('DELIVERY_ID', 0),
            'CONVERSATION_CB': row.get('CONVERSATION_CB', row.get('CONVERSATION', '')),
            'IS_CNR_ABUSER': row.get('IS_CNR_ABUSER', 0),
            'ORDER_SUBTOTAL': row.get('ORDER_SUBTOTAL', 0),
            'IS_VIP_CUSTOMER': row.get('IS_VIP_CUSTOMER', 0),
            'ISSUE_COUNT_LAST_10_ORDERS': row.get('ISSUE_COUNT_LAST_10_ORDERS', 0),
            'ISSUE_COUNT_LAST_10_DAYS': row.get('ISSUE_COUNT_LAST_10_DAYS', 0),
            'PREDICTED_ESCALATION_PROB': row.get('PREDICTED_ESCALATION_PROB', 0),
            'SH_CNR': row.get('SH_CNR', 0),
        }
        payloads.append(payload)
    
    return payloads


def apply_ceiling_logic(credit_range: str) -> float:
    """
    Apply ceiling logic to credit range prediction.
    
    $0 -> $0
    $0-5, $1-5 -> $5
    $6-10 -> $10
    $11-15 -> $15
    etc.
    
    Args:
        credit_range: Credit range string like "$6-$10" or "$0"
        
    Returns:
        Ceiling value as float
    """
    if not credit_range:
        return 0.0
    
    # Normalize the string
    credit_range = str(credit_range).replace('\u2013', '-').replace('–', '-').strip()
    
    # Handle exact $0
    if credit_range == "$0":
        return 0.0
    
    # Try to find the upper bound of the range
    # Pattern: $X-$Y or $X–$Y or $X-Y
    range_match = re.search(r'\$?(\d+)\s*[-–]\s*\$?(\d+)', credit_range)
    if range_match:
        upper_bound = int(range_match.group(2))
        return float(upper_bound)
    
    # Single value pattern: $X
    single_match = re.search(r'\$(\d+)', credit_range)
    if single_match:
        return float(single_match.group(1))
    
    # Try just a number
    num_match = re.search(r'(\d+)', credit_range)
    if num_match:
        return float(num_match.group(1))
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Run full evaluation on escalated cases dataset'
    )
    
    parser.add_argument(
        '--csv-file',
        type=str,
        default='/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/processed_dataset/dataset_final_escalated_cases.csv',
        help='Path to CSV dataset'
    )
    
    parser.add_argument(
        '--prompt-file',
        type=str,
        default='/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/system_prompt/prompt_v3.txt',
        help='Path to system prompt'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4.1-2025-04-14',
        help='Model to use'
    )
    
    parser.add_argument(
        '--run-version',
        type=int,
        default=4,
        help='Run version number'
    )
    
    parser.add_argument(
        '--escalation-threshold',
        type=float,
        default=0.2,
        help='Escalation probability threshold'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=-1,
        help='Limit number of cases (-1 for all)'
    )
    
    parser.add_argument(
        '--output-base',
        type=str,
        default='/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent',
        help='Base output path'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Save results every N cases'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FULL DATASET EVALUATION - APOLOGY CREDIT ENGINE")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.csv_file}")
    payloads = csv_to_payloads(args.csv_file)
    total_rows = len(payloads)
    print(f"Total rows in dataset: {total_rows}")
    
    # Apply limit if specified
    if args.limit > 0:
        payloads = payloads[:args.limit]
        print(f"Limited to: {len(payloads)} cases")
    
    # Load prompt
    prompt_path = Path(args.prompt_file)
    system_prompt = load_system_prompt(prompt_path)
    print(f"Loaded prompt from: {args.prompt_file}")
    
    # Setup output directories
    output_base = Path(args.output_base)
    results_dir, logs_dir = setup_output_directories(output_base, args.run_version)
    
    # Initialize engine
    engine = ApologyCreditEngine(
        system_prompt=system_prompt,
        model_name=args.model,
        run_version=args.run_version,
        escalation_threshold=args.escalation_threshold
    )
    
    print(f"\nModel: {args.model}")
    print(f"Escalation Threshold: {args.escalation_threshold}")
    print(f"Starting evaluation of {len(payloads)} cases...\n")
    
    # Run evaluation
    all_results = []
    all_failures = []
    total_credit_ceiling = 0.0
    gpt_count = 0
    skip_count = 0
    
    # Track credit distribution
    credit_distribution = {}
    
    start_time = datetime.now()
    
    for i, payload in enumerate(tqdm(payloads, desc="Evaluating")):
        try:
            escalation_prob = payload.get('PREDICTED_ESCALATION_PROB', 0) or 0
            
            if escalation_prob >= args.escalation_threshold:
                result = engine._evaluate_single(payload, temperature=0.0)
                result['inference_type'] = 'gpt'
                gpt_count += 1
            else:
                result = engine._create_low_escalation_response(payload)
                skip_count += 1
            
            # Get credit range and apply ceiling
            credit_range = result.get('recommended_credit_range', '$0')
            ceiling_value = apply_ceiling_logic(credit_range)
            total_credit_ceiling += ceiling_value
            
            # Track distribution
            credit_distribution[credit_range] = credit_distribution.get(credit_range, 0) + 1
            
            all_results.append({
                'index': i,
                'delivery_id': payload.get('DELIVERY_ID'),
                'input': payload,
                'output': result,
                'credit_range': credit_range,
                'ceiling_value': ceiling_value,
                'status': 'success'
            })
            
            # Progress update every 100 cases
            if (i + 1) % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed
                remaining = (len(payloads) - i - 1) / rate if rate > 0 else 0
                print(f"\n[{i+1}/{len(payloads)}] Running total credit (ceiling): ${total_credit_ceiling:,.2f} | "
                      f"Rate: {rate:.1f} cases/s | ETA: {remaining/60:.1f} min")
            
        except Exception as e:
            all_failures.append({
                'index': i,
                'delivery_id': payload.get('DELIVERY_ID'),
                'error': str(e),
                'status': 'failed'
            })
            print(f"\nError on case {i}: {str(e)}")
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = results_dir / f'full_evaluation_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': {
                'csv_file': args.csv_file,
                'prompt_file': args.prompt_file,
                'model': args.model,
                'escalation_threshold': args.escalation_threshold,
                'total_cases': len(payloads),
                'gpt_inference': gpt_count,
                'threshold_skip': skip_count,
                'failures': len(all_failures),
                'elapsed_time_seconds': elapsed_time,
                'timestamp': timestamp
            },
            'summary': {
                'total_credit_ceiling': total_credit_ceiling,
                'credit_distribution': credit_distribution
            },
            'results': all_results[:1000],  # Save first 1000 for inspection
            'failures': all_failures
        }, f, indent=2, default=str)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal Cases Processed: {len(payloads)}")
    print(f"  - GPT Inference: {gpt_count}")
    print(f"  - Threshold Skip ($0): {skip_count}")
    print(f"  - Failures: {len(all_failures)}")
    print(f"\nElapsed Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"\n{'='*50}")
    print(f"TOTAL APOLOGY CREDIT (Ceiling Logic): ${total_credit_ceiling:,.2f}")
    print(f"{'='*50}")
    print(f"\nCredit Distribution:")
    for credit, count in sorted(credit_distribution.items(), key=lambda x: apply_ceiling_logic(x[0])):
        ceiling = apply_ceiling_logic(credit)
        print(f"  {credit}: {count} cases -> ${ceiling:.0f} each = ${ceiling * count:,.2f}")
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

