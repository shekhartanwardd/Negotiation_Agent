#!/usr/bin/env python3
"""
Convert ORDER_SUBTOTAL values from cents to dollars in JSON payload files.

This script processes the ORDER_SUBTOTAL field in the specified JSON files,
converting values from cents (e.g., 1889) to dollars (e.g., 18.89).
"""

import json
import argparse
from pathlib import Path


def convert_subtotal_cents_to_dollars(input_path: Path, output_path: Path = None) -> dict:
    """
    Convert ORDER_SUBTOTAL values from cents to dollars in a JSON file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save converted file. If None, overwrites input file.
        
    Returns:
        Dictionary with conversion statistics
    """
    if output_path is None:
        output_path = input_path
    
    print(f"\nProcessing: {input_path}")
    
    # Load the JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Track statistics
    stats = {
        'total_records': len(data),
        'converted': 0,
        'skipped_null': 0,
        'skipped_zero': 0,
        'sample_before': [],
        'sample_after': []
    }
    
    # Process each record
    for i, record in enumerate(data):
        if 'ORDER_SUBTOTAL' in record:
            original_value = record['ORDER_SUBTOTAL']
            
            # Store sample before conversion (first 3)
            if i < 3:
                stats['sample_before'].append(original_value)
            
            if original_value is None:
                stats['skipped_null'] += 1
                continue
            
            if original_value == 0:
                stats['skipped_zero'] += 1
                continue
            
            # Convert from cents to dollars
            # Round to 2 decimal places
            converted_value = round(original_value / 100, 2)
            record['ORDER_SUBTOTAL'] = converted_value
            stats['converted'] += 1
            
            # Store sample after conversion (first 3)
            if i < 3:
                stats['sample_after'].append(converted_value)
    
    # Save the converted data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Total records: {stats['total_records']}")
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped (null): {stats['skipped_null']}")
    print(f"  Skipped (zero): {stats['skipped_zero']}")
    print(f"  Sample values before: {stats['sample_before']}")
    print(f"  Sample values after: {stats['sample_after']}")
    print(f"  Saved to: {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert ORDER_SUBTOTAL from cents to dollars in JSON files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without making changes'
    )
    args = parser.parse_args()
    
    # Define the files to process
    base_path = Path('/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent')
    files_to_process = [
        base_path / 'dataset/escalated_payload/escalated_cases_payload.json',
        base_path / 'dataset/escalated_payload/sampled_escalated_cases_payload.json',
        base_path / 'dataset/non_escalated_payload/non_escalated_cases_payload.json',
        base_path / 'dataset/non_escalated_payload/sampled_non_escalated_cases_payload.json',
    ]
    
    print("=" * 70)
    print("ORDER_SUBTOTAL CONVERSION: CENTS → DOLLARS")
    print("=" * 70)
    
    total_stats = {
        'files_processed': 0,
        'total_converted': 0,
        'total_skipped': 0
    }
    
    for file_path in files_to_process:
        if not file_path.exists():
            print(f"\nWarning: File not found: {file_path}")
            continue
        
        if args.dry_run:
            # Just show what would be done
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"\n[DRY RUN] Would process: {file_path}")
            print(f"  Records: {len(data)}")
            if data:
                sample = data[0].get('ORDER_SUBTOTAL', 'N/A')
                print(f"  Sample ORDER_SUBTOTAL: {sample} → {sample/100 if sample else 'N/A'}")
        else:
            stats = convert_subtotal_cents_to_dollars(file_path)
            total_stats['files_processed'] += 1
            total_stats['total_converted'] += stats['converted']
            total_stats['total_skipped'] += stats['skipped_null'] + stats['skipped_zero']
    
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total records converted: {total_stats['total_converted']}")
    print(f"Total records skipped: {total_stats['total_skipped']}")


if __name__ == "__main__":
    main()

