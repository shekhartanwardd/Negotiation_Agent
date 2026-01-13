"""
Full Dataset Evaluation Script (Async Version)

This script runs the apology credit engine on the full escalated cases dataset
using asynchronous API calls for significantly faster processing.

Ceiling logic: $0-5 -> $5, $6-10 -> $10, $11-15 -> $15, etc.
"""

import os
import sys
import json
import re
import math
import asyncio
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_system_prompt(prompt_path: Path) -> str:
    """Load system prompt from file."""
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def transform_payload_for_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Transform the payload to appropriate types expected by the model."""
    transformed = {}
    
    # DELIVERY_ID - convert to string
    delivery_id = payload.get('DELIVERY_ID', 0)
    if isinstance(delivery_id, float):
        transformed['DELIVERY_ID'] = str(int(delivery_id))
    else:
        transformed['DELIVERY_ID'] = str(delivery_id)
    
    # CONVERSATION_CB - keep as-is (string)
    transformed['CONVERSATION_CB'] = payload.get('CONVERSATION_CB', '')
    
    # IS_CNR_ABUSER - threshold at 0.5 for binary
    cnr_score = payload.get('IS_CNR_ABUSER', 0)
    if cnr_score is None:
        cnr_score = 0
    transformed['IS_CNR_ABUSER'] = cnr_score > 0.5
    
    # ORDER_SUBTOTAL - already in dollars
    order_subtotal = payload.get('ORDER_SUBTOTAL', 0)
    if order_subtotal is None:
        order_subtotal = 0
    transformed['ORDER_SUBTOTAL'] = round(float(order_subtotal), 2)
    
    # IS_VIP_CUSTOMER - binary
    vip_value = payload.get('IS_VIP_CUSTOMER', 0)
    if vip_value is None:
        vip_value = 0
    transformed['IS_VIP_CUSTOMER'] = bool(int(vip_value))
    
    # ISSUE_COUNT_LAST_10_ORDERS
    issue_10_orders = payload.get('ISSUE_COUNT_LAST_10_ORDERS', 0)
    if issue_10_orders is None:
        issue_10_orders = 0
    transformed['ISSUE_COUNT_LAST_10_ORDERS'] = int(round(float(issue_10_orders)))
    
    # ISSUE_COUNT_LAST_10_DAYS
    issue_10_days = payload.get('ISSUE_COUNT_LAST_10_DAYS', 0)
    if issue_10_days is None:
        issue_10_days = 0
    transformed['ISSUE_COUNT_LAST_10_DAYS'] = int(round(float(issue_10_days)))
    
    # PREDICTED_ESCALATION_PROB
    escalation_prob = payload.get('PREDICTED_ESCALATION_PROB', 0)
    if escalation_prob is None:
        escalation_prob = 0
    transformed['PREDICTED_ESCALATION_PROB'] = float(escalation_prob)
    
    # SH_CNR
    sh_cnr = payload.get('SH_CNR', 0)
    if sh_cnr is None:
        sh_cnr = 0
    transformed['SH_CNR'] = round(max(0, float(sh_cnr)), 2)
    
    return transformed


def create_low_escalation_response(payload: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Create a response for cases with low escalation probability."""
    escalation_prob = payload.get('PREDICTED_ESCALATION_PROB', 0) or 0
    delivery_id = payload.get('DELIVERY_ID', 'unknown')
    
    return {
        'delivery_id': str(int(delivery_id)) if isinstance(delivery_id, float) else str(delivery_id),
        'eligible': False,
        'reasons': [f"Escalation probability ({escalation_prob:.2%}) below threshold ({threshold:.0%})"],
        'recommended_credit_range': '$0',
        'confidence': 'HIGH',
        'notes': f'Case skipped GPT inference due to low escalation probability.',
        'inference_type': 'threshold_skip'
    }


def apply_ceiling_logic(credit_range: str) -> float:
    """Apply ceiling logic to credit range prediction."""
    if not credit_range:
        return 0.0
    
    credit_range = str(credit_range).replace('\u2013', '-').replace('–', '-').strip()
    
    if credit_range == "$0":
        return 0.0
    
    # Range pattern: $X-$Y
    range_match = re.search(r'\$?(\d+)\s*[-–]\s*\$?(\d+)', credit_range)
    if range_match:
        upper_bound = int(range_match.group(2))
        return float(upper_bound)
    
    # Single value: $X
    single_match = re.search(r'\$(\d+)', credit_range)
    if single_match:
        return float(single_match.group(1))
    
    # Just a number
    num_match = re.search(r'(\d+)', credit_range)
    if num_match:
        return float(num_match.group(1))
    
    return 0.0


def csv_to_payloads(csv_path: str) -> list:
    """Convert CSV dataset to list of payload dictionaries."""
    df = pd.read_csv(csv_path)
    
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


class AsyncApologyCreditEngine:
    """Async version of the Apology Credit Engine."""
    
    def __init__(
        self,
        system_prompt: str,
        api_key: str,
        virtual_key: str,
        base_url: str = "https://cybertron-service-gateway.doordash.team/v1",
        model_name: str = 'gpt-4.1-2025-04-14',
        escalation_threshold: float = 0.2,
        max_concurrent: int = 10,
        timeout: int = 120
    ):
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.virtual_key = virtual_key
        self.base_url = base_url
        self.model_name = model_name
        self.escalation_threshold = escalation_threshold
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        self.base_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    async def evaluate_single(
        self,
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        index: int
    ) -> Dict[str, Any]:
        """Evaluate a single case asynchronously."""
        
        escalation_prob = payload.get('PREDICTED_ESCALATION_PROB', 0) or 0
        
        # Skip GPT if below threshold
        if escalation_prob < self.escalation_threshold:
            result = create_low_escalation_response(payload, self.escalation_threshold)
            credit_range = result.get('recommended_credit_range', '$0')
            ceiling_value = apply_ceiling_logic(credit_range)
            
            return {
                'index': index,
                'delivery_id': payload.get('DELIVERY_ID'),
                'output': result,
                'credit_range': credit_range,
                'ceiling_value': ceiling_value,
                'status': 'success',
                'inference_type': 'threshold_skip'
            }
        
        # Use GPT for high escalation probability
        async with semaphore:
            try:
                transformed_payload = transform_payload_for_model(payload)
                
                messages = self.base_messages.copy()
                messages.append({
                    "role": "user",
                    "content": json.dumps(transformed_payload, indent=2)
                })
                
                headers = {
                    "Content-Type": "application/json",
                    "x-portkey-api-key": self.api_key,
                    "x-portkey-virtual-key": self.virtual_key,
                }
                
                request_body = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.0
                }
                
                url = f"{self.base_url}/chat/completions"
                
                async with session.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    
                    response_json = await response.json()
                    response_text = response_json['choices'][0]['message']['content'].strip()
                    
                    # Parse JSON response
                    if response_text.startswith('```'):
                        lines = response_text.split('\n')
                        if lines[0].startswith('```'):
                            lines = lines[1:]
                        if lines[-1].strip() == '```':
                            lines = lines[:-1]
                        response_text = '\n'.join(lines)
                    
                    result = json.loads(response_text)
                    result['inference_type'] = 'gpt'
                    
                    credit_range = result.get('recommended_credit_range', '$0')
                    ceiling_value = apply_ceiling_logic(credit_range)
                    
                    return {
                        'index': index,
                        'delivery_id': payload.get('DELIVERY_ID'),
                        'output': result,
                        'credit_range': credit_range,
                        'ceiling_value': ceiling_value,
                        'status': 'success',
                        'inference_type': 'gpt'
                    }
                    
            except asyncio.TimeoutError:
                return {
                    'index': index,
                    'delivery_id': payload.get('DELIVERY_ID'),
                    'error': 'Request timeout',
                    'status': 'failed',
                    'credit_range': '$0',
                    'ceiling_value': 0.0
                }
            except json.JSONDecodeError as e:
                return {
                    'index': index,
                    'delivery_id': payload.get('DELIVERY_ID'),
                    'error': f'JSON parse error: {str(e)}',
                    'raw_response': response_text if 'response_text' in dir() else '',
                    'status': 'failed',
                    'credit_range': '$0',
                    'ceiling_value': 0.0
                }
            except Exception as e:
                return {
                    'index': index,
                    'delivery_id': payload.get('DELIVERY_ID'),
                    'error': str(e),
                    'status': 'failed',
                    'credit_range': '$0',
                    'ceiling_value': 0.0
                }
    
    async def evaluate_batch(
        self,
        payloads: List[Dict[str, Any]],
        progress_callback=None
    ) -> Tuple[List[Dict], float]:
        """Evaluate all payloads asynchronously."""
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self.evaluate_single(session, payload, semaphore, i)
                for i, payload in enumerate(payloads)
            ]
            
            results = []
            total_credit = 0.0
            
            # Process with progress bar
            for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Evaluating"):
                result = await coro
                results.append(result)
                total_credit += result.get('ceiling_value', 0.0)
                
                if progress_callback and len(results) % 100 == 0:
                    progress_callback(len(results), total_credit)
            
            # Sort by original index
            results.sort(key=lambda x: x['index'])
            
            return results, total_credit


async def main_async():
    parser = argparse.ArgumentParser(
        description='Run async full evaluation on escalated cases dataset'
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
        '--max-concurrent',
        type=int,
        default=20,
        help='Maximum concurrent API requests'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Request timeout in seconds'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ASYNC FULL DATASET EVALUATION - APOLOGY CREDIT ENGINE")
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
    
    # Count expected GPT vs skip
    gpt_expected = sum(1 for p in payloads if (p.get('PREDICTED_ESCALATION_PROB', 0) or 0) >= args.escalation_threshold)
    skip_expected = len(payloads) - gpt_expected
    print(f"Expected GPT calls: {gpt_expected}")
    print(f"Expected threshold skips: {skip_expected}")
    
    # Load prompt and config
    prompt_path = Path(args.prompt_file)
    system_prompt = load_system_prompt(prompt_path)
    config = load_config()
    
    print(f"Loaded prompt from: {args.prompt_file}")
    
    # Setup output directory
    output_base = Path(args.output_base)
    results_dir = output_base / 'results' / f'run{args.run_version}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nModel: {args.model}")
    print(f"Escalation Threshold: {args.escalation_threshold}")
    print(f"Max Concurrent Requests: {args.max_concurrent}")
    print(f"Timeout: {args.timeout}s")
    print(f"\nStarting evaluation of {len(payloads)} cases...\n")
    
    # Initialize async engine
    engine = AsyncApologyCreditEngine(
        system_prompt=system_prompt,
        api_key=config['api_key'],
        virtual_key=config['virtual_key'],
        model_name=args.model,
        escalation_threshold=args.escalation_threshold,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout
    )
    
    start_time = datetime.now()
    
    def progress_callback(processed, total_credit):
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"\n[{processed}/{len(payloads)}] Running total: ${total_credit:,.2f} | Rate: {rate:.1f}/s")
    
    # Run evaluation
    results, total_credit = await engine.evaluate_batch(payloads, progress_callback)
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # Calculate statistics
    successful = [r for r in results if r['status'] == 'success']
    failures = [r for r in results if r['status'] == 'failed']
    gpt_count = sum(1 for r in results if r.get('inference_type') == 'gpt')
    skip_count = sum(1 for r in results if r.get('inference_type') == 'threshold_skip')
    
    # Credit distribution
    credit_distribution = {}
    for r in results:
        cr = r.get('credit_range', '$0')
        credit_distribution[cr] = credit_distribution.get(cr, 0) + 1
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'async_evaluation_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': {
                'csv_file': args.csv_file,
                'prompt_file': args.prompt_file,
                'model': args.model,
                'escalation_threshold': args.escalation_threshold,
                'max_concurrent': args.max_concurrent,
                'total_cases': len(payloads),
                'gpt_inference': gpt_count,
                'threshold_skip': skip_count,
                'failures': len(failures),
                'elapsed_time_seconds': elapsed_time,
                'timestamp': timestamp
            },
            'summary': {
                'total_credit_ceiling': total_credit,
                'credit_distribution': credit_distribution
            },
            'results': results[:1000],  # Save first 1000 for inspection
            'failures': failures
        }, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal Cases Processed: {len(payloads)}")
    print(f"  - Successful: {len(successful)}")
    print(f"  - GPT Inference: {gpt_count}")
    print(f"  - Threshold Skip ($0): {skip_count}")
    print(f"  - Failures: {len(failures)}")
    print(f"\nElapsed Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Rate: {len(payloads)/elapsed_time:.1f} cases/second")
    
    print(f"\n{'='*60}")
    print(f"TOTAL APOLOGY CREDIT (Ceiling Logic): ${total_credit:,.2f}")
    print(f"{'='*60}")
    
    print(f"\nCredit Distribution:")
    for credit, count in sorted(credit_distribution.items(), key=lambda x: apply_ceiling_logic(x[0])):
        ceiling = apply_ceiling_logic(credit)
        print(f"  {credit}: {count} cases -> ${ceiling:.0f} each = ${ceiling * count:,.2f}")
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 70)
    
    return total_credit


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

