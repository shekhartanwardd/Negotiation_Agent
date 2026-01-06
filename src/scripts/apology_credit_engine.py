"""
Apology Credit Decision Engine

This module evaluates customer eligibility for apology credits based on
conversation transcripts and structured customer data using GPT-4.1.

The engine follows strict policy rules to determine:
1. Whether a customer is eligible for an apology credit
2. What credit range should be recommended

Author: Negotiation Agent Team
Date: 2025
"""

import os
import json
import copy
import logging
import argparse
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import pandas as pd
import portkey_ai
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class DataProcessingError(Exception):
    """Raised when data processing encounters an error."""
    pass


def load_config(config_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
        
    Returns:
        Dictionary containing API configuration.
        
    Raises:
        ConfigurationError: If config file is missing or invalid.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate required fields
        required_fields = ['api_key', 'virtual_key']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ConfigurationError(
                f"Missing required configuration fields: {', '.join(missing_fields)}"
            )
            
        return config
        
    except FileNotFoundError:
        raise ConfigurationError(
            f"Configuration file not found at {config_path}. "
            "Please create a config.json file with 'api_key' and 'virtual_key' fields."
        )
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}")


def load_system_prompt(prompt_path: Path) -> str:
    """
    Load system prompt from file.
    
    Args:
        prompt_path: Path to prompt file
        
    Returns:
        System prompt content
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    try:
        with open(prompt_path, 'r') as f:
            content = f.read()
            
        logger.info(f"Loaded system prompt from {prompt_path}")
        return content.strip()
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"System prompt file not found at {prompt_path}. "
            "Please ensure the prompt file exists."
        )


def reverse_log_transform(value: float) -> float:
    """
    Reverse log1p transformation: exp(value) - 1
    
    Args:
        value: Log-transformed value
        
    Returns:
        Original value before log transformation
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    
    try:
        # Handle negative values (sign-preserving log was used)
        if value < 0:
            return -(math.exp(abs(value)) - 1)
        return math.exp(value) - 1
    except (OverflowError, ValueError):
        return 0.0


def transform_payload_for_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the payload to appropriate types expected by the model.
    
    Note: The sampled payload files now contain ORIGINAL values (not log-transformed),
    so we use them directly without reverse log transformation.
    
    Args:
        payload: Payload with original (non-log-transformed) values
        
    Returns:
        Transformed payload with proper types for GPT consumption
    """
    transformed = {}
    
    # DELIVERY_ID - convert to string
    delivery_id = payload.get('DELIVERY_ID', 0)
    if isinstance(delivery_id, float):
        transformed['DELIVERY_ID'] = str(int(delivery_id))
    else:
        transformed['DELIVERY_ID'] = str(delivery_id)
    
    # CONVERSATION_CB - keep as-is (string)
    transformed['CONVERSATION_CB'] = payload.get('CONVERSATION_CB', '')
    
    # IS_CNR_ABUSER - already a risk score 0-1, threshold at 0.5 for binary
    cnr_score = payload.get('IS_CNR_ABUSER', 0)
    if cnr_score is None:
        cnr_score = 0
    transformed['IS_CNR_ABUSER'] = cnr_score > 0.5
    
    # ORDER_SUBTOTAL - already in dollars (original value)
    order_subtotal = payload.get('ORDER_SUBTOTAL', 0)
    if order_subtotal is None:
        order_subtotal = 0
    transformed['ORDER_SUBTOTAL'] = round(float(order_subtotal), 2)
    
    # IS_VIP_CUSTOMER - already binary (0 or 1)
    vip_value = payload.get('IS_VIP_CUSTOMER', 0)
    if vip_value is None:
        vip_value = 0
    transformed['IS_VIP_CUSTOMER'] = bool(int(vip_value))
    
    # ISSUE_COUNT_LAST_10_ORDERS - already an integer count (original value)
    issue_10_orders = payload.get('ISSUE_COUNT_LAST_10_ORDERS', 0)
    if issue_10_orders is None:
        issue_10_orders = 0
    transformed['ISSUE_COUNT_LAST_10_ORDERS'] = int(round(float(issue_10_orders)))
    
    # ISSUE_COUNT_LAST_10_DAYS - already an integer count (original value)
    issue_10_days = payload.get('ISSUE_COUNT_LAST_10_DAYS', 0)
    if issue_10_days is None:
        issue_10_days = 0
    transformed['ISSUE_COUNT_LAST_10_DAYS'] = int(round(float(issue_10_days)))
    
    # PREDICTED_ESCALATION_PROB - already 0-1 probability
    escalation_prob = payload.get('PREDICTED_ESCALATION_PROB', 0)
    if escalation_prob is None:
        escalation_prob = 0
    transformed['PREDICTED_ESCALATION_PROB'] = float(escalation_prob)
    
    # SH_CNR - already in dollars (original value)
    sh_cnr = payload.get('SH_CNR', 0)
    if sh_cnr is None:
        sh_cnr = 0
    # Handle negative values (which indicate no credit)
    transformed['SH_CNR'] = round(max(0, float(sh_cnr)), 2)
    
    return transformed


class ApologyCreditEngine:
    """
    LLM-based engine for evaluating apology credit eligibility.
    
    This class uses GPT-4.1 to determine whether customers are eligible
    for apology credits and recommends appropriate credit ranges.
    """
    
    SUPPORTED_MODELS = {
        'gpt-4.1': 'gpt-4.1-2025-04-14',
        'gpt-5-nano': 'gpt-5-nano-2025-08-07',
        'gpt-4.1-2025-04-14': 'gpt-4.1-2025-04-14',
        'gpt-5-nano-2025-08-07': 'gpt-5-nano-2025-08-07'
    }
    
    def __init__(
        self,
        system_prompt: str,
        api_key: Optional[str] = None,
        virtual_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = 'gpt-4.1-2025-04-14',
        run_version: int = 1,
        escalation_threshold: float = 0.5
    ):
        """
        Initialize the Apology Credit Engine.
        
        Args:
            system_prompt: System prompt defining the engine's behavior
            api_key: API key for Portkey AI
            virtual_key: Virtual key for Portkey AI
            base_url: Base URL for the API
            model_name: Name of the model to use
            run_version: Version number for this run (for output organization)
            escalation_threshold: Minimum PREDICTED_ESCALATION_PROB to use GPT inference
        """
        self.system_prompt = system_prompt
        self.run_version = run_version
        self.escalation_threshold = escalation_threshold
        
        # Resolve model name
        self.model_name = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        # Load configuration if credentials not provided
        if api_key is None or virtual_key is None:
            config = load_config()
            api_key = api_key or config.get('api_key')
            virtual_key = virtual_key or config.get('virtual_key')
        
        # Set up base URL with fallback
        if base_url is None:
            base_url = os.environ.get(
                "PORTKEY_BASE_URL", 
                "https://cybertron-service-gateway.doordash.team/v1"
            )
        
        # Initialize Portkey client with increased timeout
        import httpx
        # Create custom httpx client with longer timeout
        http_client = httpx.Client(
            timeout=httpx.Timeout(300.0, connect=60.0),  # 5 min read timeout, 1 min connect
        )
        self.client = portkey_ai.Portkey(
            base_url=base_url,
            api_key=api_key,
            virtual_key=virtual_key,
            timeout=300.0,  # 5 minutes timeout for long conversations
            http_client=http_client,
        )
        
        # Initialize base messages
        self.base_messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
        logger.info(f"Initialized ApologyCreditEngine with model {self.model_name}")
        logger.info(f"Escalation threshold: {self.escalation_threshold}")
    
    def _create_low_escalation_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a response for cases with low escalation probability (below threshold).
        These cases skip GPT inference and return $0 credit recommendation.
        
        Args:
            payload: Original payload with customer data (already in original values, not log-transformed)
            
        Returns:
            Response dict with $0 credit and reasoning
        """
        escalation_prob = payload.get('PREDICTED_ESCALATION_PROB', 0) or 0
        is_vip = payload.get('IS_VIP_CUSTOMER', 0) or 0
        is_cnr_abuser = payload.get('IS_CNR_ABUSER', 0) or 0
        order_subtotal = payload.get('ORDER_SUBTOTAL', 0) or 0
        issue_count_orders = payload.get('ISSUE_COUNT_LAST_10_ORDERS', 0) or 0
        issue_count_days = payload.get('ISSUE_COUNT_LAST_10_DAYS', 0) or 0
        sh_cnr = payload.get('SH_CNR', 0) or 0
        delivery_id = payload.get('DELIVERY_ID', 'unknown')
        
        # Build reasoning based on payload data
        reasons = [
            f"Escalation probability ({escalation_prob:.2%}) below threshold ({self.escalation_threshold:.0%})"
        ]
        
        if is_cnr_abuser and is_cnr_abuser >= 0.1:
            reasons.append(f"CNR abuser risk score: {is_cnr_abuser:.2%}")
        
        if sh_cnr and sh_cnr > 0:
            reasons.append(f"Already received self-help credit: ${sh_cnr:.2f}")
        
        notes_parts = [
            f"Case skipped GPT inference due to low escalation probability ({escalation_prob:.4f} < {self.escalation_threshold})."
        ]
        
        if is_vip and is_vip > 0:
            notes_parts.append(f"VIP customer status noted.")
        
        if order_subtotal and order_subtotal > 0:
            notes_parts.append(f"Order subtotal: ${order_subtotal:.2f}.")
        
        if issue_count_orders and issue_count_orders > 0:
            notes_parts.append(f"Issues in last 10 orders: {int(issue_count_orders)}.")
        
        if issue_count_days and issue_count_days > 0:
            notes_parts.append(f"Issues in last 10 days: {int(issue_count_days)}.")
        
        return {
            'delivery_id': str(int(delivery_id)) if isinstance(delivery_id, float) else str(delivery_id),
            'eligible_for_apology_credit': False,
            'eligibility_reason': reasons,
            'recommended_credit_range': '$0',
            'confidence': 'high',
            'notes': ' '.join(notes_parts),
            'inference_type': 'threshold_skip'
        }
    
    def evaluate_batch(
        self, 
        payloads: List[Dict[str, Any]], 
        temperature: float = 0.0,
        show_progress: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Evaluate a batch of customer cases.
        
        Args:
            payloads: List of customer case payloads
            temperature: Model temperature (0 for deterministic)
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (successful_results, failed_cases)
        """
        results = []
        failures = []
        
        # Configure progress tracking
        iterator = payloads
        if show_progress:
            iterator = tqdm(
                payloads, 
                desc=f"Evaluating with {self.model_name}",
                unit="cases"
            )
        
        for i, payload in enumerate(iterator):
            try:
                # Check escalation probability threshold
                escalation_prob = payload.get('PREDICTED_ESCALATION_PROB', 0)
                
                if escalation_prob >= self.escalation_threshold:
                    # High escalation probability - use GPT inference
                    result = self._evaluate_single(payload, temperature)
                    result['inference_type'] = 'gpt'
                else:
                    # Low escalation probability - skip GPT, return $0
                    result = self._create_low_escalation_response(payload)
                
                results.append({
                    'index': i,
                    'input': payload,
                    'output': result,
                    'status': 'success'
                })
                
            except Exception as e:
                error_info = {
                    'index': i,
                    'input': payload,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'status': 'failed'
                }
                failures.append(error_info)
                logger.error(f"Error evaluating case {i} (DELIVERY_ID: {payload.get('DELIVERY_ID')}): {str(e)}")
        
        return results, failures
    
    def _evaluate_single(
        self, 
        payload: Dict[str, Any], 
        temperature: float,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate a single customer case.
        
        Args:
            payload: Customer case payload
            temperature: Model temperature
            max_retries: Maximum number of retries on timeout
            
        Returns:
            Parsed JSON response from the model
        """
        import time
        
        # Transform payload for model consumption
        transformed_payload = transform_payload_for_model(payload)
        
        # Create user message with the payload
        user_message = {
            "role": "user",
            "content": json.dumps(transformed_payload, indent=2)
        }
        
        # Create conversation with system prompt
        messages = copy.deepcopy(self.base_messages)
        messages.append(user_message)
        
        # Retry logic for timeout errors
        last_error = None
        for attempt in range(max_retries):
            try:
                # Get model evaluation
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                
                response_text = completion.choices[0].message.content.strip()
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                if "timeout" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        else:
            # All retries exhausted
            raise last_error
        
        # Parse JSON response
        try:
            # Handle potential markdown code blocks
            if response_text.startswith('```'):
                # Remove markdown code block markers
                lines = response_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_text = '\n'.join(lines)
            
            result = json.loads(response_text)
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {response_text[:200]}...")
            # Return raw response wrapped in a structure
            return {
                'raw_response': response_text,
                'parse_error': str(e),
                'delivery_id': transformed_payload.get('DELIVERY_ID', 'unknown')
            }


def setup_output_directories(base_path: Path, run_version: int) -> Tuple[Path, Path]:
    """
    Set up output directories for results and logs.
    
    Args:
        base_path: Base path for outputs
        run_version: Run version number
        
    Returns:
        Tuple of (results_dir, logs_dir)
    """
    results_dir = base_path / 'results' / f'run{run_version}'
    logs_dir = base_path / 'logs' / f'run{run_version}'
    
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Logs will be saved to: {logs_dir}")
    
    return results_dir, logs_dir


def save_results(
    results: List[Dict[str, Any]], 
    failures: List[Dict[str, Any]],
    results_dir: Path,
    logs_dir: Path,
    run_metadata: Dict[str, Any]
) -> None:
    """
    Save evaluation results and failure logs.
    
    Args:
        results: List of successful evaluation results
        failures: List of failed cases
        results_dir: Directory for results
        logs_dir: Directory for logs
        run_metadata: Metadata about the run
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save successful results
    results_file = results_dir / f'results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': run_metadata,
            'results': results
        }, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} results to {results_file}")
    
    # Save summary statistics
    summary = generate_summary(results)
    summary_file = results_dir / f'summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'metadata': run_metadata,
            'summary': summary
        }, f, indent=2, default=str)
    logger.info(f"Saved summary to {summary_file}")
    
    # Save failures to logs
    if failures:
        failures_file = logs_dir / f'failures_{timestamp}.json'
        with open(failures_file, 'w') as f:
            json.dump({
                'metadata': run_metadata,
                'total_failures': len(failures),
                'failures': failures
            }, f, indent=2, default=str)
        logger.warning(f"Saved {len(failures)} failures to {failures_file}")
    
    # Save run log
    run_log = {
        'timestamp': timestamp,
        'metadata': run_metadata,
        'total_processed': len(results) + len(failures),
        'successful': len(results),
        'failed': len(failures),
        'success_rate': len(results) / (len(results) + len(failures)) if (len(results) + len(failures)) > 0 else 0
    }
    run_log_file = logs_dir / f'run_log_{timestamp}.json'
    with open(run_log_file, 'w') as f:
        json.dump(run_log, f, indent=2, default=str)


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Summary statistics dictionary
    """
    total = len(results)
    if total == 0:
        return {'total': 0, 'message': 'No results to summarize'}
    
    # Count eligibility
    eligible_count = 0
    ineligible_count = 0
    parse_errors = 0
    
    # Track credit ranges
    credit_ranges = {}
    
    # Track confidence levels
    confidence_levels = {'high': 0, 'medium': 0, 'low': 0}
    
    # Track eligibility reasons
    eligibility_reasons = {}
    
    for result in results:
        output = result.get('output', {})
        
        if 'parse_error' in output:
            parse_errors += 1
            continue
        
        # Eligibility
        if output.get('eligible_for_apology_credit', False):
            eligible_count += 1
        else:
            ineligible_count += 1
        
        # Credit range
        credit_range = output.get('recommended_credit_range', 'unknown')
        credit_ranges[credit_range] = credit_ranges.get(credit_range, 0) + 1
        
        # Confidence
        confidence = output.get('confidence', 'unknown')
        if confidence in confidence_levels:
            confidence_levels[confidence] += 1
        
        # Eligibility reasons
        reasons = output.get('eligibility_reason', [])
        if isinstance(reasons, list):
            for reason in reasons:
                eligibility_reasons[reason] = eligibility_reasons.get(reason, 0) + 1
    
    return {
        'total_processed': total,
        'eligible': eligible_count,
        'ineligible': ineligible_count,
        'parse_errors': parse_errors,
        'eligibility_rate': eligible_count / (eligible_count + ineligible_count) if (eligible_count + ineligible_count) > 0 else 0,
        'credit_range_distribution': credit_ranges,
        'confidence_distribution': confidence_levels,
        'top_eligibility_reasons': dict(sorted(eligibility_reasons.items(), key=lambda x: x[1], reverse=True)[:10])
    }


def load_payloads(payload_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load customer case payloads from JSON file.
    
    Args:
        payload_path: Path to JSON payload file
        limit: Optional limit on number of cases to load
        
    Returns:
        List of payload dictionaries
    """
    logger.info(f"Loading payloads from {payload_path}")
    
    with open(payload_path, 'r') as f:
        payloads = json.load(f)
    
    if limit:
        payloads = payloads[:limit]
        logger.info(f"Limiting to first {limit} cases")
    
    logger.info(f"Loaded {len(payloads)} payloads")
    return payloads


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Apology Credit Decision Engine - Evaluate customer eligibility for apology credits'
    )
    
    parser.add_argument(
        '--payload-file',
        type=str,
        default='/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/escalated_cases_payload.json',
        help='Path to the JSON file containing customer case payloads'
    )
    
    parser.add_argument(
        '--prompt-file',
        type=str,
        default='/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/system_prompt/prompt.txt',
        help='Path to the system prompt file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['gpt-4.1', 'gpt-4.1-2025-04-14', 'gpt-5-nano', 'gpt-5-nano-2025-08-07'],
        default='gpt-4.1-2025-04-14',
        help='Model to use for evaluation'
    )
    
    parser.add_argument(
        '--run-version',
        type=int,
        default=1,
        help='Run version number (for organizing outputs)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Limit number of cases to process (default: 1000, use -1 for all)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Model temperature (0 for deterministic)'
    )
    
    parser.add_argument(
        '--output-base',
        type=str,
        default='/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent',
        help='Base path for output directories'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default=None,
        help='Path to config.json file (default: same directory as script)'
    )
    
    parser.add_argument(
        '--escalation-threshold',
        type=float,
        default=0.5,
        help='Minimum PREDICTED_ESCALATION_PROB to use GPT inference (default: 0.5). Cases below this threshold get $0 credit.'
    )
    
    args = parser.parse_args()
    
    # Handle limit=-1 for all cases
    limit = None if args.limit == -1 else args.limit
    
    # Setup paths
    payload_path = Path(args.payload_file)
    prompt_path = Path(args.prompt_file)
    output_base = Path(args.output_base)
    config_path = Path(args.config_file) if args.config_file else None
    
    # Create run metadata
    run_metadata = {
        'run_version': args.run_version,
        'model': args.model,
        'temperature': args.temperature,
        'escalation_threshold': args.escalation_threshold,
        'limit': limit,
        'payload_file': str(payload_path),
        'prompt_file': str(prompt_path),
        'start_time': datetime.now().isoformat()
    }
    
    try:
        # Setup output directories
        results_dir, logs_dir = setup_output_directories(output_base, args.run_version)
        
        # Load system prompt
        system_prompt = load_system_prompt(prompt_path)
        
        # Load payloads
        payloads = load_payloads(payload_path, limit)
        
        # Initialize engine
        engine = ApologyCreditEngine(
            system_prompt=system_prompt,
            model_name=args.model,
            run_version=args.run_version,
            escalation_threshold=args.escalation_threshold
        )
        
        # Run evaluation
        logger.info(f"Starting evaluation of {len(payloads)} cases...")
        results, failures = engine.evaluate_batch(
            payloads, 
            temperature=args.temperature
        )
        
        # Update metadata with end time
        run_metadata['end_time'] = datetime.now().isoformat()
        run_metadata['total_processed'] = len(results) + len(failures)
        run_metadata['successful'] = len(results)
        run_metadata['failed'] = len(failures)
        
        # Save results
        save_results(results, failures, results_dir, logs_dir, run_metadata)
        
        # Print summary
        print("\n" + "=" * 60)
        print("APOLOGY CREDIT ENGINE - RUN COMPLETE")
        print("=" * 60)
        print(f"Run Version: {args.run_version}")
        print(f"Model: {args.model}")
        print(f"Escalation Threshold: {args.escalation_threshold}")
        print(f"Total Processed: {len(results) + len(failures)}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(failures)}")
        
        # Count GPT vs threshold-skipped cases
        gpt_cases = sum(1 for r in results if r.get('output', {}).get('inference_type') == 'gpt')
        skipped_cases = sum(1 for r in results if r.get('output', {}).get('inference_type') == 'threshold_skip')
        print(f"\nInference Breakdown:")
        print(f"  GPT Inference: {gpt_cases}")
        print(f"  Threshold Skip ($0): {skipped_cases}")
        
        if results:
            summary = generate_summary(results)
            print(f"\nEligibility Rate: {summary['eligibility_rate']:.2%}")
            print(f"Eligible: {summary['eligible']}")
            print(f"Ineligible: {summary['ineligible']}")
            print(f"\nCredit Range Distribution:")
            for range_name, count in sorted(summary['credit_range_distribution'].items()):
                print(f"  {range_name}: {count}")
            print(f"\nConfidence Distribution:")
            for level, count in summary['confidence_distribution'].items():
                print(f"  {level}: {count}")
        
        print(f"\nResults saved to: {results_dir}")
        print(f"Logs saved to: {logs_dir}")
        print("=" * 60)
        
        logger.info("Evaluation complete.")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

