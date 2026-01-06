"""
Enhanced Apology Credit Parser Module

This module provides two main functionalities:
1. Parse conversations to extract chatbot-only portion (before human agent transfer)
2. Extract apology credit amounts mentioned in conversations

================================================================================
USAGE INSTRUCTIONS
================================================================================

In your notebook (e.g., gpt_negotiation_experiment.ipynb):

    # Step 1: Add scripts path
    import sys
    sys.path.insert(0, '../scripts')
    
    # Step 2: Import functions
    from parse_apology_credits import (
        extract_conversation_cb,                    # Parse chatbot portion
        extract_apology_credits_from_dataframe,     # Extract apology credits
        extract_apology_credit,                     # Single conversation -> amount
        process_dataframe                           # All-in-one processing
    )
    
    # Step 3: Use the functions
    
    # Option A: All-in-one processing (recommended)
    df = process_dataframe(
        df,
        conversation_col='CONVERSATION',
        extract_cb=True,           # Add CONVERSATION_CB column
        extract_ac=True,           # Add Extracted_AC column
        include_ac_details=False   # Set True for match details
    )
    
    # Option B: Just extract chatbot conversation
    df = extract_conversation_cb(df, conversation_col='CONVERSATION')
    
    # Option C: Just extract apology credits
    df = extract_apology_credits_from_dataframe(df, conversation_col='CONVERSATION')
    
    # Option D: Single conversation extraction
    amount = extract_apology_credit("I will give you $10 additional credits")

================================================================================
PATTERNS CAPTURED FOR APOLOGY CREDITS
================================================================================
1. "$X credits" / "$X additional credits" / "$X apology credits"
2. "credits of $X" / "credits amounting $X"
3. "$X in credits" / "$X in DoorDash credits"
4. "added $X credits" / "processed $X credits" / "issued $X credits"
5. "token of apology ... $X"
6. "$X for inconvenience" / "$X for the inconvenience"
7. Various Australian dollar formats (A$X)

================================================================================
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ==============================================================================
# CONVERSATION PARSING FUNCTIONS (Extract Chatbot Portion)
# ==============================================================================

def parse_conversation_before_agent_transfer(conversation: str) -> str:
    """
    Parse a conversation and return only the chatbot portion 
    (everything before human agent transfer).
    
    The function looks for the "System: Connecting you with an agent..." message
    and returns everything before that point.
    
    Args:
        conversation: String containing the full conversation
        
    Returns:
        str: Parsed conversation up to (but not including) the agent transfer message
             Returns the original conversation if no transfer message is found
             Returns empty string if conversation is None/NaN
             
    Example:
        >>> cb_only = parse_conversation_before_agent_transfer(full_conversation)
    """
    # Handle None or NaN values
    if conversation is None:
        return ""
    
    if PANDAS_AVAILABLE:
        if pd.isna(conversation):
            return ""
    
    conversation = str(conversation)
    
    # Pattern to match the system message indicating agent transfer
    # This captures the point where the customer is being connected to an agent
    transfer_patterns = [
        r"System:\s*Connecting you with an agent\.*",
        r"System:\s*You are now connected to our support agent",
        r"System:\s*Transferring you to an agent",
    ]
    
    # Find the earliest occurrence of any transfer pattern
    earliest_pos = len(conversation)
    for pattern in transfer_patterns:
        match = re.search(pattern, conversation, re.IGNORECASE)
        if match:
            earliest_pos = min(earliest_pos, match.start())
    
    # If no transfer message found, return the full conversation
    if earliest_pos == len(conversation):
        return conversation.strip()
    
    # Return everything before the transfer message
    parsed = conversation[:earliest_pos].strip()
    
    return parsed


def extract_conversation_cb(
    df,
    conversation_col: str = 'CONVERSATION',
    output_col: str = 'CONVERSATION_CB',
    verbose: bool = True
):
    """
    Add a new column with parsed conversations (chatbot portion only).
    
    This extracts the portion of each conversation that occurred before
    the customer was transferred to a human agent.
    
    Args:
        df: pandas DataFrame containing conversation data
        conversation_col: Name of the column containing full conversation text
            (default: 'CONVERSATION')
        output_col: Name of the new column to store chatbot-only conversation
            (default: 'CONVERSATION_CB')
        verbose: If True, print progress information
            (default: True)
    
    Returns:
        pandas DataFrame with new column added
        
    Example:
        >>> import sys
        >>> sys.path.insert(0, '../scripts')
        >>> from parse_apology_credits import extract_conversation_cb
        >>> 
        >>> df = extract_conversation_cb(df, conversation_col='CONVERSATION')
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for this function. Install with: pip install pandas")
    
    df = df.copy()
    
    if conversation_col not in df.columns:
        raise ValueError(f"Column '{conversation_col}' not found in DataFrame. "
                        f"Available columns: {list(df.columns)}")
    
    if verbose:
        print(f"Parsing {len(df):,} conversations...")
    
    # Apply the parsing function
    df[output_col] = df[conversation_col].apply(parse_conversation_before_agent_transfer)
    
    # Calculate stats
    if verbose:
        non_empty = (df[output_col].str.len() > 0).sum()
        truncated = (df[output_col].str.len() < df[conversation_col].fillna('').str.len()).sum()
        
        print(f"✓ Created column '{output_col}'")
        print(f"  - Non-empty conversations: {non_empty:,}")
        print(f"  - Conversations truncated (had agent transfer): {truncated:,}")
    
    return df


def process_dataframe(
    df,
    conversation_col: str = 'CONVERSATION',
    extract_cb: bool = True,
    cb_output_col: str = 'CONVERSATION_CB',
    extract_ac: bool = True,
    ac_output_col: str = 'Extracted_AC',
    include_ac_details: bool = False,
    ac_details_col: str = 'Extracted_AC_Details',
    ac_from_full_conversation: bool = True,
    verbose: bool = True
):
    """
    All-in-one function to process a DataFrame with conversation data.
    
    This function can:
    1. Extract chatbot-only portion (CONVERSATION_CB)
    2. Extract apology credit amounts from conversations
    
    Args:
        df: pandas DataFrame containing conversation data
        conversation_col: Name of the column containing conversation text
            (default: 'CONVERSATION')
        extract_cb: If True, add CONVERSATION_CB column with chatbot-only text
            (default: True)
        cb_output_col: Name for the chatbot conversation column
            (default: 'CONVERSATION_CB')
        extract_ac: If True, add Extracted_AC column with apology credit amounts
            (default: True)
        ac_output_col: Name for the apology credit amount column
            (default: 'Extracted_AC')
        include_ac_details: If True, also add column with match details
            (default: False)
        ac_details_col: Name for the apology credit details column
            (default: 'Extracted_AC_Details')
        ac_from_full_conversation: If True, extract AC from full conversation;
            if False, extract from CONVERSATION_CB only
            (default: True - AC is usually given by human agents)
        verbose: If True, print progress information
            (default: True)
    
    Returns:
        pandas DataFrame with new columns added
        
    Example:
        >>> import sys
        >>> sys.path.insert(0, '../scripts')
        >>> from parse_apology_credits import process_dataframe
        >>> 
        >>> # Process with all options
        >>> df = process_dataframe(
        ...     df,
        ...     conversation_col='CONVERSATION',
        ...     extract_cb=True,           # Add CONVERSATION_CB
        ...     extract_ac=True,           # Add Extracted_AC
        ...     include_ac_details=True    # Add Extracted_AC_Details
        ... )
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for this function. Install with: pip install pandas")
    
    df = df.copy()
    
    if verbose:
        print("=" * 60)
        print("Processing DataFrame")
        print("=" * 60)
    
    # Step 1: Extract chatbot conversation if requested
    if extract_cb:
        df = extract_conversation_cb(
            df,
            conversation_col=conversation_col,
            output_col=cb_output_col,
            verbose=verbose
        )
    
    # Step 2: Extract apology credits if requested
    if extract_ac:
        if verbose:
            print()  # Add spacing
        
        # Determine which column to use for AC extraction
        if ac_from_full_conversation:
            ac_source_col = conversation_col
        else:
            ac_source_col = cb_output_col if extract_cb else conversation_col
        
        df = extract_apology_credits_from_dataframe(
            df,
            conversation_col=ac_source_col,
            output_col=ac_output_col,
            include_details=include_ac_details,
            details_col=ac_details_col,
            verbose=verbose
        )
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Processing complete!")
        print("=" * 60)
    
    return df


# ==============================================================================
# APOLOGY CREDIT EXTRACTION CLASSES AND FUNCTIONS
# ==============================================================================

@dataclass
class CreditMatch:
    """Represents a matched credit amount with context."""
    amount: float
    pattern_type: str
    context: str
    currency: str = "USD"


def extract_dollar_amount(text: str) -> Optional[float]:
    """
    Extract a dollar amount from text like '$10.00', '$5', 'A$3.00'.
    
    Args:
        text: String containing a dollar amount
        
    Returns:
        Float value of the amount, or None if parsing fails
    """
    # Remove currency prefix (A$ for AUD, $ for USD)
    cleaned = re.sub(r'^A?\$', '', text.strip())
    # Remove commas from numbers like $1,000.00
    cleaned = cleaned.replace(',', '')
    # Remove trailing punctuation
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned)
    
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_apology_credit_context(context: str) -> bool:
    """
    Check if the context indicates this is an apology/additional credit,
    not just a regular refund.
    
    Apology credits typically have keywords like:
    - "apology", "apologize", "sorry"
    - "additional", "extra", "more"
    - "inconvenience", "trouble", "delay"
    - "token of", "gesture"
    - "on top of", "also"
    
    We want to EXCLUDE regular refunds like:
    - "processed credits of $24.89" (order refund)
    - "full refund"
    - "refund of"
    
    Args:
        context: The text context around the dollar amount
        
    Returns:
        True if this looks like an apology credit, False otherwise
    """
    context_lower = context.lower()
    
    # Strong indicators of apology credit
    apology_keywords = [
        'apology', 'apologize', 'sorry',
        'additional', 'extra', 'more',
        'inconvenience', 'trouble', 'delay',
        'token of', 'gesture',
        'on top of', 'also give', 'also add', 'also process',
        'for your time', 'for the wait',
        'compensation', 'goodwill'
    ]
    
    # Check if any apology keyword is present
    has_apology_keyword = any(kw in context_lower for kw in apology_keywords)
    
    # Keywords that suggest this is a refund, not apology credit
    refund_keywords = [
        'full refund', 'refund of', 'total refund',
        'order amount', 'order total',
        'processed credits of',  # Usually order refund
        'issued $$',  # Double dollar sign typo for refund
    ]
    
    # If it looks like a refund without apology keywords, skip it
    is_likely_refund = any(kw in context_lower for kw in refund_keywords)
    
    # Return True only if it has apology keywords and doesn't look like a pure refund
    # OR if it explicitly mentions "additional credits"
    if 'additional' in context_lower and 'credit' in context_lower:
        return True
    
    if has_apology_keyword and not is_likely_refund:
        return True
    
    # Special case: "with additional credits of $X" - this is definitely apology credit
    if 'with additional credits' in context_lower:
        return True
    
    return False


def find_apology_credits(conversation: str) -> List[CreditMatch]:
    """
    Find all apology credit mentions in a conversation.
    
    This function uses multiple regex patterns to capture different ways
    apology credits are mentioned in conversations. It specifically looks
    for APOLOGY/ADDITIONAL credits, not regular refunds.
    
    Args:
        conversation: The full conversation text
        
    Returns:
        List of CreditMatch objects with extracted amounts
    """
    if not conversation or not isinstance(conversation, str):
        return []
    
    matches = []
    
    # Pattern 1: "$X additional credit(s)" - explicitly additional
    # e.g., "$10 additional credits", "$5.00 additional DoorDash credits"
    pattern1 = r'(?:A?\$)([\d,.]+)\s+additional\s+(?:DoorDash\s+)?credits?'
    for match in re.finditer(pattern1, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-50):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "additional_credits", context))
    
    # Pattern 2: "$X apology credit(s)" - explicitly apology
    # e.g., "$10 apology credits", "$5 apology credit"
    pattern2 = r'(?:A?\$)([\d,.]+)\s+apology\s+credits?'
    for match in re.finditer(pattern2, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-50):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "apology_credits", context))
    
    # Pattern 3: "additional credits of $X" or "with additional credits of $X"
    # e.g., "with additional credits of $10.00"
    pattern3 = r'(?:with\s+)?additional\s+credits?\s+(?:of\s+)?(?:A?\$)([\d,.]+)'
    for match in re.finditer(pattern3, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-50):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "additional_credits_of", context))
    
    # Pattern 4: "$X for inconvenience" or "$X for the inconvenience"
    # e.g., "$10 for the inconvenience"
    pattern4 = r'(?:A?\$)([\d,.]+)\s+(?:credits?\s+)?(?:for\s+)?(?:the\s+)?inconvenience'
    for match in re.finditer(pattern4, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-50):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "for_inconvenience", context))
    
    # Pattern 5: "token of apology" with dollar amount
    # e.g., "as a token of apology I processed $5.00"
    pattern5 = r'token\s+of\s+apology[^$]{0,50}(?:A?\$)([\d,.]+)'
    for match in re.finditer(pattern5, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-20):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "token_of_apology", context))
    
    # Pattern 6: "as an apology" or "as apology" with dollar amount
    # e.g., "as an apology I have added $5.00 credits"
    pattern6 = r'as\s+(?:an?\s+)?apology[^$]{0,50}(?:A?\$)([\d,.]+)'
    for match in re.finditer(pattern6, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-20):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "as_apology", context))
    
    # Pattern 7: "$X more as apology" or "$X as apology"
    pattern7 = r'(?:A?\$)([\d,.]+)\s+(?:more\s+)?as\s+(?:an?\s+)?apology'
    for match in re.finditer(pattern7, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-30):min(len(conversation), match.end()+30)]
            matches.append(CreditMatch(amount, "dollar_as_apology", context))
    
    # Pattern 8: "I will also give you $X" (additional compensation)
    pattern8 = r'(?:will\s+)?also\s+(?:give|add|process|issue)\s+(?:you\s+)?(?:A?\$)([\d,.]+)'
    for match in re.finditer(pattern8, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-30):min(len(conversation), match.end()+50)]
            # Check if it's credit-related
            if re.search(r'credit', context, re.IGNORECASE):
                matches.append(CreditMatch(amount, "also_give_credits", context))
    
    # Pattern 9: "$X more as apology credits" or "issued $X more"
    pattern9 = r'(?:issued?|add(?:ed)?|process(?:ed)?)\s+(?:A?\$)([\d,.]+)\s+more'
    for match in re.finditer(pattern9, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-30):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "issued_more", context))
    
    # Pattern 10: "on top of that" + dollar amount (extra compensation)
    pattern10 = r'on\s+top\s+of\s+that[^$]{0,50}(?:A?\$)([\d,.]+)'
    for match in re.finditer(pattern10, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-20):min(len(conversation), match.end()+50)]
            matches.append(CreditMatch(amount, "on_top_of", context))
    
    # Pattern 11: "offer you $X in DoorDash credits for your time"
    pattern11 = r'offer\s+(?:you\s+)?(?:A?\$)([\d,.]+)\s+(?:in\s+)?(?:DoorDash\s+)?credits?\s+for\s+(?:your\s+)?(?:time|inconvenience|trouble|delay)'
    for match in re.finditer(pattern11, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-30):min(len(conversation), match.end()+30)]
            matches.append(CreditMatch(amount, "offer_for_time", context))
    
    # Pattern 12: Chatbot offering credits for late delivery
    # e.g., "I can offer you $5.00 in DoorDash credits for your time"
    pattern12 = r'(?:I\s+)?can\s+offer\s+(?:you\s+)?(?:A?\$)([\d,.]+)\s+(?:in\s+)?(?:DoorDash\s+)?credits?'
    for match in re.finditer(pattern12, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-30):min(len(conversation), match.end()+50)]
            # Only include if context suggests it's for inconvenience/delay
            if is_apology_credit_context(context):
                matches.append(CreditMatch(amount, "can_offer_credits", context))
    
    # Pattern 13: "$X credits for the delay/wait/trouble"
    pattern13 = r'(?:A?\$)([\d,.]+)\s+(?:in\s+)?(?:DoorDash\s+)?credits?\s+for\s+(?:the\s+)?(?:delay|wait|trouble|issue|problem)'
    for match in re.finditer(pattern13, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-30):min(len(conversation), match.end()+30)]
            matches.append(CreditMatch(amount, "credits_for_delay", context))
    
    # Pattern 14: Australian dollars - "A$X apology/additional credits"
    pattern14 = r'A\$([\d,.]+)\s+(?:additional\s+)?(?:apology\s+)?credits?'
    for match in re.finditer(pattern14, conversation, re.IGNORECASE):
        amount = extract_dollar_amount('$' + match.group(1))
        if amount:
            context = conversation[max(0, match.start()-30):min(len(conversation), match.end()+50)]
            if is_apology_credit_context(context):
                matches.append(CreditMatch(amount, "aud_apology", context, "AUD"))
    
    return matches


def extract_dollar_position(context: str, amount: float) -> Optional[int]:
    """
    Find the position of the dollar amount in the context string.
    
    Args:
        context: The context string containing the dollar amount
        amount: The dollar amount to find
        
    Returns:
        Position of the dollar sign, or None if not found
    """
    # Format amount patterns to search for
    amount_patterns = [
        f"${amount:.2f}",
        f"${amount:.0f}",
        f"${int(amount)}",
        f"A${amount:.2f}",
        f"A${amount:.0f}",
    ]
    
    for pattern in amount_patterns:
        pos = context.find(pattern)
        if pos != -1:
            return pos
    
    return None


def deduplicate_matches(matches: List[CreditMatch]) -> List[CreditMatch]:
    """
    Remove duplicate matches that refer to the same credit.
    
    This improved deduplication:
    1. Same dollar amounts for the same type of issue are usually the same credit
    2. Keeps truly distinct credit amounts (e.g., $5 + $7 = $12 is valid)
    3. Prioritizes agent mentions over customer references
    
    Logic:
    - If we see the same amount multiple times, it's likely the same credit
      being mentioned multiple times (agent offers, customer references)
    - Exception: different amounts are distinct credits ($5 + $7 = $12 total)
    
    Args:
        matches: List of CreditMatch objects
        
    Returns:
        Deduplicated list of CreditMatch objects
    """
    if not matches:
        return []
    
    # Group matches by amount
    amount_to_matches = {}
    for match in matches:
        if match.amount not in amount_to_matches:
            amount_to_matches[match.amount] = []
        amount_to_matches[match.amount].append(match)
    
    unique_matches = []
    
    for amount, amount_matches in amount_to_matches.items():
        if len(amount_matches) == 1:
            # Only one match for this amount, keep it
            unique_matches.append(amount_matches[0])
        else:
            # Multiple matches for same amount
            # Check if they look like distinct credits or same credit mentioned multiple times
            
            # Prioritize agent context over customer context
            agent_matches = []
            other_matches = []
            
            for m in amount_matches:
                context_lower = m.context.lower()
                # Check for agent indicators
                if any(indicator in context_lower for indicator in [
                    'human agent:', 'agent:', 'i will', 'i have', 'i\'ve', 
                    'processed', 'issued', 'added', 'give you'
                ]):
                    agent_matches.append(m)
                else:
                    other_matches.append(m)
            
            # If we have agent matches, prefer the first one
            # (assuming multiple mentions of same amount are the same credit)
            if agent_matches:
                # Check if agent matches look like genuinely different credits
                # e.g., offered at different times for different reasons
                contexts = [m.context.lower() for m in agent_matches]
                
                # Simple heuristic: if contexts are very different, they might be different credits
                # But by default, assume same amount = same credit
                unique_matches.append(agent_matches[0])
            elif other_matches:
                unique_matches.append(other_matches[0])
    
    return unique_matches


def calculate_total_apology_credit(conversation: str) -> Tuple[float, List[dict]]:
    """
    Calculate the total apology credit from a conversation.
    
    Args:
        conversation: The full conversation text
        
    Returns:
        Tuple of (total_credit, list of match details)
    """
    matches = find_apology_credits(conversation)
    unique_matches = deduplicate_matches(matches)
    
    # Sum up all unique credit amounts
    total = sum(m.amount for m in unique_matches)
    
    # Create match details for debugging/auditing
    details = [
        {
            "amount": m.amount,
            "pattern": m.pattern_type,
            "context": m.context[:100],
            "currency": m.currency
        }
        for m in unique_matches
    ]
    
    return total, details


def extract_apology_credit(conversation: str) -> float:
    """
    Extract total apology credit amount from a single conversation string.
    
    This is a convenience function for extracting just the total amount.
    
    Args:
        conversation: The conversation text
        
    Returns:
        Total apology credit amount as a float
    
    Example:
        >>> amount = extract_apology_credit("I will give you $10 additional credits")
        >>> print(amount)  # 10.0
    """
    total, _ = calculate_total_apology_credit(conversation)
    return round(total, 2)


def extract_apology_credit_with_details(conversation: str) -> dict:
    """
    Extract apology credit with full details from a single conversation.
    
    Args:
        conversation: The conversation text
        
    Returns:
        Dictionary with 'amount' and 'details' keys
        
    Example:
        >>> result = extract_apology_credit_with_details(conversation)
        >>> print(result['amount'])  # 10.0
        >>> print(result['details'])  # [{'amount': 10.0, 'pattern': 'additional_credits', ...}]
    """
    total, details = calculate_total_apology_credit(conversation)
    return {
        'amount': round(total, 2),
        'details': details
    }


def extract_apology_credits_from_dataframe(
    df,
    conversation_col: str = 'CONVERSATION',
    output_col: str = 'Extracted_AC',
    include_details: bool = False,
    details_col: str = 'Extracted_AC_Details',
    verbose: bool = True
):
    """
    Add extracted apology credit column(s) to a pandas DataFrame.
    
    This function processes each row's conversation column and extracts
    apology credits, adding the results as new column(s).
    
    Args:
        df: pandas DataFrame containing conversation data
        conversation_col: Name of the column containing conversation text
            (default: 'CONVERSATION')
        output_col: Name of the column to store extracted amounts
            (default: 'Extracted_AC')
        include_details: If True, also add a column with match details
            (default: False)
        details_col: Name of the column to store match details
            (default: 'Extracted_AC_Details')
        verbose: If True, print progress information
            (default: True)
    
    Returns:
        pandas DataFrame with new column(s) added
        
    Example:
        >>> import sys
        >>> sys.path.insert(0, '../scripts')
        >>> from parse_apology_credits import extract_apology_credits_from_dataframe
        >>> 
        >>> # Add just the amount
        >>> df = extract_apology_credits_from_dataframe(df, conversation_col='CONVERSATION')
        >>> 
        >>> # Add amount and details
        >>> df = extract_apology_credits_from_dataframe(
        ...     df, 
        ...     conversation_col='CONVERSATION',
        ...     include_details=True
        ... )
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for this function. Install with: pip install pandas")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    if conversation_col not in df.columns:
        raise ValueError(f"Column '{conversation_col}' not found in DataFrame. "
                        f"Available columns: {list(df.columns)}")
    
    total_rows = len(df)
    if verbose:
        print(f"Extracting apology credits from {total_rows:,} rows...")
    
    # Process each row
    amounts = []
    details_list = []
    
    for idx, row in df.iterrows():
        conversation = row.get(conversation_col, '')
        if pd.isna(conversation):
            conversation = ''
        
        total, details = calculate_total_apology_credit(str(conversation))
        amounts.append(round(total, 2))
        if include_details:
            details_list.append(details if details else None)
    
    # Add columns to DataFrame
    df[output_col] = amounts
    if include_details:
        df[details_col] = details_list
    
    # Print statistics
    if verbose:
        with_ac = sum(1 for a in amounts if a > 0)
        total_amount = sum(amounts)
        print(f"✓ Added column '{output_col}'")
        print(f"  - Rows with extracted AC > 0: {with_ac:,} ({100*with_ac/total_rows:.1f}%)")
        print(f"  - Total extracted amount: ${total_amount:,.2f}")
        if with_ac > 0:
            print(f"  - Average (when > 0): ${total_amount/with_ac:.2f}")
        if include_details:
            print(f"  - Also added details column: '{details_col}'")
    
    return df


def process_payload_file(input_path: Path, output_path: Path) -> dict:
    """
    Process a payload file and add extracted apology credits.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        
    Returns:
        Statistics about the processing
    """
    print(f"Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} records...")
    
    stats = {
        "total_records": len(data),
        "records_with_extracted_ac": 0,
        "total_extracted_amount": 0.0,
        "records_with_existing_ac": 0,
        "records_with_both": 0
    }
    
    for record in data:
        conversation = record.get('CONVERSATION_CB', '') or record.get('CONVERSATION', '')
        
        # Extract apology credits
        total_ac, details = calculate_total_apology_credit(conversation)
        
        # Add new fields
        record['Extracted_AC'] = round(total_ac, 2)
        record['Extracted_AC_Details'] = details if details else None
        
        # Update stats
        if total_ac > 0:
            stats["records_with_extracted_ac"] += 1
            stats["total_extracted_amount"] += total_ac
        
        existing_ac = record.get('Parsed_AC', 0) or 0
        if existing_ac > 0:
            stats["records_with_existing_ac"] += 1
            if total_ac > 0:
                stats["records_with_both"] += 1
    
    # Save processed data
    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return stats


def main():
    """Main entry point."""
    
    # Process the main escalated payload file
    input_file = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/escalated_cases_payload.json")
    output_file = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/escalated_cases_payload_with_extracted_ac.json")
    
    print("=" * 60)
    print("Processing escalated_cases_payload.json")
    print("=" * 60)
    
    stats = process_payload_file(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("Processing Statistics:")
    print("=" * 60)
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Records with extracted AC: {stats['records_with_extracted_ac']:,} ({100*stats['records_with_extracted_ac']/stats['total_records']:.1f}%)")
    print(f"  Total extracted amount: ${stats['total_extracted_amount']:,.2f}")
    print(f"  Records with existing Parsed_AC: {stats['records_with_existing_ac']:,}")
    print(f"  Records with both: {stats['records_with_both']:,}")
    
    # Also process sampled file if it exists
    sampled_input = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/sampled_payload.json")
    sampled_output = Path("/Users/shekhar.tanwar/Documents/Projects/NegotiatonAgent/dataset/escalated_payload/sampled_payload_with_extracted_ac.json")
    
    if sampled_input.exists():
        print("\n" + "=" * 60)
        print("Processing sampled_payload.json")
        print("=" * 60)
        
        stats2 = process_payload_file(sampled_input, sampled_output)
        
        print("\n" + "=" * 60)
        print("Sampled Dataset Statistics:")
        print("=" * 60)
        print(f"  Total records: {stats2['total_records']:,}")
        print(f"  Records with extracted AC: {stats2['records_with_extracted_ac']:,}")
        print(f"  Total extracted amount: ${stats2['total_extracted_amount']:,.2f}")
    
    # Show some examples
    print("\n" + "=" * 60)
    print("Sample Extractions (first 5 with credits):")
    print("=" * 60)
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    shown = 0
    for record in data:
        if record.get('Extracted_AC', 0) > 0 and shown < 5:
            print(f"\n  DELIVERY_ID: {record.get('DELIVERY_ID')}")
            print(f"  Existing Parsed_AC: ${record.get('Parsed_AC', 0)}")
            print(f"  Extracted_AC: ${record.get('Extracted_AC', 0)}")
            if record.get('Extracted_AC_Details'):
                print(f"  Details:")
                for detail in record['Extracted_AC_Details'][:3]:
                    print(f"    - ${detail['amount']} ({detail['pattern']})")
                    print(f"      Context: ...{detail['context'][:80]}...")
            shown += 1
    
    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()

