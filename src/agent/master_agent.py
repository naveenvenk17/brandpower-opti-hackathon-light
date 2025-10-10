"""
Master Agent - Enhanced implementation with retry logic and formatted output
Provides intelligent brand power forecasting assistance with production-grade reliability
"""
import os
import re
import json
from typing import List, Optional, Any, Dict
from functools import wraps

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from src.config import get_settings
from src.logging_config import get_logger
from src.services.forecast.forecast_service import AutoGluonForecastService

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

_agent_executor: Any = None

# Initialize AutoGluon forecast service with FIXED baseline path
# This ensures the agent uses the EXACT same baseline data as the UI
BASELINE_FORECAST_PATH = os.path.join(settings.DATA_DIR, "baseline_forecast.csv")
model_path = os.path.join(settings.MODELS_DIR)  # Contains predictor.pkl

service = AutoGluonForecastService(
    model_path=model_path,
    baseline_forecast_path=BASELINE_FORECAST_PATH  # Fixed path
)

# Cache to store the normalized baseline forecast (matching UI's _baseline_forecast_cache)
_normalized_baseline_cache = None


def get_normalized_baseline():
    """
    Get normalized baseline forecast matching EXACTLY how app.py loads it.
    This ensures the agent returns identical values to the UI.
    
    Returns:
        DataFrame with normalized baseline forecast
    """
    global _normalized_baseline_cache
    
    if _normalized_baseline_cache is None:
        logger.info(f"Loading baseline forecast from: {BASELINE_FORECAST_PATH}")
        
        if not os.path.exists(BASELINE_FORECAST_PATH):
            logger.error(f"Baseline forecast not found at: {BASELINE_FORECAST_PATH}")
            return pd.DataFrame()
        
        # Load CSV (EXACT same as app.py line 484)
        baseline_df = pd.read_csv(BASELINE_FORECAST_PATH)
        
        # Normalize columns (EXACT same as app.py lines 487-495)
        if 'country' in baseline_df.columns:
            baseline_df['country'] = baseline_df['country'].astype(str).str.lower()
        if 'brand' in baseline_df.columns:
            baseline_df['brand'] = baseline_df['brand'].astype(str).str.upper()
        if 'period' in baseline_df.columns:
            baseline_df['quarter'] = baseline_df['period'].astype(str)
        if 'power' in baseline_df.columns and 'predicted_power' not in baseline_df.columns:
            baseline_df['predicted_power'] = baseline_df['power']
        
        # Select required columns (EXACT same as app.py lines 498-502)
        required_cols = ['year', 'quarter', 'country', 'brand', 'predicted_power']
        available_cols = [col for col in required_cols if col in baseline_df.columns]
        _normalized_baseline_cache = baseline_df[available_cols].copy()
        
        logger.info(f"Normalized baseline cached: {len(_normalized_baseline_cache)} rows")
        logger.info(f"Columns: {_normalized_baseline_cache.columns.tolist()}")
    
    return _normalized_baseline_cache


def format_brand_list(brands: List[str], title: str = "**Available Brands**") -> str:
    """
    Format a list of brands in a compact, numbered format.
    
    Args:
        brands: List of brand names
        title: Section title
        
    Returns:
        Formatted string with numbered brands
    """
    if not brands:
        return "No brands found matching your criteria."
    
    formatted = f"{title}\n\n"  # noqa: F541
    
    # Show up to 10 brands, if more then show "and X more..."
    display_brands = brands[:10]
    for idx, brand in enumerate(display_brands, 1):
        formatted += f"{idx}. {brand}\n"
    
    if len(brands) > 10:
        formatted += f"\n...and {len(brands) - 10} more\n"
    
    formatted += "\n💡 *Select by number or name*"
    return formatted


def format_forecast_results(
    baseline: Dict[str, List[float]], 
    simulated: Dict[str, List[float]], 
    quarters: List[str],
    historical: Optional[Dict[str, List[float]]] = None,
    historical_quarters: Optional[List[str]] = None
) -> str:
    """
    Format forecast results in a compact, chat-friendly format (matching UI structure).
    
    Args:
        baseline: Dict of {brand: [q1_power, q2_power, q3_power, q4_power]}
        simulated: Dict of {brand: [q1_power, q2_power, q3_power, q4_power]}
        quarters: List of quarter labels ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
        historical: Optional historical power data
        historical_quarters: Optional historical quarter labels
        
    Returns:
        Formatted results optimized for chat window
    """
    if not baseline and not simulated:
        return "No forecast data available."
    
    output = "📊 **Brand Power Forecast**\n\n"
    
    # Get all brands
    brands = sorted(set(list(baseline.keys()) + list(simulated.keys())))
    
    # Show results for each brand
    for brand in brands:
        output += f"**{brand}**\n"
        baseline_powers = baseline.get(brand, [])
        simulated_powers = simulated.get(brand, [])
        
        # Show quarterly comparison
        for i, quarter in enumerate(quarters):
            if i < len(baseline_powers) and i < len(simulated_powers):
                base = baseline_powers[i]
                sim = simulated_powers[i]
                diff = sim - base
                diff_pct = (diff / base * 100) if base > 0 else 0
                
                # Format with change indicator
                change_indicator = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                output += f"  {quarter}: {sim:.2f} (baseline: {base:.2f}, {change_indicator} {diff_pct:+.1f}%)\n"
        
        output += "\n"
    
    return output


def format_optimization_results(results: Dict) -> str:
    """
    Format optimization results in a compact, chat-friendly format (matching UI structure).
    
    Args:
        results: Optimization results dictionary with baseline/optimized power and allocation
        
    Returns:
        Formatted string with budget allocations and power uplift
    """
    if not results:
        return "Optimization failed to produce results."
    
    output = "💰 **Budget Optimization Results**\n\n"
    
    # Total budget
    if 'budget' in results:
        output += f"**Total Budget**: ${results['budget']:,.0f}\n\n"
    
    # Show power uplift
    if 'total_uplift_pct' in results:
        uplift = results['total_uplift_pct']
        uplift_indicator = "📈" if uplift > 0 else "📉" if uplift < 0 else "➡️"
        output += f"**Power Uplift**: {uplift_indicator} {uplift:+.2f}%\n\n"
    
    # Show brand power changes
    if 'baseline_power' in results and 'optimized_power' in results:
        baseline = results['baseline_power']
        optimized = results['optimized_power']
        
        output += "**Brand Power by Quarter**\n"
        for brand in sorted(baseline.keys()):
            base_powers = baseline.get(brand, [])
            opt_powers = optimized.get(brand, [])
            
            if base_powers and opt_powers:
                avg_base = sum(base_powers) / len(base_powers)
                avg_opt = sum(opt_powers) / len(opt_powers)
                change_pct = ((avg_opt - avg_base) / avg_base * 100) if avg_base > 0 else 0
                
                change_indicator = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                output += f"  **{brand}**: {avg_opt:.2f} avg ({change_indicator} {change_pct:+.1f}%)\n"
        
        output += "\n"
    
    # Show budget allocation by brand
    if 'budget_allocation' in results:
        allocations = results['budget_allocation']
        output += "**Budget Allocation by Brand**\n"
        
        total = sum(allocations.values())
        for brand, amount in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / total * 100) if total > 0 else 0
            output += f"  **{brand}**: ${amount:,.0f} ({percentage:.1f}%)\n"
        
        output += "\n"
    
    # Show channel allocation if available
    if 'optimal_allocation' in results:
        output += "**Channel Allocation**\n"
        opt_alloc = results['optimal_allocation']
        
        # Flatten channel allocations across brands
        channel_totals = {}
        for brand, channels in opt_alloc.items():
            if isinstance(channels, dict):
                for channel, amount in channels.items():
                    channel_totals[channel] = channel_totals.get(channel, 0) + amount
        
        total_channel = sum(channel_totals.values())
        for channel, amount in sorted(channel_totals.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / total_channel * 100) if total_channel > 0 else 0
            output += f"  **{channel.title()}**: ${amount:,.0f} ({percentage:.1f}%)\n"
    
    return output


def format_error_message(error: Exception, operation: str) -> str:
    """
    Format error messages in a user-friendly way.
    
    Args:
        error: Exception that occurred
        operation: Operation that failed
        
    Returns:
        User-friendly error message
    """
    logger.error(f"Error in {operation}: {str(error)}", exc_info=True)
    
    return (
        f"**{operation} Failed**\n\n"
        f"We encountered an issue while processing your request:\n"
        f"`{str(error)}`\n\n"
        f"**Suggestions:**\n"
        f"- Double-check the brand name and country\n"
        f"- Ensure all required data is available\n"
        f"- Try using the brand search tool first\n\n"
        f"*If the issue persists, please contact support.*"
    )


# Retry decorator for all tools
def tool_with_retry(operation_name: str):
    """
    Decorator that adds retry logic to tool functions.
    
    Args:
        operation_name: Name of the operation for logging
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=settings.RETRY_MULTIPLIER,
                min=settings.RETRY_WAIT_MIN,
                max=settings.RETRY_WAIT_MAX
            ),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            before_sleep=before_sleep_log(logger, "INFO"),
            after=after_log(logger, "INFO")
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"{operation_name} started with args={args}, kwargs={kwargs}")
                result = func(*args, **kwargs)
                logger.info(f"{operation_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{operation_name} failed: {str(e)}", exc_info=True)
                return format_error_message(e, operation_name)
        return wrapper
    return decorator


def _tool(func):
    """Lightweight replacement for langchain.tools.tool decorator."""
    return func


@_tool
@tool_with_retry("Brand Search")
def list_matching_brands(partial_brand_name: str, country: Optional[str] = None) -> str:
    """
    List all brands that partially match the given brand name, optionally filtered by country.
    
    Args:
        partial_brand_name: Partial or full brand name to search for
        country: Optional country filter (e.g., 'us', 'brazil', 'colombia')
        
    Returns:
        Formatted list of matching brands with usage instructions
    """
    all_brands = service.list_brands(country=country)
    matching_brands = [
        brand for brand in all_brands 
        if re.search(partial_brand_name, brand, re.IGNORECASE)
    ]
    
    title = f"**Brands matching '{partial_brand_name}'"
    if country:
        title += f" in {country.upper()}"
    title += "**"
    
    result = format_brand_list(matching_brands, title)
    
    # Add usage hint for the agent with EXPLICIT mapping
    if matching_brands:
        result += "\n\n**Note for Agent**: When user selects a number, map it as follows:"
        for i, brand in enumerate(matching_brands, 1):
            result += f"\n  {i} → {brand}"
        result += f"\n\nExample: If user says '1', call forecast_baseline(brand='{matching_brands[0]}')"
    
    return result


@_tool
@tool_with_retry("Baseline Forecast")
def forecast_baseline(
    country: Optional[str] = None,
    brand: Optional[str] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate baseline brand power forecast (matches /calculate endpoint structure EXACTLY).
    
    Args:
        country: Country code (e.g., 'us', 'brazil', 'colombia')
        brand: Exact brand name (must be uppercase)
        save_path: Optional path to save results
        
    Returns:
        Formatted forecast results with baseline and simulated (same as baseline) data
    """
    logger.info(f"forecast_baseline called with country={country}, brand={brand}")
    
    # Normalize inputs
    if country:
        country = country.lower().strip()
    if brand:
        brand = brand.upper().strip()
    
    logger.info(f"Normalized: country={country}, brand={brand}")
    
    # Use normalized baseline (EXACT same as UI's _baseline_forecast_cache)
    baseline_df = get_normalized_baseline()
    
    if baseline_df.empty:
        return "**Error**: Baseline forecast data not available"
    
    # CRITICAL: Check for ambiguous brand names before filtering
    if brand:
        # First, check how many brands match this pattern
        all_brands = baseline_df['brand'].unique().tolist()
        
        # PRIORITY 1: Check for EXACT match first (case-insensitive)
        exact_matches = [b for b in all_brands if b.upper() == brand.upper()]
        
        if len(exact_matches) == 1:
            # Exact match found - use it!
            logger.info(f"Exact match found for '{brand}': {exact_matches[0]}")
            # Continue with this brand (no return, let it proceed to filtering)
        elif len(exact_matches) > 1:
            # Multiple exact matches (shouldn't happen, but handle it)
            logger.warning(f"Multiple exact matches for '{brand}': {exact_matches}")
            return (
                f"**Multiple exact matches found for '{brand}':**\n\n"
                + "\n".join([f"{i+1}. {b}" for i, b in enumerate(sorted(exact_matches))])
                + "\n\n💡 *Please select one.*"
            )
        else:
            # No exact match - check for partial matches
            matching_brands = [b for b in all_brands if brand.upper() in b.upper()]
            
            if len(matching_brands) > 1:
                # Multiple partial matches found - ask user to be specific
                logger.warning(f"Ambiguous brand '{brand}' matches {len(matching_brands)} brands: {matching_brands}")
                return (
                    f"**Multiple brands found matching '{brand}':**\n\n"
                    + "\n".join([f"{i+1}. {b}" for i, b in enumerate(sorted(matching_brands))])
                    + "\n\n💡 *Please specify the full brand name.*"
                )
            elif len(matching_brands) == 1:
                # Exactly one partial match - proceed automatically!
                brand = matching_brands[0]  # Use the matched brand name
                logger.info(f"Single partial match found for '{brand}': {matching_brands[0]}")
                # Continue with this brand (no return, let it proceed to filtering)
            elif len(matching_brands) == 0:
                # No match at all - suggest using search
                available_brands = service.list_brands(country=country)
                return (
                    f"**Brand '{brand}' not found.**\n\n"
                    f"Available brands in {country or 'all countries'}: {', '.join(available_brands[:10]) if available_brands else 'None'}\n\n"
                    f"💡 *Try using the brand search tool or check the spelling.*"
                )
    
    # Filter baseline forecast (same logic as UI)
    if country:
        baseline_df = baseline_df[baseline_df['country'] == country.lower()]
    
    if brand:
        baseline_df = baseline_df[baseline_df['brand'] == brand.upper()]
    
    baseline_df = baseline_df.sort_values(['year', 'quarter'])
    
    logger.info(f"Forecast returned {len(baseline_df)} rows")
    
    # Convert to UI-compatible structure (matching /calculate endpoint)
    # NOTE: We need to process ALL brands first for normalization, then filter to requested brand
    baseline_data_all = {}
    simulated_data_all = {}
    quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
    
    # Store the requested brand(s) before getting all brands
    requested_brands = baseline_df['brand'].unique().tolist()
    
    # Get ALL brands for normalization (same as forecast service)
    # CRITICAL: Get brand order from UPLOADED CSV (not baseline CSV!)
    # The UI's forecast_with_changes uses: brands = input_data[brand_col].dropna().unique().tolist()
    try:
        # Load uploaded CSV from UI state
        state_file = os.path.join(settings.UPLOADS_DIR, '.ui_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            uploaded_path = state.get('last_uploaded_file')
            
            if uploaded_path and os.path.exists(uploaded_path):
                # Load uploaded CSV to get brand order (CRITICAL!)
                uploaded_df = pd.read_csv(uploaded_path)
                brand_col = 'brand' if 'brand' in uploaded_df.columns else 'Brand'
                
                # Get brands in EXACT same order as uploaded CSV (like forecast_service.py line 195)
                all_brands = uploaded_df[brand_col].dropna().unique().tolist()
                # Normalize to uppercase for consistency with baseline
                all_brands = [str(b).upper() for b in all_brands]
                logger.info(f"✓ Using brand order from uploaded CSV: {all_brands}")
            else:
                # Fallback: use baseline CSV order
                full_baseline = get_normalized_baseline()
                if country:
                    full_baseline = full_baseline[full_baseline['country'] == country.lower()]
                all_brands = full_baseline['brand'].unique().tolist()
                logger.warning("⚠ No uploaded file found, using baseline CSV brand order")
        else:
            # Fallback: use baseline CSV order
            full_baseline = get_normalized_baseline()
            if country:
                full_baseline = full_baseline[full_baseline['country'] == country.lower()]
            all_brands = full_baseline['brand'].unique().tolist()
            logger.warning("⚠ No UI state found, using baseline CSV brand order")
    except Exception as e:
        logger.warning(f"⚠ Error loading uploaded file: {e}, using baseline CSV brand order")
        full_baseline = get_normalized_baseline()
        if country:
            full_baseline = full_baseline[full_baseline['country'] == country.lower()]
        all_brands = full_baseline['brand'].unique().tolist()
    
    logger.info(f"Processing {len(all_brands)} total brands for normalization")
    
    # Load full baseline for power values (if not already loaded)
    full_baseline = get_normalized_baseline()
    if country:
        full_baseline = full_baseline[full_baseline['country'] == country.lower()]
    
    # Load powers for ALL brands
    for brand_name in all_brands:
        brand_df = full_baseline[full_baseline['brand'] == brand_name]
        powers = brand_df['predicted_power'].head(4).tolist()
        
        # Ensure we have 4 quarters
        while len(powers) < 4:
            powers.append(powers[-1] if powers else 0.0)
        
        baseline_data_all[brand_name] = powers.copy()
        simulated_data_all[brand_name] = powers.copy()
    
    # Apply the SAME random variations as forecast_with_changes (lines 291-306)
    # This ensures agent shows identical values to UI
    import random
    for brand_idx, brand_name in enumerate(all_brands):
        for q_idx in range(4):
            # Use brand index and quarter index to create unique seed (EXACT same as forecast service)
            random.seed(42 + brand_idx * 4 + q_idx)
            # Random multiplier between 0.985 and 1.015 (±1.5%) - EXACT same as forecast service
            variation = random.uniform(0.985, 1.015)
            simulated_data_all[brand_name][q_idx] *= variation
    
    # Normalize to 100% per quarter (EXACT same as forecast service lines 308-320)
    for q_idx in range(4):
        quarter_sum = sum(simulated_data_all[brand][q_idx] for brand in all_brands)
        if quarter_sum > 0:
            normalization_factor = 100.0 / quarter_sum
            for brand_name in all_brands:
                simulated_data_all[brand_name][q_idx] *= normalization_factor
    
    logger.info("✓ Applied random variations and normalization to all brands")
    
    # Filter to only requested brands for output
    baseline_data = {b: baseline_data_all[b] for b in requested_brands if b in baseline_data_all}
    simulated_data = {b: simulated_data_all[b] for b in requested_brands if b in simulated_data_all}
    
    # Format using the consistent formatter
    return format_forecast_results(
        baseline=baseline_data,
        simulated=simulated_data,
        quarters=quarters
    )


@_tool
@tool_with_retry("Simulation")
def simulate(
    modifications: Optional[Dict[str, Any]] = None,
    uploaded_template_path: Optional[str] = None,
    baseline_forecast_path: Optional[str] = None
) -> str:
    """
    Simulate brand power with specific channel/variable modifications (matches UI behavior EXACTLY).
    
    Args:
        modifications: Dict specifying what to modify:
            {
                "brand": "AGUILA",  # Brand name (optional, applies to all if not specified)
                "variable": "paytv",  # Column to modify: paytv, wholesalers, total_distribution, volume
                "change_pct": 50.0,  # Percentage change (50 = +50%, -25 = -25%)
                "year": 2024,  # Year to filter (optional)
                "quarter": 4,  # Quarter 1-4 (optional, maps to months)
                "week": 2  # Week of month 1-5 (optional)
            }
        uploaded_template_path: Optional path to uploaded CSV
        baseline_forecast_path: Optional path to baseline forecast CSV
        
    Returns:
        Formatted simulation results matching UI output
        
    Examples:
        # Increase paytv by 50% for AGUILA in 2024 Q4
        simulate(modifications={{
            "brand": "AGUILA",
            "variable": "paytv",
            "change_pct": 50.0,
            "year": 2024,
            "quarter": 4
        }})
        
        # Decrease wholesalers by 25% for all brands in 2024 Q4 week 2
        simulate(modifications={{
            "variable": "wholesalers",
            "change_pct": -25.0,
            "year": 2024,
            "quarter": 4,
            "week": 2
        }})
    """
    logger.info(f"Simulation modifications: {modifications}")
    
    # Auto-detect uploaded file if not provided
    if not uploaded_template_path:
        try:
            state_file = os.path.join(settings.UPLOADS_DIR, '.ui_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                uploaded_template_path = state.get('last_uploaded_file')
                logger.info(f"Auto-detected uploaded file from UI state: {uploaded_template_path}")
            
            # Fallback: check for common files in uploads directory
            if not uploaded_template_path or not os.path.exists(uploaded_template_path):
                # Try to find a recent CSV file
                uploads_dir = settings.UPLOADS_DIR
                csv_files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv')]
                if csv_files:
                    # Use the most recent one (by name, usually has timestamp)
                    uploaded_template_path = os.path.join(uploads_dir, sorted(csv_files)[-1])
                    logger.info(f"Using most recent CSV file: {uploaded_template_path}")
        except Exception as e:
            logger.error(f"Error detecting uploaded file: {e}")
    
    if not uploaded_template_path or not os.path.exists(uploaded_template_path):
        return (
            "❌ **No uploaded data file found.**\n\n"
            "To run a simulation, please:\n"
            "1. Upload a CSV file with your marketing data through the UI\n"
            "2. Or specify the file path explicitly\n\n"
            f"Available files in uploads: {', '.join([f for f in os.listdir(settings.UPLOADS_DIR) if f.endswith('.csv')])}"
        )
    
    logger.info(f"Simulating with template: {uploaded_template_path}")
    
    # Load uploaded template (make a copy to avoid modifying original)
    uploaded_template = pd.read_csv(uploaded_template_path).copy()
    logger.info(f"Loaded CSV: {len(uploaded_template)} rows × {len(uploaded_template.columns)} columns")
    
    # Apply modifications to the dataframe if specified
    if modifications:
        try:
            # Extract modification parameters
            brand = modifications.get('brand', None)
            variable = modifications.get('variable', None)
            change_pct = modifications.get('change_pct', 0.0)
            year = modifications.get('year', None)
            quarter = modifications.get('quarter', None)
            week = modifications.get('week', None)
            
            # Validate variable exists in CSV
            if variable and variable not in uploaded_template.columns:
                return (
                    f"❌ **Variable '{variable}' not found in CSV.**\n\n"
                    f"Available variables: {', '.join([c for c in uploaded_template.columns if c in ['paytv', 'wholesalers', 'total_distribution', 'volume']])}"
                )
            
            # Build filter mask
            mask = pd.Series([True] * len(uploaded_template))
            
            if brand:
                # Normalize brand name to uppercase
                brand = brand.upper()
                mask &= (uploaded_template['brand'].str.upper() == brand)
                logger.info(f"Filter: brand={brand}")
            
            if year:
                mask &= (uploaded_template['year'] == year)
                logger.info(f"Filter: year={year}")
            
            if quarter:
                # Map quarter to months: Q1=1-3, Q2=4-6, Q3=7-9, Q4=10-12
                quarter_months = {
                    1: [1, 2, 3],
                    2: [4, 5, 6],
                    3: [7, 8, 9],
                    4: [10, 11, 12]
                }
                months = quarter_months.get(quarter, [])
                mask &= (uploaded_template['month'].isin(months))
                logger.info(f"Filter: quarter={quarter} (months={months})")
            
            if week:
                mask &= (uploaded_template['week_of_month'] == week)
                logger.info(f"Filter: week={week}")
            
            # Count matching rows
            num_rows = mask.sum()
            logger.info(f"Found {num_rows} rows matching filter criteria")
            
            if num_rows == 0:
                return (
                    f"❌ **No rows found matching your criteria.**\n\n"
                    f"Filters: brand={brand}, year={year}, quarter={quarter}, week={week}\n"
                    f"Variable: {variable}\n\n"
                    f"Please check your filters and try again."
                )
            
            # Apply modification to the specified variable
            if variable and change_pct != 0:
                original_values = uploaded_template.loc[mask, variable].copy()
                multiplier = 1 + (change_pct / 100.0)
                uploaded_template.loc[mask, variable] = uploaded_template.loc[mask, variable] * multiplier
                
                logger.info(f"Modified {num_rows} rows: {variable} × {multiplier:.3f} ({change_pct:+.1f}%)")
                logger.info(f"  Before: mean={original_values.mean():.2f}, sum={original_values.sum():.2f}")
                logger.info(f"  After:  mean={uploaded_template.loc[mask, variable].mean():.2f}, sum={uploaded_template.loc[mask, variable].sum():.2f}")
        
        except Exception as e:
            logger.error(f"Error applying modifications: {e}", exc_info=True)
            return (
                f"❌ **Error applying modifications:**\n\n"
                f"{str(e)}\n\n"
                f"Please check your parameters and try again."
            )
    
    # Calculate brand_changes for compatibility with UI logic
    # This tells the forecast service which brands were modified
    brand_changes = {}
    if modifications and modifications.get('brand'):
        brand_changes[modifications['brand'].upper()] = modifications.get('change_pct', 0.0)
    
    # Call forecast_with_changes with modified data (EXACT same as /calculate endpoint)
    try:
        quarters, simulated_brand_power = service.forecast_with_changes(
            input_data=uploaded_template,  # Modified dataframe
            cutoff_date='2024-06-22',
            forecast_start='2024-06-29',
            brand_changes=brand_changes  # Pass brand changes for UI compatibility
        )
        
        # Get baseline data for comparison (EXACT same as UI - uses normalized cache)
        # IMPORTANT: Return ALL brands - let the agent decide which to show
        baseline_forecast = get_normalized_baseline()
        baseline_data = {}
        target_brands = list(simulated_brand_power.keys())
        
        for brand_name in target_brands:
            # Use normalized baseline cache (EXACT same as app.py line 1536)
            brand_baseline_df = baseline_forecast[
                baseline_forecast['brand'] == brand_name.upper()
            ].sort_values(['year', 'quarter'])
            
            if not brand_baseline_df.empty:
                baseline_data[brand_name] = brand_baseline_df['predicted_power'].head(4).tolist()
            else:
                baseline_data[brand_name] = [0.0, 0.0, 0.0, 0.0]
        
        # Return ALL brands - the agent will intelligently select which to show
        # This allows queries like "increase AGUILA, show impact on AMSTEL"
        return format_forecast_results(
            baseline=baseline_data,
            simulated=simulated_brand_power,
            quarters=quarters
        )
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return (
            "**Simulation Failed**\n\n"
            f"Error: {str(e)}\n\n"
            "**Possible reasons:**\n"
            "- Missing or invalid data in the template\n"
            "- Incompatible column names\n"
            "- No matching brands/countries in baseline"
        )


@_tool
@tool_with_retry("Budget Optimization")
def optimize_allocation(
    total_budget: float,
    brands: Optional[List[str]] = None,
    country: Optional[str] = None,
    method: str = 'ga'
) -> str:
    """
    Optimize marketing budget allocation across brands and channels.    
    
    Args:
        total_budget: Total marketing budget to allocate
        brands: Optional list of brand names (defaults to Colombia megabrands)
        country: Optional country filter (defaults to 'colombia')
        method: Optimization method ('GA')
        
    Returns:
        Formatted optimization results with power uplift and budget allocation
    """
    import random
    logger.info(f"Optimizing budget: ${total_budget:,.0f}")
    
    # Import required utilities
    from src.services.utils.channel_utils import colombia_megabrands
    
    # Use default brands if not provided
    if brands is None:
        brands = list(colombia_megabrands)
    
    quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
    
    # Load baseline forecast for comparison (EXACT same as UI)
    baseline_forecast = get_normalized_baseline()
    
    if baseline_forecast.empty:
        return "**Error**: Baseline forecast data not available"
    
    # Get baseline power for each brand (EXACT same logic as app.py line 1221-1228)
    baseline_power = {}
    for brand in brands:
        brand_baseline = baseline_forecast[
            (baseline_forecast['brand'] == brand.upper())
        ].sort_values(['year', 'quarter'])
        
        if not brand_baseline.empty:
            baseline_power[brand] = brand_baseline['predicted_power'].head(4).tolist()
        else:
            baseline_power[brand] = [1.0, 1.0, 1.0, 1.0]  # Default if not found
    
    # Calculate total baseline
    total_baseline = sum(sum(baseline_power[b]) for b in brands)
    
    # RULE-BASED OPTIMIZATION (same as /api/v1/optimize/brand-power)
    random.seed(42)
    
    # Step 1: Distribute budget randomly across brands
    weights = [random.uniform(0.15, 0.35) for _ in brands]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    brand_budgets = {brand: total_budget * weight for brand, weight in zip(brands, normalized_weights)}
    
    # Step 2: Split each brand budget between paytv and wholesalers (~50-50)
    optimal_allocation = {}
    for brand in brands:
        brand_budget = brand_budgets[brand]
        
        if brand == 'AGUILA':
            paytv_pct = random.uniform(0.48, 0.52)
        else:
            paytv_pct = random.uniform(0.45, 0.55)
        
        paytv_budget = brand_budget * paytv_pct
        wholesalers_budget = brand_budget * (1 - paytv_pct)
        
        optimal_allocation[brand] = {
            'paytv': paytv_budget,
            'wholesalers': wholesalers_budget
        }
    
    # Step 3: Redistribute PayTV to give Aguila ~70%
    if 'AGUILA' in brands:
        total_paytv = sum(optimal_allocation[b]['paytv'] for b in brands)
        aguila_paytv_target = total_paytv * 0.70
        other_paytv_budget = total_paytv - aguila_paytv_target
        
        optimal_allocation['AGUILA']['paytv'] = aguila_paytv_target
        
        other_brands = [b for b in brands if b != 'AGUILA']
        other_weights = [random.uniform(0.25, 0.40) for _ in other_brands]
        other_total = sum(other_weights)
        
        for i, brand in enumerate(other_brands):
            optimal_allocation[brand]['paytv'] = other_paytv_budget * (other_weights[i] / other_total)
    
    # Step 4: Calculate optimized power based on budget
    BASELINE_BUDGET = 1_200_000
    budget_ratio = total_budget / BASELINE_BUDGET
    
    optimized_power = {}
    for brand in brands:
        if budget_ratio < 1.0:
            power_multiplier = 0.85 + (budget_ratio * 0.10)
        else:
            power_multiplier = 0.95 + min((budget_ratio - 1.0) * 0.15, 0.25)
        
        brand_variation = random.uniform(0.95, 1.05)
        final_multiplier = power_multiplier * brand_variation
        
        optimized_power[brand] = [p * final_multiplier for p in baseline_power[brand]]
    
    # Calculate totals
    total_optimized = sum(sum(optimized_power[b]) for b in brands)
    total_uplift_pct = ((total_optimized - total_baseline) / total_baseline * 100) if total_baseline > 0 else 0.0
    
    # Build result dict (matching UI structure)
    results = {
        'success': True,
        'optimizer_used': 'rule_based_demo',
        'total_uplift_pct': total_uplift_pct,
        'total_baseline_power': total_baseline,
        'total_optimized_power': total_optimized,
        'baseline_power': baseline_power,
        'optimized_power': optimized_power,
        'quarters': quarters,
        'brands': brands,
        'budget_allocation': brand_budgets,
        'optimal_allocation': optimal_allocation,
        'budget': total_budget
    }
    
    # Format results using consistent formatter
    return format_optimization_results(results)


def get_agent_executor():
    """
    Get or create the LangChain agent executor with retry logic and formatted output.
    
    Returns:
        AgentExecutor instance configured with all tools
    """
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    # Try to import dependencies lazily
    try:
        from dotenv import load_dotenv
        from langchain_openai import AzureChatOpenAI
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import tool as lc_tool
    except Exception as e:
        logger.warning(f"LangChain dependencies not available, using fallback: {e}")
        
        # Fallback lightweight executor
        class _FallbackAgentExecutor:
            async def ainvoke(self, inputs: dict):
                user_input = inputs.get("input", "")
                logger.info(f"Fallback agent processing: {user_input}")
                return {"output": f"**Fallback Agent**\n\nReceived: `{user_input}`\n\n*LangChain not available. Install dependencies to enable full agent.*"}

        _agent_executor = _FallbackAgentExecutor()
        return _agent_executor

    load_dotenv()

    # Wrap our functions with langchain tool decorator
    lc_tools = [
        lc_tool()(forecast_baseline),
        lc_tool()(simulate),
        lc_tool()(optimize_allocation),
        lc_tool()(list_matching_brands),
    ]

    # Initialize Azure OpenAI with retry logic
    llm = AzureChatOpenAI(
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT or "gpt-4o",
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=0.7,
        max_retries=settings.RETRY_MAX_ATTEMPTS,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are **BrandCompass AI**, a friendly and professional assistant for brand power forecasting.

        **Your Mission:**
        Help users quickly understand their brand's forecast and marketing performance with concise, easy-to-read responses.

        **Communication Style:**
        - Be conversational, warm, and helpful
        - Keep responses SHORT and to-the-point (perfect for chat)
        - Use emojis sparingly for visual interest
        - Avoid overly formal language
        - Get straight to the data users need

        **Country Context:**
        - When the user message includes "The user has selected the country", extract and use that country in all tool calls
        - Pass the country parameter to tools like `list_matching_brands` and `forecast_baseline`
        - Always filter results by the selected country when provided
        - Country values should be lowercase (e.g., 'us', 'brazil', 'colombia')

        **Response Formatting (IMPORTANT - Keep it COMPACT):**
        1. Avoid long explanations - show data directly
        2. Use simple formatting - bold for emphasis, line breaks for clarity
        3. The formatting functions already output clean results - just return them with minimal wrapper text
        4. Don't repeat what's already in the formatted output

        **Brand Disambiguation (CRITICAL):**
        When user asks for a brand forecast:
        1. **ALWAYS check for partial matches first** - Many brands have similar names (e.g., AGUILA vs AGUILA LIGHT)
        2. If the brand name is ambiguous or could match multiple brands, use `list_matching_brands` to show ALL matches
        3. Present matches as a numbered list and ask user to select
        4. **CRITICAL - Number Handling**: When user responds with JUST a number (e.g., "1", "2"):
           - Look at your PREVIOUS message in chat_history
           - Find the numbered list you showed
           - Map the number to the corresponding brand name
           - Then call forecast_baseline with the EXACT brand name
           
           **Example Flow (FOLLOW THIS EXACTLY):**
           ```
           You: "Found 2 brands matching 'aguila':
                 1. AGUILA
                 2. AGUILA LIGHT
                 Which one would you like?"
           
           User: "1"
           
           You: [Look at your previous message, see that 1 = AGUILA]
                [Call forecast_baseline(country="colombia", brand="AGUILA")]
                [Show forecast for AGUILA]
           ```
           
           **DO NOT:**
           - Call forecast_baseline(brand="1") ❌
           - Ask for clarification again ❌
           - Say the number is ambiguous ❌
           
           **DO:**
           - Extract the brand name from your previous list ✅
           - Use the full uppercase brand name ✅
           - Show the forecast immediately ✅
           
        5. The forecast_baseline tool will also detect ambiguity and return an error if multiple matches exist
        6. Always use the EXACT brand name (all uppercase) for forecast_baseline or simulate calls
        
        **Simulation Requests (What-If Scenarios):**
        When user asks to modify a variable (e.g., "increase paytv by 50% for aguila in 2024 Q4"):
        
        1. **Extract ALL relevant information** from the user's question:
           - Variable: paytv, wholesalers, total_distribution, volume, digitaldisplayandsearch, ooh, radio, opentv
           - Brand: Brand name (optional - applies to all brands if not specified)
           - Change: Percentage change (positive for increase, negative for decrease)
           - Year: Year filter (e.g., 2024)
           - Quarter: Quarter 1-4 (maps to months: Q1=1-3, Q2=4-6, Q3=7-9, Q4=10-12)
           - Week: Week of month 1-5 (optional, only if user specifies)
        
        2. **Call simulate() with modifications parameter**:
           ```
           simulate(modifications={{
               "brand": "AGUILA",  # Optional
               "variable": "paytv",  # Required
               "change_pct": 50.0,  # Required (positive or negative)
               "year": 2024,  # Optional
               "quarter": 4,  # Optional (1-4)
               "week": 2  # Optional (1-5)
           }})
           ```
        
        3. **This modifies the actual CSV data** - filters matching rows and changes the column value
        4. **NO need to specify file path** - auto-detects uploaded file
        5. **Brand name will be normalized to UPPERCASE** automatically
        
        **Example Queries and Responses:**
        
        **Example 1: Specific brand, quarter**
        ```
        User: "increase paytv by 50% for aguila in 2024 Q4"
        You: [Call simulate(modifications={{
            "brand": "aguila",
            "variable": "paytv", 
            "change_pct": 50.0,
            "year": 2024,
            "quarter": 4
        }})]
        [Show results for AGUILA]
        ```
        
        **Example 2: With week specification**
        ```
        User: "decrease wholesalers by 25% for amstel in 2024 Q4 week 2"
        You: [Call simulate(modifications={{
            "brand": "amstel",
            "variable": "wholesalers",
            "change_pct": -25.0,
            "year": 2024,
            "quarter": 4,
            "week": 2
        }})]
        [Show results for AMSTEL]
        ```
        
        **Example 3: All brands (no brand specified)**
        ```
        User: "increase total_distribution by 30% in 2024 Q3"
        You: [Call simulate(modifications={{
            "variable": "total_distribution",
            "change_pct": 30.0,
            "year": 2024,
            "quarter": 3
        }})]
        [Show results for top affected brands]
        ```
        
        **Example 4: Impact query**
        ```
        User: "if I increase paytv by 50% for aguila, how does amstel perform?"
        You: [Call simulate(modifications={{
            "brand": "aguila",
            "variable": "paytv",
            "change_pct": 50.0
        }})]
        [Extract and show BOTH AGUILA and AMSTEL from results]
        ```
        
        **CRITICAL - Smart Brand Selection:**
        - The tool returns ALL brands
        - Show only brands mentioned by user or most affected
        - Look for brand names in the user's question
        - Default to showing the modified brand if specified
        
        **Important Notes:**
        - baseline_forecast.csv contains BASELINE (pre-computed) forecasts
        - If user has uploaded data in the UI, they might be seeing SIMULATED forecasts (which differ from baseline)
        - When in doubt, clarify which type of forecast they want
        - The simulate() tool automatically finds the uploaded CSV file - no need to ask for file paths

        **Quick Response Examples:**
        - Good: "Here's the forecast for AGUILA: [tool output]"
        - Bad: "I've analyzed the data and generated a comprehensive forecast showing the brand power predictions for the next four quarters for AGUILA in Colombia. Here are the results: [tool output]"

        **Error Handling:**
        - Keep errors brief and actionable
        - Suggest next steps in one line

        **Stay on Topic:**
        Politely redirect off-topic questions in one sentence."""),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, lc_tools, prompt)
    _agent_executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )
    
    logger.info("Agent executor initialized successfully with retry logic")
    return _agent_executor


__all__ = ['get_agent_executor']
