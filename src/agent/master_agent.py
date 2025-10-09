"""
Master Agent - Enhanced implementation with retry logic and formatted output
Provides intelligent brand power forecasting assistance with production-grade reliability
"""
import os
import re
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

# Initialize AutoGluon forecast service
model_path = os.path.join(settings.MODELS_DIR)  # Contains predictor.pkl
baseline_csv_path = os.path.join(settings.DATA_DIR, "baseline_forecast.csv")

service = AutoGluonForecastService(
    model_path=model_path,
    baseline_forecast_path=baseline_csv_path
)


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
    
    formatted += f"\n💡 *Select by number or name*"
    return formatted


def format_forecast_results(results: List[Dict], brand: str = None, country: str = None) -> str:
    """
    Format forecast results in a compact, chat-friendly format.
    
    Args:
        results: List of forecast dictionaries
        brand: Brand name (for highlighting)
        country: Country name (for highlighting)
        
    Returns:
        Formatted results optimized for chat window
    """
    if not results:
        return "No forecast data available."
    
    # Build compact header
    header = f"📊 **{brand or 'Brand'}"
    if country:
        header += f" ({country.upper()})"
    header += "**\n\n"
    
    # Convert to DataFrame and sort by period
    df = pd.DataFrame(results)
    
    # Sort by year and quarter
    if 'year' in df.columns and 'quarter' in df.columns:
        df = df.sort_values(['year', 'quarter'])
    
    # Format each quarter's result in compact form
    output = header
    
    for idx, row in df.iterrows():
        # Get the power value
        power = row.get('predicted_power') or row.get('simulated_power') or row.get('power')
        
        if power is not None:
            year = row.get('year', '')
            quarter = row.get('quarter', '')
            
            # Ultra-compact format
            output += f"**{year} {quarter}**: {power:.2f}\n"
    
    # Add compact summary
    if len(results) > 0:
        powers = [r.get('predicted_power') or r.get('simulated_power') or r.get('power', 0) for r in results]
        valid_powers = [p for p in powers if p is not None]
        if valid_powers:
            avg_power = sum(valid_powers) / len(valid_powers)
            min_power = min(valid_powers)
            max_power = max(valid_powers)
            
            output += f"\n📈 Avg: {avg_power:.2f} • Min: {min_power:.2f} • Max: {max_power:.2f}"
    
    return output


def format_optimization_results(results: Dict) -> str:
    """
    Format optimization results in a compact, chat-friendly format.
    
    Args:
        results: Optimization results dictionary
        
    Returns:
        Formatted string with budget allocations
    """
    if not results:
        return "Optimization failed to produce results."
    
    output = "💰 **Budget Optimization**\n\n"
    
    # Total budget
    if 'total_budget' in results:
        output += f"**Total**: ${results['total_budget']:,.0f}\n\n"
    
    # Allocations
    if 'allocation' in results:
        allocations = results['allocation']
        
        # Sort by allocation amount (descending)
        sorted_alloc = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
        
        total = sum(allocations.values())
        for channel, amount in sorted_alloc:
            percentage = (amount / total * 100) if total > 0 else 0
            output += f"**{channel.title()}**: ${amount:,.0f} ({percentage:.1f}%)\n"
    
    # Expected lift/impact
    if 'expected_lift' in results:
        output += f"\n📈 Lift: {results['expected_lift']:.2f}%"
    if 'projected_impact' in results:
        output += f" • Impact: {results['projected_impact']:.2f}"
    
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
    
    # Add usage hint for the agent
    if matching_brands:
        result += "\n\n**Note**: When user selects a number, use the corresponding brand name in subsequent tool calls."
        result += f"\nExample: If user says '1', use brand='{matching_brands[0]}'"
    
    return result


@_tool
@tool_with_retry("Baseline Forecast")
def forecast_baseline(
    country: Optional[str] = None,
    brand: Optional[str] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate baseline brand power forecast.
    
    Args:
        country: Country code (e.g., 'us', 'brazil', 'colombia')
        brand: Exact brand name (must be uppercase)
        save_path: Optional path to save results
        
    Returns:
        Formatted forecast results
    """
    logger.info(f"forecast_baseline called with country={country}, brand={brand}")
    
    # Normalize inputs
    if country:
        country = country.lower().strip()
    if brand:
        brand = brand.upper().strip()
    
    logger.info(f"Normalized: country={country}, brand={brand}")
    
    df = service.forecast_baseline(country, brand, save_path)
    
    logger.info(f"Forecast returned {len(df)} rows")
    
    if df.empty:
        # Provide helpful debugging info
        available_brands = service.list_brands(country=country)
        return (
            "**No Forecast Data Found**\n\n"
            f"No data available for:\n"
            f"- **Brand**: `{brand or 'all brands'}`\n"
            f"- **Country**: `{country or 'all countries'}`\n\n"
            f"**Available brands in {country or 'all countries'}**: {', '.join(available_brands[:10]) if available_brands else 'None'}\n\n"
            f"Try using the brand search tool to find available brands."
        )
    
    results = df.to_dict(orient='records')
    return format_forecast_results(results, brand=brand, country=country)


@_tool
@tool_with_retry("Simulation")
def simulate(uploaded_template_path: str, baseline_forecast_path: Optional[str] = None) -> str:
    """
    Simulate brand power with new marketing allocations from an uploaded template.
    
    Args:
        uploaded_template_path: Path to uploaded CSV template
        baseline_forecast_path: Optional path to baseline forecast CSV
        
    Returns:
        Formatted simulation results
    """
    uploaded_template = pd.read_csv(uploaded_template_path)
    baseline_forecast = pd.read_csv(baseline_forecast_path) if baseline_forecast_path else None
    
    df = service.simulate(uploaded_template, baseline_forecast)
    
    if df.empty:
        return (
            "**Simulation Failed**\n\n"
            "The simulation did not produce any results.\n\n"
            "**Possible reasons:**\n"
            "- Missing or invalid data in the template\n"
            "- Incompatible column names\n"
            "- No matching brands/countries in baseline"
        )
    
    results = df.to_dict(orient='records')
    return "**Simulation Results**\n\n" + format_forecast_results(results)


@_tool
@tool_with_retry("Budget Optimization")
def optimize_allocation(
    total_budget: float,
    channels: Optional[List[str]] = None,
    method: str = 'gradient',
    digital_cap: float = 0.99,
    tv_cap: float = 0.5
) -> str:
    """
    Optimize marketing budget allocation across channels.
    
    Args:
        total_budget: Total marketing budget to allocate
        channels: Optional list of channel names to optimize
        method: Optimization method ('gradient' or 'evolutionary')
        digital_cap: Maximum fraction for digital channels (0-1)
        tv_cap: Maximum fraction for TV spending (0-1)
        
    Returns:
        Formatted optimization results with allocations
    """
    results = service.optimize_allocation(
        total_budget=total_budget,
        channels=channels,
        method=method,
        digital_cap=digital_cap,
        tv_cap=tv_cap
    )
    
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

        **Brand Disambiguation:**
        When a brand is not found or ambiguous:
        1. Use `list_matching_brands` to find potential matches (pass country if available)
        2. Present matches as a numbered list
        3. **CRITICAL**: When user provides ONLY a number (e.g., "1", "2"), you MUST map it to the corresponding brand name from your previous list
           
           **Example:**
           - You show: "1. AGUILA\n2. AGUILA LIGHT"
           - User types: "1"
           - You call: forecast_baseline(country="colombia", brand="AGUILA")
           - DO NOT call: forecast_baseline(country="colombia", brand="1")
           
        4. Always use the EXACT brand name (all uppercase) for forecast_baseline or simulate calls

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
