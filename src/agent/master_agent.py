"""
Master Agent - Local implementation using src services
"""
import os
import re
from typing import List, Optional, Any

import pandas as pd

from src.services.simulation.simulation_service import ForecastSimulateService


_agent_executor: Any = None

# Initialize service with local data
baseline_features_path = "data/forecast_features.csv"
baseline_features_df = pd.read_csv(baseline_features_path)
model_path = "models/brand_power_forecaster.pkl"
service = ForecastSimulateService(model_path=model_path, baseline_features_df=baseline_features_df)


def _tool(func):
    """Lightweight replacement for langchain.tools.tool decorator to avoid import-time deps.
    When langchain is available, get_agent_executor will wrap real tools.
    """
    return func


@_tool
def list_matching_brands(partial_brand_name: str, country: Optional[str] = None) -> List[str]:
    """List all brands that partially match the given brand name, optionally filtered by country."""
    all_brands = service.list_brands(country=country)
    matching_brands = [brand for brand in all_brands if re.search(partial_brand_name, brand, re.IGNORECASE)]
    return matching_brands


@_tool
def forecast_baseline(country: Optional[str] = None, brand: Optional[str] = None, save_path: Optional[str] = None):
    """Generate baseline forecast. Can be filtered by country and brand."""
    df = service.forecast_baseline(country, brand, save_path)
    if df.empty:
        return "No data found for the specified country and brand."
    return df.to_dict(orient='records')


@_tool
def simulate(uploaded_template_path: str, baseline_forecast_path: Optional[str] = None):
    """Simulate brand power with new marketing allocations from an uploaded template."""
    uploaded_template = pd.read_csv(uploaded_template_path)
    baseline_forecast = pd.read_csv(baseline_forecast_path) if baseline_forecast_path else None
    df = service.simulate(uploaded_template, baseline_forecast)
    if df.empty:
        return "Simulation did not produce any results."
    return df.to_dict(orient='records')


@_tool
def optimize_allocation(
    total_budget: float,
    channels: Optional[List[str]] = None,
    method: str = 'gradient',
    digital_cap: float = 0.99,
    tv_cap: float = 0.5
):
    """Optimize marketing allocation using the MarketingImpactModel."""
    return service.optimize_allocation(total_budget, channels, method, digital_cap, tv_cap)


def get_agent_executor():
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
    except Exception:
        # Fallback lightweight executor used in tests when langchain stack isn't available
        class _FallbackAgentExecutor:
            async def ainvoke(self, inputs: dict):
                user_input = inputs.get("input", "")
                return {"output": f"[fallback agent] {user_input}"}

        _agent_executor = _FallbackAgentExecutor()
        return _agent_executor

    load_dotenv()

    # Wrap our plain functions with langchain tool decorator
    lc_tools = [
        lc_tool()(forecast_baseline),
        lc_tool()(simulate),
        lc_tool()(optimize_allocation),
        lc_tool()(list_matching_brands),
    ]

    llm = AzureChatOpenAI(
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are BrandCompass AI, a professional and polite assistant for the BrandCompass.ai application.

Your primary goal is to help users with brand power forecasting, simulating marketing spend changes, and optimizing marketing budget allocation.

When a user asks for an action, always consider using the available tools. 

**Brand Disambiguation:**
If a user's query involves a brand name that is not an exact match for available data, or if a tool call fails because a brand is not found, you MUST first use the `list_matching_brands` tool to find potential matches. Then, present these matches to the user as a numbered list and politely ask them to select the exact brand by number. Once the user provides a selection, use that specific brand to call the appropriate tool (e.g., `forecast_baseline`, `simulate`, `optimize_allocation`).

**Formatting Tool Results:**
When returning the results of a tool, do not just return the raw output. Instead, format the results in a clear, human-readable way using Markdown. For example, if you get a JSON with a list of items, present it as a Markdown table or a bulleted list.

**Guardrails:**
If the user asks a question that is not related to brand power, marketing, or the functionalities of this application, politely decline and guide them back to the application's purpose.

Always be professional and courteous in your responses."""),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, lc_tools, prompt)
    _agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True)
    return _agent_executor


__all__ = ['get_agent_executor']
