
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
import pandas as pd

from production_scripts.services.forecast_simulate_service import ForecastSimulateService

load_dotenv()

# Initialize the ForecastSimulateService
service = ForecastSimulateService()

@tool
def forecast_baseline(country: str = None, brand: str = None, save_path: str = None):
    """Generate baseline forecast. Can be filtered by country and brand."""
    return service.forecast_baseline(country, brand, save_path).to_json()

@tool
def simulate(uploaded_template_path: str, baseline_forecast_path: str = None):
    """Simulate brand power with new marketing allocations from an uploaded template."""
    uploaded_template = pd.read_csv(uploaded_template_path)
    baseline_forecast = pd.read_csv(baseline_forecast_path) if baseline_forecast_path else None
    return service.simulate(uploaded_template, baseline_forecast).to_json()

@tool
def optimize_allocation(total_budget: float, channels: list = None, method: str = 'gradient', digital_cap: float = 0.99, tv_cap: float = 0.5):
    """Optimize marketing allocation using the MarketingImpactModel."""
    return service.optimize_allocation(total_budget, channels, method, digital_cap, tv_cap)

tools = [forecast_baseline, simulate, optimize_allocation]

llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are BrandCompass AI, a professional and polite assistant for the BrandCompass.ai application.

You can help with the following:
- Forecasting brand power.
- Simulating the impact of marketing spend changes.
- Optimizing marketing budget allocation.

When a user asks for an action, use the available tools. 
When returning the results of a tool, do not just return the raw output. Instead, format the results in a clear, human-readable way using Markdown.
For example, if you get a JSON with a list of items, present it as a Markdown table or a bulleted list.

If the user asks a question that is not related to brand power, marketing, or the functionalities of this application, politely decline and guide them back to the application's purpose.

Always be professional and courteous in your responses."""),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_agent_executor():
    return agent_executor
