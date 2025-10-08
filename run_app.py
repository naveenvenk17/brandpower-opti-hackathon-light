#!/usr/bin/env python3
"""
Helper to run the FastAPI POC locally.
"""
import uvicorn
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import os
import json
from datetime import datetime

# Import the functional service
from production_scripts.services.forecast_simulate_service import ForecastSimulateService
from frontend.utils import get_optimizable_columns

from production_scripts.agent.master_agent import get_agent_executor
from langchain_core.messages import AIMessage, HumanMessage

# Minimal POC FastAPI app with JSON persistence
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
EXPERIMENTS_JSON = os.path.join(DATA_DIR, "experiments.json")

def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("fastapi_app")

app = FastAPI(title="BrandCompass API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (Updated for new workflow) ---
class SimulateRequest(BaseModel):
    edited_rows: List[Dict[str, Any]]
    columns: List[str]
    target_brands: List[str]
    max_horizon: int = Field(4, ge=1, le=4)

class OptimizeRequest(BaseModel):
    total_budget: float = Field(..., gt=0)
    channels: Optional[List[str]] = None
    method: str = 'gradient'
    digital_cap: float = Field(0.99, gt=0, le=1.0)
    tv_cap: float = Field(0.5, gt=0, le=1.0)

class CreateExperimentRequest(BaseModel):
    name: str
    description: Optional[str] = None
    country: str
    brand: str
    scenarios: List[Dict]

# --- Service Instantiation ---
_service_cache = None
_baseline_forecast_cache = None

def get_service_and_baseline():
    """Initializes and caches the core forecasting service and its baseline forecast."""
    global _service_cache, _baseline_forecast_cache
    if _service_cache is None:
        logger.info("Initializing ForecastSimulateService...")
        _service_cache = ForecastSimulateService()
        logger.info("Service initialized.")
    if _baseline_forecast_cache is None:
        logger.info("Generating and caching baseline forecast...")
        _baseline_forecast_cache = _service_cache.forecast_baseline()
        logger.info("Baseline forecast cached.")
    return _service_cache, _baseline_forecast_cache

# --- API Endpoints ---

@app.on_event("startup")
def startup_event():
    get_service_and_baseline()
    if not os.path.exists(EXPERIMENTS_JSON):
        _save_json(EXPERIMENTS_JSON, {"experiments": []})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/v1/forecast/baseline/{country}/{brand}")
def forecast_baseline(country: str, brand: str, max_horizon: int = 4):
    _, baseline_forecast = get_service_and_baseline()
    try:
        logger.info(f"Baseline request: country={country} brand={brand}")
        brand_baseline = baseline_forecast[
            (baseline_forecast['country'] == country) &
            (baseline_forecast['brand'] == brand)
        ].sort_values(['year', 'quarter'])

        if brand_baseline.empty:
            raise HTTPException(status_code=404, detail=f"Baseline data not found for {country}/{brand}")
        
        preds = brand_baseline['predicted_power'].head(max_horizon).tolist()
        return {"predictions": preds}
    except Exception as e:
        logger.exception(f"Baseline error for {country}/{brand}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/simulate/scenario")
def simulate_scenario(req: SimulateRequest):
    """
    Simulates brand power based on a dataframe of user edits, as per the user's specified workflow.
    """
    service, baseline_forecast = get_service_and_baseline()
    try:
        logger.info(f"Simulate request for brands: {req.target_brands}")
        
        # 1. Create DataFrame from frontend data
        edited_df = pd.DataFrame(req.edited_rows, columns=req.columns)
        if edited_df.empty:
            raise HTTPException(status_code=400, detail="edited_rows cannot be empty.")

        # 2. Call the service's simulate method
        simulation_result = service.simulate(edited_df, baseline_forecast)

        # 3. Structure the response for the target brands
        baseline_data = {}
        simulated_data = {}
        
        for brand_name in req.target_brands:
            # Find the brand's country
            brand_country = edited_df[edited_df['brand'] == brand_name]['country'].iloc[0]

            # Get baseline predictions
            brand_baseline_df = baseline_forecast[
                (baseline_forecast['country'] == brand_country) & (baseline_forecast['brand'] == brand_name)
            ].sort_values(['year', 'quarter'])
            baseline_data[brand_name] = brand_baseline_df['predicted_power'].head(req.max_horizon).tolist()

            # Get simulated predictions
            brand_simulated_df = simulation_result[
                (simulation_result['country'] == brand_country) & (simulation_result['brand'] == brand_name)
            ].sort_values(['year', 'quarter'])
            simulated_data[brand_name] = brand_simulated_df['simulated_power'].head(req.max_horizon).tolist()

        return {
            'baseline': baseline_data,
            'simulated': simulated_data,
            'quarters': ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
        }

    except Exception as e:
        logger.exception(f"Simulate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Experiment Endpoints (Unchanged) ---
@app.post("/api/v1/optimize/allocation")
def optimize_allocation(req: OptimizeRequest):
    """
    Optimizes marketing allocation based on a total budget.
    """
    service, _ = get_service_and_baseline()
    try:
        logger.info(f"Optimization request for total budget: {req.total_budget}")
        optimization_results = service.optimize_allocation(
            total_budget=req.total_budget,
            channels=req.channels,
            method=req.method,
            digital_cap=req.digital_cap,
            tv_cap=req.tv_cap
        )
        return optimization_results
    except Exception as e:
        logger.exception(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/experiments")
def create_experiment(req: CreateExperimentRequest):
    store = _load_json(EXPERIMENTS_JSON, {"experiments": []})
    exp_id = f"exp_{int(datetime.utcnow().timestamp()*1000)}"
    record = req.dict()
    record["id"] = exp_id
    record["created_at"] = datetime.utcnow().isoformat()
    record["updated_at"] = datetime.utcnow().isoformat()
    store.setdefault("experiments", []).append(record)
    _save_json(EXPERIMENTS_JSON, store)
    return {"id": exp_id, "scenario_count": len(req.scenarios)}

@app.get("/api/v1/experiments")
def list_experiments(page: int = 1, page_size: int = 20):
    store = _load_json(EXPERIMENTS_JSON, {"experiments": []})
    exps = store.get("experiments", [])
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "experiments": exps[start:end],
        "pagination": {"page": page, "page_size": page_size, "total_count": len(exps)},
    }

@app.delete("/api/v1/experiments/{exp_id}")
def delete_experiment_by_id(exp_id: str):
    store = _load_json(EXPERIMENTS_JSON, {"experiments": []})
    original_count = len(store.get("experiments", []))
    store["experiments"] = [e for e in store.get("experiments", []) if e.get("id") != exp_id]
    if len(store["experiments"]) == original_count:
        raise HTTPException(status_code=404, detail="Experiment not found")
    _save_json(EXPERIMENTS_JSON, store)
    return {"deleted": True}

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, str]]] = None
    selected_country: Optional[str] = None

chat_history = []

@app.post("/api/v1/chat")
def chat(req: ChatRequest):
    agent_executor = get_agent_executor()
    
    history = []
    if req.chat_history:
        for msg in req.chat_history:
            if msg.get('role') == 'user':
                history.append(HumanMessage(content=msg.get('content')))
            elif msg.get('role') == 'assistant':
                history.append(AIMessage(content=msg.get('content')))

    input_message = req.message
    if req.selected_country and req.selected_country != 'None':
        input_message = f"The user has selected the country '{req.selected_country}'.\n\nUser query: {req.message}"

    response = agent_executor.invoke({"input": input_message, "chat_history": history})
    return {"response": response['output']}

if __name__ == "__main__":
    logger.info("Starting BrandCompass FastAPI server...")
    uvicorn.run(
        "run_app:app",
        host="0.0.0.0",
        port=8010,
        log_level="info",
        reload=True
    )
