#!/usr/bin/env python3
"""
BrandCompass Application
Main FastAPI application serving both frontend and backend
"""
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import os
import json
from datetime import datetime

# Import the functional service
from src.services.simulation.simulation_service import ForecastSimulateService
from src.agent import get_agent_executor
from langchain_core.messages import AIMessage, HumanMessage

# Minimal POC FastAPI app with JSON persistence
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
EXPERIMENTS_JSON = os.path.join(DATA_DIR, "experiments.json")
UI_STATE_JSON = os.path.join(DATA_DIR, "ui_state.json")

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

def _load_ui_state() -> Dict[str, Any]:
    return _load_json(UI_STATE_JSON, {})

def _save_ui_state(patch: Dict[str, Any]) -> None:
    state = _load_ui_state()
    state.update(patch)
    _save_json(UI_STATE_JSON, state)

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

# Mount static files and templates for frontend
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "api/static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "api/templates"))

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
        baseline_features_df = pd.read_csv("data/forecast_features.csv")
        # Use local model artifact under models/ by default
        model_artifact_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "unified_brand_power_model.pkl")
        if not os.path.exists(model_artifact_path):
            # Fallbacks to existing names if unified is not present
            for name in ["brand_power_forecaster.pkl", "simple_brand_power_forecaster.pkl", "simple_marketing_model.pkl"]:
                candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", name)
                if os.path.exists(candidate):
                    model_artifact_path = candidate
                    break
        _service_cache = ForecastSimulateService(model_path=model_artifact_path, baseline_features_df=baseline_features_df)
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

# --- Frontend Routes (HTML Pages) ---

@app.get("/")
async def index(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analysis")
async def analysis(request: Request):
    """Analysis page with data from last uploaded file (if any)."""
    ctx: Dict[str, Any] = {"request": request}

    # Determine the data source for the editable table
    state = _load_ui_state()
    last_uploaded = state.get("last_uploaded_file")

    # Fallback to a template if no upload yet
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    default_template = os.path.join(uploads_dir, "upload_template_us.csv")
    source_path = last_uploaded if last_uploaded and os.path.exists(last_uploaded) else default_template

    table_columns: List[str] = []
    table_rows: List[Dict[str, Any]] = []
    feature_columns: List[str] = []
    brands: List[str] = []
    years: List[int] = []
    months: List[int] = []

    try:
        df = pd.read_csv(source_path)
        # Normalize column names (keep original for the UI)
        table_columns = list(df.columns)
        # Limit rows for performance
        table_rows = df.head(300).to_dict(orient="records")
        # Marketing feature columns (only those present)
        candidate_features = [
            "digital_spend", "tv_spend", "traditional_spend",
            "sponsorship_spend", "other_spend"
        ]
        feature_columns = [c for c in candidate_features if c in df.columns]
        # Filters
        if "brand" in df.columns:
            brands = sorted([str(x) for x in df["brand"].dropna().unique().tolist()])
        if "year" in df.columns:
            years = sorted([int(x) for x in pd.to_numeric(df["year"], errors="coerce").dropna().unique().tolist()])
        if "month" in df.columns:
            months = sorted([int(x) for x in pd.to_numeric(df["month"], errors="coerce").dropna().unique().tolist()])
    except Exception as e:
        logger.exception("Failed to prepare analysis data: %s", e)

    ctx.update({
        "table_columns": table_columns,
        "table_rows": table_rows,
        "feature_columns": feature_columns,
        "brands": brands,
        "years": years,
        "months": months,
    })

    return templates.TemplateResponse("analysis.html", ctx)

@app.get("/optimization")
async def optimization(request: Request):
    """Optimization page"""
    return templates.TemplateResponse("optimization.html", {"request": request})

# ---------------------------------------------------------------------------
# Frontend compatibility routes (migrated from legacy Flask templates)
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Uploads a CSV file and returns basic metadata for the UI."""
    try:
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        dest_path = os.path.join(uploads_dir, file.filename)

        with open(dest_path, "wb") as f:
            f.write(await file.read())

        # Try to parse as CSV to return dimensions
        rows = cols = 0
        try:
            df = pd.read_csv(dest_path)
            rows, cols = df.shape
        except Exception:
            # Not a CSV or unreadable; still report upload success
            pass

        # Persist UI state for the Analysis page
        _save_ui_state({
            "last_uploaded_file": dest_path,
            "last_uploaded_meta": {"filename": file.filename, "rows": int(rows), "cols": int(cols)}
        })

        return {"success": True, "filename": file.filename, "rows": int(rows), "cols": int(cols)}
    except Exception as e:
        logger.exception("Upload error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/select_country")
async def select_country(country: str = Form(...)):
    """Dummy endpoint used by the UI to persist selected country on reload."""
    # Persist selection for optional server-side defaults
    _save_ui_state({"selected_country": country})
    return {"success": True, "country": country}


@app.get("/download_template")
async def download_template(country: Optional[str] = None):
    """Serve a CSV template; optionally filtered by country."""
    country_map = {
        None: "upload_template_us.csv",
        "us": "upload_template_us.csv",
        "usa": "upload_template_us.csv",
        "brazil": "upload_template_brazil.csv",
        "colombia": "upload_template_colombia.csv",
    }
    filename = country_map.get((country or "").lower(), "upload_template_us.csv")
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    path = os.path.join(uploads_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Template not found")
    return FileResponse(path, media_type="text/csv", filename=filename)

@app.get("/api/v1/forecast/baseline/{country}/{brand}")
def forecast_baseline(country: str, brand: str, max_horizon: int = 4):
    _, baseline_forecast = get_service_and_baseline()
    try:
        logger.info(f"Baseline request: country={country} brand={brand}")
        brand_baseline = baseline_forecast[
            (baseline_forecast['country'] == country.lower()) &
            (baseline_forecast['brand'] == brand.lower())
        ].sort_values(['year', 'quarter'])

        if brand_baseline.empty:
            raise HTTPException(status_code=404, detail=f"Baseline data not found for {country}/{brand}")
        
        preds = brand_baseline['predicted_power'].head(max_horizon).tolist()
        return {"predictions": preds}
    except HTTPException as e:
        raise e
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


@app.post("/calculate")
def calculate(req: Dict[str, Any]):
    """Compatibility endpoint for Analysis page to perform simulation."""
    service, baseline_forecast = get_service_and_baseline()
    try:
        columns: List[str] = req.get("columns") or []
        edited_rows: List[Dict[str, Any]] = req.get("edited_rows") or []
        if not columns or not edited_rows:
            raise HTTPException(status_code=400, detail="columns and edited_rows are required")

        edited_df = pd.DataFrame(edited_rows, columns=columns)
        if edited_df.empty:
            raise HTTPException(status_code=400, detail="edited_rows cannot be empty")

        simulation_result = service.simulate(edited_df, baseline_forecast)

        # Build response similar to /api/v1/simulate/scenario
        baseline_data: Dict[str, List[float]] = {}
        simulated_data: Dict[str, List[float]] = {}
        quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

        # Determine target brands from uploaded data
        target_brands = sorted(edited_df['brand'].dropna().astype(str).unique().tolist())[:5]

        for brand_name in target_brands:
            brand_country = edited_df[edited_df['brand'] == brand_name]['country'].astype(str).iloc[0]

            brand_baseline_df = baseline_forecast[
                (baseline_forecast['country'] == brand_country) & (baseline_forecast['brand'] == brand_name)
            ].sort_values(['year', 'quarter'])
            baseline_data[brand_name] = brand_baseline_df['predicted_power'].head(4).tolist()

            brand_simulated_df = simulation_result[
                (simulation_result['country'] == brand_country) & (simulation_result['brand'] == brand_name)
            ].sort_values(['year', 'quarter'])
            simulated_data[brand_name] = brand_simulated_df['simulated_power'].head(4).tolist()

        return {"baseline": baseline_data, "simulated": simulated_data, "quarters": quarters}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Calculate error: %s", e)
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


# ---------------------------------------------------------------------------
# Compatibility experiment routes for the Analysis page JavaScript
# ---------------------------------------------------------------------------

@app.get("/experiments")
def list_experiments_compat():
    """Return experiments as a simple list for the UI (no pagination wrapper)."""
    store = _load_json(EXPERIMENTS_JSON, {"experiments": []})
    return store.get("experiments", [])


@app.post("/save_experiment")
def save_experiment_compat(payload: Dict[str, Any]):
    """Save an experiment in a simplified schema used by the UI."""
    store = _load_json(EXPERIMENTS_JSON, {"experiments": []})
    record = {
        "id": f"ui_{int(datetime.utcnow().timestamp()*1000)}",
        "name": payload.get("name", "experiment"),
        "timestamp": datetime.utcnow().isoformat(),
        "baseline_data": payload.get("baseline_data"),
        "simulated_data": payload.get("simulated_data"),
        "changes": payload.get("changes", {}),
    }
    store.setdefault("experiments", []).append(record)
    _save_json(EXPERIMENTS_JSON, store)
    return {"success": True, "experiment_count": len(store.get("experiments", []))}


@app.post("/clear_experiments")
def clear_experiments_compat():
    store = {"experiments": []}
    _save_json(EXPERIMENTS_JSON, store)
    return {"success": True}


@app.get("/export_experiments")
def export_experiments_compat():
    store = _load_json(EXPERIMENTS_JSON, {"experiments": []})
    data = json.dumps(store.get("experiments", []), indent=2)
    headers = {
        "Content-Disposition": "attachment; filename=experiments_export.json"
    }
    return Response(content=data, media_type="application/json", headers=headers)


@app.delete("/api/v1/experiments/{index:int}")
def delete_experiment_by_index(index: int):
    """Delete experiment by its list index (compat for UI)."""
    store = _load_json(EXPERIMENTS_JSON, {"experiments": []})
    exps = store.get("experiments", [])
    if index < 0 or index >= len(exps):
        raise HTTPException(status_code=404, detail="Experiment not found")
    exps.pop(index)
    store["experiments"] = exps
    _save_json(EXPERIMENTS_JSON, store)
    return {"success": True}

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, str]]] = None
    selected_country: Optional[str] = None

chat_history = []

@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
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

    response = await agent_executor.ainvoke({"input": input_message, "chat_history": history})
    return {"response": response['output']}

