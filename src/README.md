# Source Code Structure

This directory contains the web application source code for BrandCompass.

## Directory Structure

```
src/
├── app.py                     # Main Flask application entry point
├── api/                       # API and web interface
│   ├── templates/            # HTML templates for Flask
│   └── static/               # Static assets (CSS, JS)
├── agent/                     # Agent module
│   └── master_agent.py       # Wrapper for production agent
├── forecast/                  # Brand power forecasting
│   └── forecast_service.py   # Baseline forecast generation
├── simulation/                # What-if simulation services
│   └── simulation_service.py # Marketing change simulation
├── optimization/              # Marketing optimization logic
│   └── marketing_optimizer.py # Budget allocation optimizer
├── utils/                     # Utility functions
│   └── web_utils.py          # Web app helper functions
└── models/                    # ML model interfaces
```

## Module Descriptions

### `app.py`
Main Flask web application containing all route handlers and web logic.

### `api/`
Web interface components:
- **templates/**: Jinja2 HTML templates
- **static/**: CSS and JavaScript files

### `agent/`
Agent functionality wrapper that interfaces with the production agent system.

### `forecast/`
Brand power baseline forecasting services.

### `simulation/`
What-if simulation services for analyzing marketing changes.

### `optimization/`
Marketing budget optimization logic and algorithms.

### `utils/`
Helper functions and utilities for the web application.

### `models/`
Clean interface to ML models (wraps production_scripts/models).

## Usage

The web application can be run using either:

1. **Flask app** (traditional web interface):
   ```bash
   python run_flask.py
   ```

2. **FastAPI app** (API + web interface):
   ```bash
   python run_app.py
   ```

## Architecture Notes

This `src/` directory contains the **web application layer** only. The core ML and production logic resides in `production_scripts/` which is kept separate for modularity.

The web app imports and wraps production functionality to provide a clean, user-friendly interface.

