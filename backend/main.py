"""
FastAPI backend for the Smart Grid Carbon Reduction System.

Provides endpoints for:
1. Fetching energy data (weather, carbon intensity)
2. Forecasting carbon intensity using LSTM with weather features
3. Running RL scheduler to optimize energy consumption
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.schemas import (
    CarbonForecastRequest,
    CarbonForecastResponse,
    OperationResponse,
    RLRequest,
    RLScheduleResponse,
)
from backend.services import data_service, forecast_service, rl_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend")

app = FastAPI(
    title="EcoGrid: Carbon-Aware Energy Scheduler",
    version="2.0.0",
    description=(
        "AI-powered energy scheduling system that uses weather-based carbon intensity "
        "forecasting and reinforcement learning to minimize CO₂ emissions."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    config.ensure_processed_dir()
    logger.info("Processed data directory ensured at %s.", config.PROCESSED_DATA_DIR)


@app.get("/status", response_model=OperationResponse)
async def status() -> OperationResponse:
    """Health check endpoint."""
    logger.debug("Status endpoint invoked.")
    return OperationResponse(message="EcoGrid backend is running.")


@app.get("/fetch-data", response_model=OperationResponse)
async def fetch_data() -> OperationResponse:
    """Fetch latest weather and carbon intensity data from external APIs."""
    logger.info("Fetch-data endpoint triggered.")
    try:
        path, metadata = await run_in_threadpool(data_service.fetch_fresh_data)
        return OperationResponse(
            message="Fetched weather and carbon intensity data.",
            path=str(path),
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception("Data fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/forecast-carbon", response_model=CarbonForecastResponse)
async def forecast_carbon(
    request: CarbonForecastRequest = CarbonForecastRequest(),
) -> CarbonForecastResponse:
    """
    Forecast carbon intensity for the next N hours using LSTM.
    
    The model uses weather features (temperature, humidity, wind speed) 
    to predict future carbon intensity values.
    """
    logger.info(
        "Forecast-carbon endpoint triggered for %d hours.",
        request.horizon_hours,
    )
    try:
        result = await run_in_threadpool(
            forecast_service.run_carbon_intensity_forecast,
            request.resolve_data_path(),
            request.horizon_hours,
        )
        return CarbonForecastResponse(
            message="Carbon intensity forecast generated using weather-based LSTM.",
            path=str(result.path),
            forecast_hours=result.forecast_hours,
            history_start=result.history_start.isoformat(),
            history_end=result.history_end.isoformat(),
        )
    except Exception as exc:
        logger.exception("Carbon forecast failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/run-rl", response_model=RLScheduleResponse)
async def run_rl(request: RLRequest) -> RLScheduleResponse:
    """
    Run the RL scheduler to optimize energy consumption.
    
    Uses carbon intensity data (from forecast or fetched data) to determine 
    the optimal times to consume energy, minimizing total CO₂ emissions.
    """
    logger.info(
        "Run-RL endpoint triggered: total_kwh=%.2f, max_kw=%.2f, hours=%d.",
        request.total_kwh,
        request.max_kw,
        request.hours,
    )
    try:
        path, schedule_records, metadata = await run_in_threadpool(
            rl_service.run_rl_scheduler,
            request.total_kwh,
            request.max_kw,
            request.hours,
            request.resolve_carbon_path(),
            request.episodes,
        )
        return RLScheduleResponse(
            message="RL scheduling completed successfully.",
            path=str(path),
            carbon_saving_percent=float(metadata.get("carbon_saving_percent", 0.0)),
            total_emissions_kg=float(metadata.get("total_emissions_kg", 0.0)),
            total_energy_scheduled_kwh=float(metadata.get("total_energy_scheduled_kwh", 0.0)),
            energy_target_met_percent=float(metadata.get("energy_target_met_percent", 0.0)),
            schedule=schedule_records,
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception("RL scheduling failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def get_app() -> FastAPI:
    """Convenience accessor for ASGI servers."""
    return app
