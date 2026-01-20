"""Forecast service for carbon intensity prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend import config
from backend.forecasting.carbon_forecast import CarbonForecastResult, run_carbon_forecast

logger = logging.getLogger(__name__)


def run_carbon_intensity_forecast(
    data_path: Optional[Path] = None,
    horizon_hours: int = config.DEFAULT_FORECAST_HORIZON_HOURS,
) -> CarbonForecastResult:
    """Run the carbon intensity forecast workflow.
    
    Uses weather features (temperature, humidity, wind speed) to predict
    future carbon intensity values for the scheduling horizon.
    """
    processed_dir = config.ensure_processed_dir()
    resolved_data_path = data_path

    if resolved_data_path is None:
        # Look for combined weather + carbon data
        candidate = processed_dir / "raw_latest.csv"
        if candidate.exists():
            resolved_data_path = candidate
            logger.info("Using latest dataset at %s for carbon forecasting.", resolved_data_path)
        else:
            logger.warning(
                "No existing dataset found. Forecast will rely on synthetic data."
            )

    result = run_carbon_forecast(
        data_path=resolved_data_path,
        output_dir=processed_dir,
        horizon_hours=horizon_hours,
    )
    
    logger.info(
        "Carbon forecast complete using history %s -> %s. Saved to %s.",
        result.history_start.isoformat(),
        result.history_end.isoformat(),
        result.path,
    )
    return result
