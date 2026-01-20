"""RL scheduling service for carbon-aware energy optimization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from backend import config
from backend.rl.train_ppo import train_and_schedule

logger = logging.getLogger(__name__)


def run_rl_scheduler(
    energy_kwh: float,
    max_power_kw: float,
    hours: int,
    carbon_intensity_path: Optional[Path] = None,
    episodes: int = 200,
) -> Tuple[Path, list[dict], dict]:
    """
    Train the RL scheduler and generate an optimized energy schedule.
    
    Priority order for carbon intensity data:
    1. LSTM forecast (carbon_forecast_next24.csv) - predicted values
    2. User-provided path
    3. Latest fetched carbon data (carbon_latest.csv)
    4. Combined raw data (raw_latest.csv)
    5. Synthetic data (fallback)
    """
    processed_dir = config.ensure_processed_dir()
    resolved_carbon_path = carbon_intensity_path
    data_source = "user-provided"

    if resolved_carbon_path is None:
        # Priority 1: Use LSTM carbon forecast if available
        forecast_path = processed_dir / "carbon_forecast_next24.csv"
        if forecast_path.exists():
            resolved_carbon_path = forecast_path
            data_source = "LSTM forecast"
            logger.info("Using LSTM carbon intensity forecast for scheduling.")
        else:
            # Priority 2: Latest fetched carbon data
            carbon_latest = processed_dir / "carbon_latest.csv"
            if carbon_latest.exists():
                resolved_carbon_path = carbon_latest
                data_source = "fetched data"
            else:
                # Priority 3: Combined raw dataset
                raw_latest = processed_dir / "raw_latest.csv"
                if raw_latest.exists():
                    resolved_carbon_path = raw_latest
                    data_source = "raw combined data"

    if resolved_carbon_path:
        logger.info(
            "Using carbon data from %s (%s) for RL scheduling.",
            resolved_carbon_path,
            data_source,
        )
    else:
        logger.warning(
            "No carbon intensity data found. Scheduler will use synthetic data."
        )
        data_source = "synthetic"

    schedule_path, schedule_records, metadata = train_and_schedule(
        energy_kwh=energy_kwh,
        max_power_kw=max_power_kw,
        hours=hours,
        carbon_intensity_path=resolved_carbon_path,
        output_dir=processed_dir,
        episodes=episodes,
    )
    
    # Add data source to metadata for transparency
    metadata["carbon_data_source"] = data_source
    
    logger.info("RL scheduling complete. Saved to %s.", schedule_path)
    return schedule_path, schedule_records, metadata
