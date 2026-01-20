from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, validator


class CarbonForecastRequest(BaseModel):
    """Request for carbon intensity forecast using weather data."""
    data_path: Optional[str] = Field(
        None,
        description="Optional path to combined weather+carbon data CSV.",
    )
    horizon_hours: PositiveInt = Field(
        24,
        le=168,
        description="Number of hours to forecast carbon intensity (up to 7 days).",
    )

    def resolve_data_path(self) -> Optional[Path]:
        return Path(self.data_path).expanduser().resolve() if self.data_path else None


class CarbonForecastResponse(BaseModel):
    """Response from carbon intensity forecast."""
    status: str = "forecast_done"
    message: str
    path: str
    forecast_hours: int
    history_start: str
    history_end: str


class RLRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_kwh: PositiveFloat = Field(
        ...,
        alias="energy_kwh",
        description="Total energy in kWh the agent must schedule.",
    )
    max_kw: PositiveFloat = Field(
        ...,
        alias="max_power_kw",
        description="Maximum power draw allowed per hour.",
    )
    hours: PositiveInt = Field(
        ...,
        le=168,
        description="Scheduling horizon in hours (max 7 days).",
    )
    carbon_intensity_path: Optional[str] = Field(
        None,
        description="Optional path to a CSV containing carbon intensity data.",
    )
    episodes: PositiveInt = Field(
        50,
        description="Number of PPO episodes to train (if supported).",
    )

    @validator("max_kw")
    def validate_limits(cls, value: float, values: dict[str, object]) -> float:
        energy = values.get("total_kwh")
        hours = values.get("hours")
        if energy and hours and value * hours < energy:
            raise ValueError(
                "max_kw * hours must be >= total_kwh to make scheduling feasible."
            )
        return value

    def resolve_carbon_path(self) -> Optional[Path]:
        return (
            Path(self.carbon_intensity_path).expanduser().resolve()
            if self.carbon_intensity_path
            else None
        )


class OperationResponse(BaseModel):
    status: str = "done"
    message: str
    path: Optional[str] = None
    metadata: dict[str, object] | None = None


class ScheduleItem(BaseModel):
    hour: PositiveInt
    carbon_intensity_gco2_per_kwh: float = Field(..., ge=0)
    energy_kwh: float = Field(..., ge=0)


class RLScheduleResponse(BaseModel):
    status: str = "rl_done"
    message: str
    path: str
    carbon_saving_percent: float
    total_emissions_kg: float
    total_energy_scheduled_kwh: float
    energy_target_met_percent: float
    schedule: list[ScheduleItem]
    metadata: dict[str, object] | None = None
