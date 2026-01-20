"""
Carbon Intensity Forecaster using LSTM with Weather Features.

Predicts future carbon intensity (gCO₂/kWh) based on weather data
(temperature, humidity, wind speed). The predictions are used by the
RL scheduler to optimize energy consumption timing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend import config
from backend.utils.files import ensure_directory

logger = logging.getLogger(__name__)

# Use last 7 days of data for training
LOOKBACK_HOURS = 7 * 24

# Weather features used for prediction
WEATHER_FEATURES = ["temperature_c", "relative_humidity_pct", "wind_speed_mps"]


@dataclass
class CarbonForecastResult:
    """Result of carbon intensity forecast."""
    path: Path
    history_start: datetime
    history_end: datetime
    forecast_hours: int


def run_carbon_forecast(
    data_path: Optional[Path] = None,
    output_dir: Path | str = config.PROCESSED_DATA_DIR,
    horizon_hours: int = config.DEFAULT_FORECAST_HORIZON_HOURS,
) -> CarbonForecastResult:
    """Run the carbon intensity forecast pipeline using weather features."""
    output_dir = ensure_directory(Path(output_dir))
    
    # Load combined data with weather and carbon intensity
    data_df = _load_combined_data(data_path)
    
    if data_df.empty:
        raise ValueError("No data available for carbon intensity forecasting.")
    
    history_end = data_df["timestamp"].max()
    cutoff = history_end - timedelta(hours=LOOKBACK_HOURS - 1)
    trimmed_data = data_df[data_df["timestamp"] >= cutoff]
    
    if len(trimmed_data) < LOOKBACK_HOURS // 2:
        logger.warning(
            "Limited data available (%d points); using full history of %d points.",
            len(trimmed_data),
            len(data_df),
        )
        trimmed_data = data_df
    
    history_start = trimmed_data["timestamp"].min()
    
    # Extract features and target
    weather_features = trimmed_data[WEATHER_FEATURES].values.astype(np.float32)
    carbon_target = trimmed_data["carbon_intensity_gco2_per_kwh"].values.astype(np.float32)
    timestamps = trimmed_data["timestamp"]
    
    # Train and forecast
    model = CarbonLSTMForecaster()
    logger.info(
        "Training Carbon LSTM with %d historical points (weather → carbon) to predict %d hours.",
        len(carbon_target),
        horizon_hours,
    )
    
    forecast_values = model.forecast(weather_features, carbon_target, horizon_hours)
    
    # Generate forecast timestamps
    last_timestamp = timestamps.max()
    forecast_index = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
        tz=last_timestamp.tzinfo or timezone.utc,
    )
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        "timestamp": forecast_index,
        "predicted_carbon_intensity_gco2_per_kwh": forecast_values,
    })
    
    output_path = output_dir / "carbon_forecast_next24.csv"
    forecast_df.to_csv(output_path, index=False)
    logger.info("Carbon intensity forecast saved to %s.", output_path)
    
    return CarbonForecastResult(
        path=output_path,
        history_start=history_start,
        history_end=history_end,
        forecast_hours=horizon_hours,
    )


def _load_combined_data(data_path: Optional[Path]) -> pd.DataFrame:
    """Load data containing weather features and carbon intensity."""
    candidates: list[Path] = []
    
    if data_path and data_path.exists():
        candidates.append(data_path)
    else:
        # Prefer raw_latest.csv which has all combined data
        for candidate in (
            config.PROCESSED_DATA_DIR / "raw_latest.csv",
            config.PROCESSED_DATA_DIR / "raw_year.csv",
        ):
            if candidate.exists():
                candidates.append(candidate)
                break
    
    required_cols = ["timestamp", "carbon_intensity_gco2_per_kwh"] + WEATHER_FEATURES
    
    for path in candidates:
        try:
            df = pd.read_csv(path)
            if all(col in df.columns for col in required_cols):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.sort_values("timestamp").dropna(subset=required_cols)
                logger.info("Loaded %d rows from %s for carbon forecasting.", len(df), path)
                return df
            else:
                missing = [c for c in required_cols if c not in df.columns]
                logger.warning("Missing columns %s in %s", missing, path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
    
    logger.warning("No suitable data found; generating synthetic data.")
    return _generate_synthetic_data()


def _generate_synthetic_data(hours: int = 24 * 30) -> pd.DataFrame:
    """Generate synthetic weather and carbon data for testing."""
    np.random.seed(42)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=now, periods=hours, freq="h", tz=timezone.utc)
    
    # Simulate weather patterns
    hour_of_day = np.array([t.hour for t in index])
    day_of_year = np.array([t.timetuple().tm_yday for t in index])
    
    # Temperature: daily cycle + seasonal variation
    temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + 5 * np.sin(2 * np.pi * hour_of_day / 24)
    temp += np.random.normal(0, 2, hours)
    
    # Humidity: inverse correlation with temperature
    humidity = 60 - 0.5 * temp + np.random.normal(0, 10, hours)
    humidity = np.clip(humidity, 20, 95)
    
    # Wind speed: random with some pattern
    wind = 5 + 3 * np.sin(2 * np.pi * hour_of_day / 12) + np.random.exponential(2, hours)
    wind = np.clip(wind, 0, 25)
    
    # Carbon intensity: inversely correlated with wind (more wind = more renewables = less carbon)
    # Also higher during peak demand hours
    base_carbon = 350
    wind_effect = -8 * wind  # Higher wind → lower carbon
    demand_effect = 50 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak at noon
    carbon = base_carbon + wind_effect + demand_effect + np.random.normal(0, 20, hours)
    carbon = np.clip(carbon, 150, 600)
    
    return pd.DataFrame({
        "timestamp": index,
        "temperature_c": temp,
        "relative_humidity_pct": humidity,
        "wind_speed_mps": wind,
        "carbon_intensity_gco2_per_kwh": carbon,
    })


@dataclass
class _TorchArtifacts:
    torch: Any
    nn: Any
    optim: Any


class CarbonLSTMForecaster:
    """LSTM forecaster that predicts carbon intensity from weather features."""
    
    def __init__(
        self,
        sequence_length: int = 24,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 50,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._torch = self._load_torch()
    
    def forecast(
        self,
        weather_features: np.ndarray,
        carbon_target: np.ndarray,
        horizon: int,
    ) -> list[float]:
        """Train on weather→carbon relationship and forecast future carbon intensity."""
        if len(carbon_target) < self.sequence_length + 1:
            logger.warning("Insufficient data for LSTM; using heuristic forecast.")
            return self._heuristic_forecast(weather_features, carbon_target, horizon)
        
        if self._torch is None:
            logger.warning("PyTorch not available; using heuristic forecast.")
            return self._heuristic_forecast(weather_features, carbon_target, horizon)
        
        artifacts = self._torch
        torch = artifacts.torch
        nn = artifacts.nn
        optim = artifacts.optim
        
        # Normalize features and target
        features_norm, feat_mean, feat_std = self._normalize_2d(weather_features)
        target_norm, target_mean, target_std = self._normalize_1d(carbon_target)
        
        n_features = weather_features.shape[1]
        
        # Prepare training data
        X_train, y_train = self._create_sequences(features_norm, target_norm)
        
        if len(X_train) == 0:
            logger.warning("No training sequences created; using heuristic forecast.")
            return self._heuristic_forecast(weather_features, carbon_target, horizon)
        
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        # Build model
        model = self._build_model(n_features, self.hidden_size, nn)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            predictions = model(X_tensor).squeeze()
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug("Epoch %d/%d - Loss: %.5f", epoch + 1, self.epochs, loss.item())
        
        # Forecast future values
        model.eval()
        predictions: list[float] = []
        
        # Start with last sequence of normalized features
        current_features = torch.tensor(
            features_norm[-self.sequence_length:],
            dtype=torch.float32
        ).unsqueeze(0)
        
        with torch.no_grad():
            for step in range(horizon):
                pred_norm = model(current_features).item()
                pred_value = self._denormalize(pred_norm, target_mean, target_std)
                pred_value = max(100.0, min(800.0, pred_value))  # Clip to reasonable range
                predictions.append(float(pred_value))
                
                # For next step, use last known weather pattern (simplified)
                # In production, you'd use weather forecast data
                next_features = current_features[:, -1:, :].clone()
                current_features = torch.cat([
                    current_features[:, 1:, :],
                    next_features
                ], dim=1)
        
        return predictions
    
    def _create_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create training sequences from features and target."""
        X, y = [], []
        for i in range(len(target) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_size: int, hidden_size: int, nn: Any) -> Any:
        """Build PyTorch LSTM model."""
        import torch
        
        class CarbonLSTM(torch.nn.Module):
            def __init__(self, input_size: int, hidden_size: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.2,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                return self.fc(last_hidden)
        
        return CarbonLSTM(input_size, hidden_size)
    
    def _heuristic_forecast(
        self,
        weather_features: np.ndarray,
        carbon_target: np.ndarray,
        horizon: int,
    ) -> list[float]:
        """Simple heuristic: use correlation between wind and carbon."""
        # Higher wind → lower carbon (more renewables)
        recent_carbon = carbon_target[-24:] if len(carbon_target) >= 24 else carbon_target
        recent_wind = weather_features[-24:, 2] if len(weather_features) >= 24 else weather_features[:, 2]
        
        base_carbon = float(np.mean(recent_carbon))
        wind_mean = float(np.mean(recent_wind))
        
        # Create a simple daily pattern with slight randomness
        predictions = []
        for hour in range(horizon):
            # Daily pattern: higher during day, lower at night
            hour_of_day = hour % 24
            daily_factor = 30 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            
            # Add small random variation
            noise = np.random.normal(0, 15)
            
            pred = base_carbon + daily_factor + noise
            pred = max(150.0, min(600.0, pred))
            predictions.append(float(pred))
        
        return predictions
    
    @staticmethod
    def _normalize_1d(values: np.ndarray) -> tuple[np.ndarray, float, float]:
        mean = float(np.mean(values))
        std = float(np.std(values)) or 1.0
        return (values - mean) / std, mean, std
    
    @staticmethod
    def _normalize_2d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std[std == 0] = 1.0
        return (values - mean) / std, mean, std
    
    @staticmethod
    def _denormalize(value: float, mean: float, std: float) -> float:
        return value * std + mean
    
    @staticmethod
    def _load_torch() -> Optional[_TorchArtifacts]:
        try:
            import torch
            from torch import nn, optim
            return _TorchArtifacts(torch=torch, nn=nn, optim=optim)
        except Exception as exc:
            logger.warning("PyTorch not available: %s", exc)
            return None
