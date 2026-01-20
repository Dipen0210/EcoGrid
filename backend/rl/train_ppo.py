from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import os

import numpy as np
import pandas as pd

from backend import config
from backend.rl.envs import CarbonAwareEnv
from backend.utils.files import ensure_directory, latest_file

logger = logging.getLogger(__name__)
ENABLE_SB3 = os.getenv("ENABLE_SB3_PPO", "").lower() in {"1", "true", "yes", "on"}


def train_and_schedule(
    energy_kwh: float,
    max_power_kw: float,
    hours: int,
    carbon_intensity_path: Optional[Path],
    output_dir: Path | str = config.PROCESSED_DATA_DIR,
    episodes: int = 200,
) -> Tuple[Path, list[dict], dict]:
    """Train the PPO agent (or heuristic fallback) and persist the resulting schedule."""
    output_dir = ensure_directory(Path(output_dir))

    carbon_df = _load_carbon_dataset(
        path=carbon_intensity_path, horizon_hours=hours
    )
    intensities = carbon_df["carbon_intensity_gco2_per_kwh"].to_numpy(dtype=float)

    trainer = PPOScheduler(episodes=episodes)
    logger.info(
        "Training PPO scheduler for %d hours, total energy %.2f kWh, max power %.2f kW.",
        hours,
        energy_kwh,
        max_power_kw,
    )
    schedule = trainer.train(intensities, energy_kwh, max_power_kw)

    schedule_df = _build_schedule_dataframe(carbon_df, schedule)
    output_path = output_dir / "rl_schedule.csv"
    schedule_df.to_csv(output_path, index=False)
    logger.info("RL schedule persisted to %s.", output_path)

    schedule_records = _schedule_records(schedule_df)
    metadata = _schedule_metrics(schedule_df, energy_kwh)
    metadata.update({"episodes": trainer.episodes, "implementation": trainer.mode})

    return output_path, schedule_records, metadata


# ---------------------------------------------------------------------------
# PPO orchestration
# ---------------------------------------------------------------------------


@dataclass
class PPOScheduler:
    episodes: int

    def __post_init__(self) -> None:
        self.mode = self._detect_backend()

    def _detect_backend(self) -> str:
        if not ENABLE_SB3:
            logger.info(
                "Stable-Baselines3 PPO disabled (set ENABLE_SB3_PPO=1 to enable); using heuristic scheduler."
            )
            return "heuristic"
        try:
            import gymnasium as gym  # noqa: F401
            from stable_baselines3 import PPO  # noqa: F401

            return "stable-baselines3"
        except Exception:
            logger.warning(
                "stable-baselines3 / gymnasium not available; falling back to heuristic scheduler."
            )
            return "heuristic"

    def train(
        self, carbon_intensity: np.ndarray, energy_kwh: float, max_power_kw: float
    ) -> np.ndarray:
        if self.mode == "stable-baselines3":
            try:
                return self._train_with_sb3(carbon_intensity, energy_kwh, max_power_kw)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "PPO training using stable-baselines3 failed; using heuristic. %s",
                    exc,
                )
        return self._train_with_heuristic(
            carbon_intensity, energy_kwh, max_power_kw
        )

    def _train_with_sb3(
        self, carbon_intensity: np.ndarray, energy_kwh: float, max_power_kw: float
    ) -> np.ndarray:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO

        horizon = len(carbon_intensity)
        env = CarbonAwareEnv(
            carbon_intensity=carbon_intensity,
            energy_kwh=energy_kwh,
            max_power_kw=max_power_kw,
        )

        # Wrap environment for SB3 compatibility
        class _WrappedEnv(gym.Env):
            metadata = {"render.modes": []}

            def __init__(self, inner_env: CarbonAwareEnv) -> None:
                super().__init__()
                self.inner_env = inner_env
                self.action_space = spaces.Box(
                    low=0.0, high=max_power_kw, shape=(1,), dtype=np.float32
                )
                self.observation_space = spaces.Box(
                    low=0.0,
                    high=np.array(
                        [
                            max(carbon_intensity.max(), 1.0),
                            energy_kwh,
                            1.0,
                        ]
                    ),
                    shape=(3,),
                    dtype=np.float32,
                )

            def reset(self, *, seed=None, options=None):
                state = self.inner_env.reset()
                return state.astype(np.float32), {}

            def step(self, action):
                result = self.inner_env.step(float(action[0]))
                obs = result.state.astype(np.float32)
                reward = float(result.reward)
                done = result.done
                info = {"emissions": result.emissions}
                return obs, reward, done, False, info

        wrapped_env = _WrappedEnv(env)
        model = PPO(
            "MlpPolicy",
            wrapped_env,
            verbose=0,
            n_steps=min(32, horizon),
            batch_size=32,
            learning_rate=3e-4,
        )
        model.learn(total_timesteps=int(self.episodes * horizon))

        schedule = np.zeros(horizon, dtype=float)
        obs, _ = wrapped_env.reset()
        for hour in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = wrapped_env.step(action)
            schedule[hour] = float(np.clip(action[0], 0.0, max_power_kw))
            if terminated or truncated:
                break

        # Adjust final schedule to meet exact energy requirement
        schedule = _enforce_energy_constraint(schedule, energy_kwh, max_power_kw)
        return schedule

    def _train_with_heuristic(
        self, carbon_intensity: np.ndarray, energy_kwh: float, max_power_kw: float
    ) -> np.ndarray:
        self.mode = "heuristic_blended"
        horizon = len(carbon_intensity)
        if horizon == 0:
            return np.array([], dtype=float)

        intensities = carbon_intensity.astype(float)
        max_intensity = float(np.max(intensities)) if horizon else 0.0
        weights = max_intensity - intensities + 1e-3
        weights = np.clip(weights, 1e-6, None)
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            weights = np.ones_like(weights) / horizon
        else:
            weights = weights / weights_sum

        schedule = weights * energy_kwh
        schedule = np.clip(schedule, 0.0, max_power_kw)

        deficit = energy_kwh - float(schedule.sum())
        if deficit > 1e-6:
            order = np.argsort(intensities)
            for idx in order:
                if deficit <= 1e-6:
                    break
                available = max_power_kw - schedule[idx]
                if available <= 0:
                    continue
                delta = min(available, deficit)
                schedule[idx] += delta
                deficit -= delta

        schedule = _enforce_energy_constraint(schedule, energy_kwh, max_power_kw)
        return schedule


def _load_carbon_dataset(
    path: Optional[Path], horizon_hours: int
) -> pd.DataFrame:
    columns = [
        "timestamp",
        "carbon_intensity_gco2_per_kwh",
    ]
    df: Optional[pd.DataFrame] = None

    candidates: list[Path] = []
    if path and path.exists():
        candidates.append(path)
    else:
        for candidate in (
            config.PROCESSED_DATA_DIR / "carbon_latest.csv",
            config.PROCESSED_DATA_DIR / "raw_latest.csv",
        ):
            if candidate.exists():
                candidates.append(candidate)
                break
        else:
            latest_carbon = latest_file(config.PROCESSED_DATA_DIR, "carbon_intensity_*.csv")
            if latest_carbon:
                candidates.append(latest_carbon)
            latest_energy = latest_file(config.PROCESSED_DATA_DIR, "energy_data_*.csv")
            if latest_energy:
                candidates.append(latest_energy)

    for candidate in candidates:
        try:
            logger.debug("Attempting to load carbon data from %s.", candidate)
            df = pd.read_csv(candidate)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", candidate, exc)

    if df is None or df.empty:
        logger.warning("Falling back to synthetic carbon intensity data.")
        return _synthetic_carbon_dataframe(horizon_hours)

    if "carbon_intensity_gco2_per_kwh" not in df.columns:
        # Handle LSTM forecast file which uses 'predicted_carbon_intensity_gco2_per_kwh'
        if "predicted_carbon_intensity_gco2_per_kwh" in df.columns:
            df["carbon_intensity_gco2_per_kwh"] = df["predicted_carbon_intensity_gco2_per_kwh"]
            logger.info("Using LSTM-predicted carbon intensity values.")
        else:
            possible = [col for col in df.columns if "carbon" in col.lower()]
            if possible:
                df["carbon_intensity_gco2_per_kwh"] = df[possible[0]]
            else:
                logger.warning(
                    "No carbon intensity column found; generating synthetic profile."
                )
                return _synthetic_carbon_dataframe(horizon_hours)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    else:
        df["timestamp"] = pd.date_range(
            end=datetime.now(timezone.utc), periods=len(df), freq="H", tz=timezone.utc
        )

    df = df.sort_values("timestamp").dropna(subset=["carbon_intensity_gco2_per_kwh"])
    if len(df) < horizon_hours:
        df = _extend_to_horizon(df, horizon_hours)
    else:
        df = df.tail(horizon_hours)
    df = df.reset_index(drop=True)
    return df[columns]


def _synthetic_carbon_dataframe(hours: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(start=now, periods=hours, freq="H", tz=timezone.utc)
    phase = np.linspace(0, np.pi * 2, hours)
    intensity = 350 + 80 * np.sin(phase) + np.random.normal(0, 15, hours)
    intensity = np.clip(intensity, 150, 600)
    return pd.DataFrame(
        {
            "timestamp": index,
            "carbon_intensity_gco2_per_kwh": intensity,
        }
    )


def _extend_to_horizon(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    deficit = hours - len(df)
    last_row = df.iloc[-1]
    timestamps = pd.date_range(
        start=last_row["timestamp"] + pd.Timedelta(hours=1),
        periods=deficit,
        freq="H",
        tz=timezone.utc,
    )
    filler = pd.DataFrame(
        {
            "timestamp": timestamps,
            "carbon_intensity_gco2_per_kwh": last_row[
                "carbon_intensity_gco2_per_kwh"
            ],
        }
    )
    return pd.concat([df, filler], ignore_index=True)


def _build_schedule_dataframe(
    carbon_df: pd.DataFrame, schedule: np.ndarray
) -> pd.DataFrame:
    schedule = schedule[: len(carbon_df)]
    emissions = schedule * carbon_df["carbon_intensity_gco2_per_kwh"].to_numpy() / 1000.0
    return pd.DataFrame(
        {
            "timestamp": carbon_df["timestamp"],
            "scheduled_energy_kwh": schedule,
            "carbon_intensity_gco2_per_kwh": carbon_df[
                "carbon_intensity_gco2_per_kwh"
            ],
            "emissions_kg": emissions,
        }
    )


def _schedule_metrics(schedule_df: pd.DataFrame, target_energy_kwh: float) -> dict:
    total_energy = float(schedule_df["scheduled_energy_kwh"].sum())
    total_emissions = float(schedule_df["emissions_kg"].sum())
    avg_intensity = float(
        (
            schedule_df["carbon_intensity_gco2_per_kwh"]
            .where(schedule_df["scheduled_energy_kwh"] > 0)
            .mean()
        )
        or 0.0
    )
    hours = len(schedule_df)
    baseline_energy_per_hour = target_energy_kwh / hours if hours else 0.0
    baseline_emissions = float(
        (schedule_df["carbon_intensity_gco2_per_kwh"] * baseline_energy_per_hour).sum()
        / 1000.0
    )
    carbon_saving_percent = 0.0
    if baseline_emissions > 0:
        carbon_saving_percent = max(
            (baseline_emissions - total_emissions) / baseline_emissions * 100.0, 0.0
        )
    energy_target_met_percent = 0.0
    if target_energy_kwh > 0:
        energy_target_met_percent = min(total_energy / target_energy_kwh * 100.0, 100.0)

    return {
        "total_energy_scheduled_kwh": total_energy,
        "target_energy_kwh": target_energy_kwh,
        "total_emissions_kg": total_emissions,
        "average_carbon_intensity_gco2_per_kwh": avg_intensity,
        "energy_deficit_kwh": float(max(target_energy_kwh - total_energy, 0.0)),
        "carbon_saving_percent": carbon_saving_percent,
        "energy_target_met_percent": energy_target_met_percent,
    }


def _enforce_energy_constraint(
    schedule: np.ndarray, target_energy_kwh: float, max_power_kw: float
) -> np.ndarray:
    schedule = np.clip(schedule, 0.0, max_power_kw)
    total = schedule.sum()
    if abs(total - target_energy_kwh) < 1e-6:
        return schedule

    adjustment = target_energy_kwh - total
    if abs(adjustment) <= 1e-6:
        return schedule

    per_hour_adjust = adjustment / len(schedule)
    schedule += per_hour_adjust
    schedule = np.clip(schedule, 0.0, max_power_kw)

    # Final correction by distributing residue greedily
    residue = target_energy_kwh - schedule.sum()
    if abs(residue) > 1e-5:
        direction = np.sign(residue)
        for idx in range(len(schedule)):
            if abs(residue) <= 1e-5:
                break
            available = (
                max_power_kw - schedule[idx]
                if direction > 0
                else schedule[idx]
            )
            delta = min(abs(residue), available)
            schedule[idx] += direction * delta
            residue -= direction * delta

    return schedule


def _schedule_records(schedule_df: pd.DataFrame) -> list[dict]:
    records = []
    for idx, row in enumerate(schedule_df.itertuples(index=False), start=1):
        records.append(
            {
                "hour": idx,
                "carbon_intensity_gco2_per_kwh": float(
                    row.carbon_intensity_gco2_per_kwh
                ),
                "energy_kwh": float(row.scheduled_energy_kwh),
            }
        )
    return records
