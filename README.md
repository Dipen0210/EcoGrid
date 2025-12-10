# ⚡ Smart Grid Carbon Reduction System (Reinforcement Learning Edition)

An AI-powered platform that forecasts electricity demand, retrieves real-time carbon
intensity, and optimises energy consumption with reinforcement learning to reduce CO₂
emissions.

---

## 🚀 Features
- LSTM-based demand forecasting with graceful PyTorch fallbacks.
- PPO reinforcement-learning scheduler (heuristic fallback if SB3 is unavailable).
- Carbon-aware optimisation powered by the Electricity Maps API.
- FastAPI backend microservices for data collection, forecasting, and optimisation.
- Streamlit dashboard for interactive visualisation of schedules and metrics.

---

## 🧩 Project Architecture
```
User
  ↓
Streamlit Dashboard (frontend/app.py)
  ↓ HTTP
FastAPI Backend (backend/main.py)
  ↓
Data/ML Layer (backend/data, backend/forecasting, backend/rl)
  ↓
Processed Outputs (backend/data/processed/*.csv)
  ↓
Visualisations & Metrics (frontend/app.py → Plotly/Streamlit)
```

Core data flow:
1. `/fetch-data` pulls demand (EIA), weather (Open-Meteo), and carbon intensity (Electricity Maps); saves to `backend/data/processed/raw_latest.csv`.
2. `/forecast-demand` trains LSTM on ~1 year of history and writes `backend/data/processed/forecast_next24.csv`.
3. `/run-rl` trains the PPO scheduler (or heuristic fallback) based on user constraints and exports `backend/data/processed/rl_schedule.csv`.

---

## Dashboard

<img width="1470" height="580" alt="Screenshot 2025-10-26 at 12 36 13 AM" src="https://github.com/user-attachments/assets/66273244-7e8d-4bea-949b-44f0f7006c85" />


<img width="1213" height="591" alt="Screenshot 2025-11-03 at 10 01 12 AM" src="https://github.com/user-attachments/assets/88f92514-7c42-478d-94bd-295a90cb01c4" />



## 🧰 Tech Stack
- **Python**, **FastAPI**, **Streamlit**
- **TensorFlow**, **Stable-Baselines3**, **scikit-learn**
- **Plotly**, **Pandas**, **NumPy**, **Matplotlib**
- Optional extras: **PyTorch**, **Gymnasium**, **Electricity Maps API**

---

## ⚙️ Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### Backend
```bash
cd backend
uvicorn main:app --reload
# API docs → http://127.0.0.1:8000/docs
```

### Frontend
```bash
cd frontend
streamlit run app.py
# Dashboard → http://localhost:8501
```

Environment variables (optional) should be set before starting the backend:

| Variable               | Description                                             |
|------------------------|---------------------------------------------------------|
| `EIA_API_KEY`          | EIA demand API key (defaults to provided sample key)    |
| `EIA_SERIES_ID`        | Demand series (default `EBA.NYIS-ALL.D.H`)              |
| `OPEN_METEO_LATITUDE`  | Latitude for weather data (default NYC `40.7128`)       |
| `OPEN_METEO_LONGITUDE` | Longitude for weather data (default NYC `-74.0060`)     |
| `ELECTRICITY_MAPS_TOKEN` | Token for carbon intensity API (required for live data) |
| `ELECTRICITY_MAPS_ZONE`  | Zone identifier (default `US-NY`)                       |

All generated datasets and artefacts live under `backend/data/processed/`.

---

## 📡 API Endpoints

| Method | Endpoint           | Description                                                       |
|--------|--------------------|-------------------------------------------------------------------|
| GET    | `/status`          | Health check for the backend service.                            |
| GET    | `/fetch-data`      | Downloads and stores the latest demand, weather, and carbon data.|
| POST   | `/forecast-demand` | Runs the LSTM forecaster and saves a 24-hour demand projection.  |
| POST   | `/run-rl`          | Trains the PPO scheduler and returns an optimised energy plan.   |

Each task returns JSON metadata plus the path to the generated CSV in
`backend/data/processed/`.

---

## 📈 Workflow Summary
1. Fetch latest demand, weather, and carbon intensity data.
2. Train and forecast the next 24 hours of demand using the LSTM model.
3. Run the RL agent to allocate energy to the lowest-carbon hours.
4. Visualise the optimised schedule, carbon intensity, and CO₂ reduction metrics in Streamlit.

---

## 🧪 Example Input (`POST /run-rl`)
```json
{
  "total_kwh": 70,
  "max_kw": 7.2,
  "hours": 24
}
```

## 🧾 Example Output
```json
{
  "status": "rl_done",
  "message": "RL scheduling completed successfully.",
  "path": "backend/data/processed/rl_schedule.csv",
  "carbon_saving_percent": 21.7,
  "total_emissions_kg": 42.8,
  "total_energy_scheduled_kwh": 70.0,
  "energy_target_met_percent": 100.0,
  "schedule": [
    {"hour": 1, "carbon_intensity_gco2_per_kwh": 240, "energy_kwh": 2.8},
    {"hour": 2, "carbon_intensity_gco2_per_kwh": 180, "energy_kwh": 4.5},
    "... more hourly entries ..."
  ]
}
```

---

## 🧠 Future Improvements
- Integrate solar and wind generation forecasts using Open-Meteo solar radiation data.
- Extend to multi-agent RL for coordinating multiple devices or facilities.
- Connect with IoT smart plugs, EV APIs, or building management systems for closed-loop control.

---

## ✅ Repository Checklist
1. `requirements.txt` includes all backend and frontend dependencies.
2. `backend/` and `frontend/` directories contain runnable applications.
3. `backend/data/processed/` holds all generated CSV artefacts.
4. Documentation covers architecture, setup, endpoints, and example payloads.

Once the backend and frontend are running, the complete system can be exercised locally with:
```bash
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```
