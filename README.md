# ⚡ EcoGrid: Carbon-Aware Energy Scheduler

An AI-powered energy scheduling system that uses **weather-based carbon intensity forecasting** and **reinforcement learning** to minimize CO₂ emissions.

---

## 🎯 What It Does

EcoGrid intelligently schedules your energy consumption to run during the **cleanest hours** of the day — when renewable energy is abundant and carbon intensity is low.

**Example:** Instead of charging your EV immediately, EcoGrid schedules it for 3 AM when wind power is high, reducing your carbon footprint by ~20%.

---

## 🏗️ Architecture (Connected Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   📡 Fetch Data                                             │
│   └── Weather: Temperature, Humidity, Wind Speed            │
│   └── Carbon Intensity: gCO₂/kWh from Electricity Maps      │
│                                                             │
│              ↓                                              │
│                                                             │
│   🔮 LSTM Carbon Forecaster                                 │
│   └── Input: Weather features (Solar, wind, hydro)          │
│   └── Output: Predicted carbon intensity (next 24h)         │
│                                                             │
│              ↓                                              │
│                                                             │
│   🧠 RL Scheduler (PPO)                                     │
│   └── Input: Carbon forecast + energy requirements          │
│   └── Output: Optimized hourly energy schedule              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Weather → Carbon correlation is real:
- More wind or high sun beams → More wind power or more solar power → Lower carbon intensity
- Higher demand (cold/hot days) → More fossil fuels → Higher carbon intensity

---

## 🚀 Features

- **Weather-based LSTM** predicts future carbon intensity using temperature, humidity, and wind speed
- **PPO Reinforcement Learning** optimizes when to consume energy (with heuristic fallback)
- **Real-time data** from EIA, Electricity Maps, and Open-Meteo APIs
- **Interactive dashboard** with Plotly visualizations

---

## Dashboard

<img width="1470" height="745" alt="Screenshot 2026-01-19 at 6 27 11 PM" src="https://github.com/user-attachments/assets/dddfd7ae-38b1-4c48-9ee3-8a0e55bcb038" />

<img width="1470" height="744" alt="Screenshot 2026-01-19 at 6 27 23 PM" src="https://github.com/user-attachments/assets/e33437d6-5bcb-4942-8f7b-4d9fb62f3979" />

<img width="1470" height="753" alt="Screenshot 2026-01-19 at 6 27 41 PM" src="https://github.com/user-attachments/assets/41fed56b-8e64-4b56-99a2-a67b92eb0744" />

<img width="1470" height="469" alt="Screenshot 2026-01-19 at 6 28 10 PM" src="https://github.com/user-attachments/assets/ae5bf520-fe65-4b26-bc76-65537a7e20b2" />

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI |
| ML/Forecasting | PyTorch LSTM |
| RL | Stable-Baselines3 (PPO) |
| Frontend | Streamlit, Plotly |
| Data | Pandas, NumPy |

---

### Run the system

```bash
# Terminal 1: Start backend
uvicorn backend.main:app --reload

# Terminal 2: Start frontend
streamlit run frontend/app.py
```

- **Dashboard:** http://localhost:8501
- **Backend:** http://127.0.0.1:8000

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Health check |
| GET | `/fetch-data` | Download weather + carbon data from APIs |
| POST | `/forecast-carbon` | Run LSTM to predict carbon intensity |
| POST | `/run-rl` | Run RL scheduler to optimize energy consumption |

---

## 📊 Example Workflow

1. **Fetch Data** → Gets latest weather and carbon intensity
2. **Forecast Carbon** → LSTM predicts next 24h of carbon intensity
3. **Run RL Scheduler** → Optimizes when to use your 70 kWh

**Result:** Schedule shows when to consume energy for minimal CO₂ emissions.

---

## 🔧 Environment Variables (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `ELECTRICITY_MAPS_TOKEN` | API token for carbon data | - |
| `ELECTRICITY_MAPS_ZONE` | Grid zone | `US-NY` |
| `ENABLE_SB3_PPO` | Enable Stable-Baselines3 | `false` |

---

## 📁 Project Structure

```
EcoGrid/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Configuration
│   ├── schemas.py              # Pydantic models
│   ├── forecasting/
│   │   └── carbon_forecast.py  # LSTM carbon intensity predictor
│   ├── rl/
│   │   ├── train_ppo.py        # PPO training + heuristic fallback
│   │   └── envs.py             # RL environment
│   └── services/
│       ├── data_service.py     # Data fetching
│       ├── forecast_service.py # Forecast orchestration
│       └── rl_service.py       # RL orchestration
├── frontend/
│   └── app.py                  # Streamlit dashboard
└── requirements.txt
```

---

## 🧠 How It Works

### 1. Weather → Carbon Correlation
Wind speed is inversely correlated with carbon intensity. When it's windy, more electricity comes from wind farms (zero carbon), reducing grid-wide emissions.

### 2. LSTM Forecasting
The model learns patterns like:
- Wind typically picks up at night → Lower carbon overnight
- Hot afternoons → AC demand → Higher carbon

### 3. RL Optimization
Given your energy needs and the carbon forecast, the agent decides: *"I'll schedule most consumption during hours 2-6 AM when carbon is lowest."*

---

## 📈 Sample Output

```json
{
  "carbon_saving_percent": 21.7,
  "total_emissions_kg": 42.8,
  "schedule": [
    {"hour": 1, "carbon_intensity": 280, "energy_kwh": 2.8},
    {"hour": 2, "carbon_intensity": 220, "energy_kwh": 5.5},
    ...
  ]
}
```

---

## 🌱 Future Improvements

- [ ] Real-time IoT integration (smart plugs, EV chargers)
- [ ] Multi-objective optimization (cost + carbon)
- [ ] Transformer models for better forecasting
- [ ] Multi-region support

---

## 📜 License

MIT License
