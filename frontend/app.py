"""
EcoGrid Dashboard - Carbon-Aware Energy Scheduling System.

Single-page sequential flow:
1. Fetch weather + carbon data
2. LSTM predicts future carbon intensity
3. RL scheduler optimizes energy consumption
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ecogrid-dashboard")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Backend communication helpers
# ---------------------------------------------------------------------------

def build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def request_backend(
    method: str,
    url: str,
    *,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    try:
        response = requests.request(method=method, url=url, json=json_payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.exception("Backend request failed: %s", exc)
        st.error(f"Request failed: {exc}")
        return {}


def read_csv_if_exists(path_str: Optional[str]) -> Optional[pd.DataFrame]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="EcoGrid",
        page_icon="⚡",
        layout="wide",
    )
    
    # Header
    st.title("⚡ EcoGrid: Carbon-Aware Energy Scheduler")
    st.markdown("**Pipeline:** Fetch Data → Forecast Carbon → Optimize Schedule")
    st.markdown("---")
    
    # Initialize session state
    if "fetch_result" not in st.session_state:
        st.session_state["fetch_result"] = None
    if "forecast_result" not in st.session_state:
        st.session_state["forecast_result"] = None
    if "rl_result" not in st.session_state:
        st.session_state["rl_result"] = None
    
    backend_url = DEFAULT_BACKEND_URL
    
    # Check backend status
    status = request_backend("GET", build_url(backend_url, "/status"))
    if status:
        st.success(f"✅ {status.get('message', 'Backend connected')}")
    else:
        st.error("❌ Backend not reachable")
        return
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Parameters")
    total_kwh = st.sidebar.number_input("Total Energy (kWh)", min_value=10.0, max_value=500.0, value=70.0)
    max_kw = st.sidebar.number_input("Max Power (kW)", min_value=1.0, max_value=20.0, value=7.2)
    hours = st.sidebar.number_input("Horizon (hours)", min_value=1, max_value=48, value=24, step=1)
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All", use_container_width=True):
        st.session_state["fetch_result"] = None
        st.session_state["forecast_result"] = None
        st.session_state["rl_result"] = None
        st.rerun()
    
    # =========================================================================
    # STEP 1: FETCH DATA
    # =========================================================================
    st.header("📡 Step 1: Fetch Data")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        fetch_clicked = st.button("🔄 Fetch Data", use_container_width=True)
    
    if fetch_clicked:
        with st.spinner("Fetching weather and carbon data..."):
            response = request_backend("GET", build_url(backend_url, "/fetch-data"))
            if response:
                dataset = read_csv_if_exists(response.get("path"))
                st.session_state["fetch_result"] = {"response": response, "dataset": dataset}
    
    # Show fetch results
    if st.session_state["fetch_result"]:
        result = st.session_state["fetch_result"]
        response = result.get("response", {})
        dataset = result.get("dataset")
        
        st.success(f"✅ {response.get('message', 'Data fetched')}")
        
        if isinstance(dataset, pd.DataFrame) and not dataset.empty:
            # Summary metrics
            if "carbon_intensity_gco2_per_kwh" in dataset.columns:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Data Points", len(dataset))
                m2.metric("Avg Carbon", f"{dataset['carbon_intensity_gco2_per_kwh'].mean():.0f} gCO₂/kWh")
                m3.metric("Min Carbon", f"{dataset['carbon_intensity_gco2_per_kwh'].min():.0f} gCO₂/kWh")
                m4.metric("Max Carbon", f"{dataset['carbon_intensity_gco2_per_kwh'].max():.0f} gCO₂/kWh")
            
            with st.expander("📋 View Raw Data"):
                st.dataframe(dataset.tail(20), use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # STEP 2: FORECAST CARBON
    # =========================================================================
    st.header("🔮 Step 2: Forecast Carbon Intensity")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        forecast_clicked = st.button("🧠 Run LSTM", use_container_width=True)
    
    if forecast_clicked:
        with st.spinner("Training LSTM model (weather → carbon)..."):
            response = request_backend("POST", build_url(backend_url, "/forecast-carbon"), timeout=180)
            if response:
                dataframe = read_csv_if_exists(response.get("path"))
                st.session_state["forecast_result"] = {"response": response, "dataframe": dataframe}
    
    # Show forecast results
    if st.session_state["forecast_result"]:
        result = st.session_state["forecast_result"]
        response = result.get("response", {})
        dataframe = result.get("dataframe")
        
        st.success(f"✅ {response.get('message', 'Forecast generated')}")
        st.info(f"📊 Trained on data from {response.get('history_start', 'N/A')[:10]} to {response.get('history_end', 'N/A')[:10]}")
        
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            # Determine column name
            carbon_col = "predicted_carbon_intensity_gco2_per_kwh"
            if carbon_col not in dataframe.columns:
                carbon_col = "carbon_intensity_gco2_per_kwh"
            
            if carbon_col in dataframe.columns:
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(dataframe["timestamp"]),
                    y=dataframe[carbon_col],
                    mode="lines+markers",
                    name="Predicted Carbon",
                    line=dict(color="#2ecc71", width=3),
                    marker=dict(size=8),
                ))
                fig.update_layout(
                    title="🌿 Predicted Carbon Intensity (Next 24h)",
                    xaxis_title="Time",
                    yaxis_title="Carbon Intensity (gCO₂/kWh)",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Best/worst hours
                df_sorted = dataframe.sort_values(carbon_col)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🌱 Greenest Hours (Low Carbon):**")
                    for _, row in df_sorted.head(5).iterrows():
                        ts = pd.to_datetime(row["timestamp"])
                        val = row[carbon_col]
                        st.markdown(f"- {ts.strftime('%H:%M')} → {val:.0f} gCO₂/kWh")
                with col2:
                    st.markdown("**🔥 Dirtiest Hours (High Carbon):**")
                    for _, row in df_sorted.tail(5).iloc[::-1].iterrows():
                        ts = pd.to_datetime(row["timestamp"])
                        val = row[carbon_col]
                        st.markdown(f"- {ts.strftime('%H:%M')} → {val:.0f} gCO₂/kWh")
    
    st.markdown("---")
    
    # =========================================================================
    # STEP 3: RUN RL SCHEDULER
    # =========================================================================
    st.header("🧠 Step 3: Run RL Scheduler")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        rl_clicked = st.button("⚡ Optimize", use_container_width=True)
    
    if rl_clicked:
        with st.spinner("Running RL scheduler..."):
            payload = {"total_kwh": total_kwh, "max_kw": max_kw, "hours": int(hours)}
            response = request_backend("POST", build_url(backend_url, "/run-rl"), json_payload=payload, timeout=180)
            if response:
                schedule_records = response.get("schedule") or []
                dataframe = pd.DataFrame(schedule_records) if schedule_records else None
                st.session_state["rl_result"] = {"response": response, "dataframe": dataframe}
    
    # Show RL results
    if st.session_state["rl_result"]:
        result = st.session_state["rl_result"]
        response = result.get("response", {})
        dataframe = result.get("dataframe")
        
        st.success(f"✅ {response.get('message', 'Scheduling complete')}")
        
        # Data source indicator
        metadata = response.get("metadata") or {}
        data_source = metadata.get("carbon_data_source", "unknown")
        if data_source == "LSTM forecast":
            st.info("🧠 **Using LSTM-predicted carbon intensity** for scheduling")
        else:
            st.info(f"📊 Carbon data source: {data_source}")
        
        # Metrics
        carbon_saving = float(response.get("carbon_saving_percent", 0.0))
        total_emissions = float(response.get("total_emissions_kg", 0.0))
        energy_met = float(response.get("energy_target_met_percent", 0.0))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("🌿 CO₂ Reduction", f"{carbon_saving:.1f}%", delta="vs uniform")
        m2.metric("💨 Total Emissions", f"{total_emissions:.2f} kg")
        m3.metric("⚡ Energy Target", f"{energy_met:.0f}%")
        
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dataframe["hour"],
                y=dataframe["energy_kwh"],
                name="Energy (kWh)",
                marker_color="#3498db",
                yaxis="y1",
            ))
            fig.add_trace(go.Scatter(
                x=dataframe["hour"],
                y=dataframe["carbon_intensity_gco2_per_kwh"],
                mode="lines+markers",
                name="Carbon (gCO₂/kWh)",
                line=dict(color="#e74c3c", width=3),
                yaxis="y2",
            ))
            fig.update_layout(
                title="📊 Optimized Schedule vs Carbon Intensity",
                xaxis_title="Hour",
                yaxis=dict(title="Energy (kWh)", side="left", color="#3498db"),
                yaxis2=dict(title="Carbon (gCO₂/kWh)", overlaying="y", side="right", color="#e74c3c"),
                template="plotly_dark",
                height=450,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            dispatched = dataframe["energy_kwh"].sum()
            st.info(f"⚡ Scheduled **{dispatched:.1f} kWh** across {len(dataframe)} hours")
            
            with st.expander("📋 View Schedule Details"):
                st.dataframe(dataframe, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("🔗 API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")


if __name__ == "__main__":
    main()
