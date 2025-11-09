import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Streamlit page setup
st.set_page_config(page_title="Wildlife Population Forecast", layout="wide")

# Title and info
st.title("🌍 Wildlife Population Forecasting Dashboard")
st.markdown("""
Explore **real-time forecasts** of wildlife population trends by species and region using **LSTM neural networks**.
Use the dropdowns to filter species and region, and view interactive plots up to **year 2030**.
""")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("wildlife_population.csv")

df = load_data()

# Sidebar controls
st.sidebar.header("🔎 Filter Options")
species = st.sidebar.selectbox("Select Species", sorted(df["Species"].unique()))
region = st.sidebar.selectbox("Select Region", sorted(df["Region"].unique()))
window_size = st.sidebar.slider("LSTM Window Size", 2, 10, 3)
epochs = st.sidebar.slider("Training Epochs", 50, 300, 100, step=50)

# Filter dataset
filtered = df[(df["Species"] == species) & (df["Region"] == region)].sort_values("Year")
years = filtered["Year"].values.reshape(-1, 1)
populations = filtered["Population"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
pop_scaled = scaler.fit_transform(populations)

# Sequence creation
def create_sequences(data, window_size=3):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(pop_scaled, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=epochs, verbose=0)

# Forecast function
def forecast(model, data, steps=5, window_size=3):
    prediction_input = data[-window_size:]
    preds = []
    for _ in range(steps):
        pred = model.predict(prediction_input.reshape(1, window_size, 1), verbose=0)[0]
        preds.append(pred[0])
        prediction_input = np.vstack((prediction_input[1:], pred))
    return np.array(preds)

# Predict future (till 2030)
future_steps = max(2030 - int(filtered["Year"].max()), 5)
future_preds_scaled = forecast(model, pop_scaled, steps=future_steps, window_size=window_size)
future_preds = scaler.inverse_transform(future_preds_scaled.reshape(-1, 1))

# Create future years
last_year = int(filtered["Year"].max())
future_years = np.arange(last_year + 1, last_year + 1 + future_steps)

# --- INTERACTIVE PLOTLY CHART ---
fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(
    x=filtered["Year"], 
    y=filtered["Population"], 
    mode='lines+markers',
    name='Actual Population',
    line=dict(color='green', width=3),
    marker=dict(size=8)
))

# Forecasted data
fig.add_trace(go.Scatter(
    x=future_years, 
    y=future_preds.flatten(), 
    mode='lines+markers',
    name='Forecasted Population (till 2030)',
    line=dict(color='orange', width=3, dash='dash'),
    marker=dict(size=8, symbol='x')
))

# Styling
fig.update_layout(
    title=f"📈 Population Forecast for {species} in {region}",
    xaxis_title="Year",
    yaxis_title="Population",
    hovermode='x unified',
    template='plotly_white',
    legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=None,  # full width in Streamlit
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Forecast summary
st.markdown("---")
st.subheader("📊 Forecast Summary")
forecast_df = pd.DataFrame({
    "Year": future_years,
    "Predicted Population": future_preds.flatten()
})
st.dataframe(forecast_df, use_container_width=True)
