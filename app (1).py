
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Wildlife Population Forecast", layout="wide")

# Title and description
st.title("Wildlife Population Forecasting")
st.markdown("Explore and forecast wildlife population trends by species and region using LSTM models.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("wildlife_population.csv")
    return df

df = load_data()

# User selections
species = st.selectbox("Select Species", sorted(df["Species"].unique()))
region = st.selectbox("Select Region", sorted(df["Region"].unique()))

# Filter data
filtered = df[(df["Species"] == species) & (df["Region"] == region)].sort_values("Year")
years = filtered["Year"].values.reshape(-1, 1)
populations = filtered["Population"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
pop_scaled = scaler.fit_transform(populations)

# Prepare sequences
def create_sequences(data, window_size=3):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(pop_scaled)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build and train model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Forecast future values
def forecast(model, data, steps=5, window_size=3):
    prediction_input = data[-window_size:]
    preds = []
    for _ in range(steps):
        pred = model.predict(prediction_input.reshape(1, window_size, 1), verbose=0)[0]
        preds.append(pred[0])
        prediction_input = np.vstack((prediction_input[1:], pred))
    return np.array(preds)

future_preds_scaled = forecast(model, pop_scaled, steps=5)
future_preds = scaler.inverse_transform(future_preds_scaled.reshape(-1, 1))

# Future years
last_year = int(filtered["Year"].max())
future_years = np.arange(last_year + 1, last_year + 6)

# Plot
fig, ax = plt.subplots()
ax.plot(filtered["Year"], filtered["Population"], label="Actual", marker='o')
ax.plot(future_years, future_preds, label="Forecast", marker='x')
ax.set_xlabel("Year")
ax.set_ylabel("Population")
ax.set_title(f"Population Forecast for {species} in {region}")
ax.legend()
st.pyplot(fig)
