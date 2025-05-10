# Solar System Dynamics - Predicting All Planets' Orbits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === 1. Load or Simulate Data ===
# (You can adjust this part if you already have a real CSV)

# Planet radii (in meters)
planets = {
    'Mercury': 5.79e10,
    'Venus': 1.08e11,
    'Earth': 1.496e11,
    'Mars': 2.28e11,
    'Jupiter': 7.78e11,
    'Saturn': 1.43e12,
    'Uranus': 2.87e12,
    'Neptune': 4.5e12,
    'Pluto': 5.9e12
}

np.random.seed(42)
steps = 365
times = np.linspace(0, 2 * np.pi, steps)

# Create full orbital dataset
orbit_data = []
for planet, radius in planets.items():
    x = radius * np.cos(times) + np.random.normal(0, radius * 0.01, size=steps)
    y = radius * np.sin(times) + np.random.normal(0, radius * 0.01, size=steps)
    temp_df = pd.DataFrame({'planet': planet, 'x': x, 'y': y})
    orbit_data.append(temp_df)

full_orbit_df = pd.concat(orbit_data, ignore_index=True)

# === 2. Feature Engineering ===
full_orbit_df['x_prev'] = full_orbit_df.groupby('planet')['x'].shift(1)
full_orbit_df['y_prev'] = full_orbit_df.groupby('planet')['y'].shift(1)
full_orbit_df['time_prev'] = np.tile(np.linspace(0, 1, steps), len(planets))
full_orbit_df = full_orbit_df.dropna()

# === 3. Train Random Forest for each planet ===
planet_models = {}
for planet in full_orbit_df['planet'].unique():
    planet_data = full_orbit_df[full_orbit_df['planet'] == planet]
    
    X = planet_data[['time_prev', 'x_prev', 'y_prev']]
    y = planet_data[['x', 'y']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    planet_models[planet] = (rf, X_test, y_test)

print("✅ Finished training models for all planets.")

# === 4. Plot Predictions for All Planets ===
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

for idx, (planet, (model, X_test, y_test)) in enumerate(planet_models.items()):
    ax = axes[idx]
    
    y_pred = model.predict(X_test)
    
    ax.scatter(y_test['x'], y_test['y'], label='Actual', alpha=0.6, s=20)
    ax.scatter(y_pred[:, 0], y_pred[:, 1], label='Predicted', alpha=0.6, s=20)
    ax.scatter(0, 0, color='yellow', edgecolors='black', s=200, marker='o', label='Sun')
    ax.set_title(f"{planet} Orbit Prediction", fontsize=14)
    ax.set_xlabel('X position (meters)')
    ax.set_ylabel('Y position (meters)')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
