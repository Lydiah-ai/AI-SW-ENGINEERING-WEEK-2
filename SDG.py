import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset (replace 'crop_data.csv' with actual dataset)
df = pd.read_csv('crop_data.csv')

# Select relevant features (e.g., rainfall, temperature, soil quality)
features = ['rainfall', 'temperature', 'soil_quality']
target = 'crop_yield'

# Data preprocessing
df.dropna(inplace=True)  # Remove missing values
X = df[features]
y = df[target]

# Split data into training and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Visualization: Actual vs Predicted values
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")
plt.title("Crop Yield Prediction: Actual vs Predicted")
plt.show()