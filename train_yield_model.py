import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("dataset/processed_yield_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)




X = df.drop(["yield", "production"], axis=1)
y = df["yield"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))




model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nYield Prediction Model Trained Successfully!")


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n")
print("=" * 50)
print("             MODEL EVALUATION")
print("=" * 50)

print(f"\nMean Absolute Error (MAE): {mae:.4f}")

print(f"Mean Squared Error (MSE): {mse:.4f}")

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

print(f"R² Score: {r2:.4f}")

comparison = pd.DataFrame({
    "Actual Yield": y_test.values,
    "Predicted Yield": y_pred
})

print("\n")
print("=" * 50)
print("       ACTUAL vs PREDICTED YIELD")
print("=" * 50)

print(comparison.head(10))


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

# Perfect prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.title("Actual vs Predicted Yield")

plt.tight_layout()

plt.show()


joblib.dump(
    model,
    "yield_model.pkl"
)

print("\nYield model saved as: yield_model.pkl")

print("\nYield Prediction Training Completed Successfully!")