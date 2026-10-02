import joblib
import numpy as np
import pandas as pd

# Load trained model and label encoder
model = joblib.load("crop_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Get input from user
print("Enter soil and weather details:")

N = float(input("Nitrogen (N) kg/ha: "))
P = float(input("Phosphorus (P) kg/ha: "))
K = float(input("Potassium (K) kg/ha: "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity(%): "))
ph = float(input("Soil pH (0-14): "))
rainfall = float(input("Rainfall (mm): "))

# Create input dataframe
user_input = pd.DataFrame(
    [[N, P, K, temperature, humidity, ph, rainfall]],
    columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
)

# Get probability for every crop
probabilities = model.predict_proba(user_input)[0]

# Get indices of top 3 probabilities
top3_indices = np.argsort(probabilities)[-3:][::-1]

print("\n🌱 TOP 3 CROP RECOMMENDATIONS")
print("-" * 40)

for rank, index in enumerate(top3_indices, start=1):
    crop = encoder.inverse_transform([index])[0]
    confidence = probabilities[index] * 100

    print(f"{rank}. {crop.capitalize()} - {confidence:.2f}% confidence")