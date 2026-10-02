import joblib
import pandas as pd


# ============================
# Find Matching Value
# ============================

def find_matching_value(user_input, available_values):

    user_input = (
        str(user_input)
        .strip()
        .lower()
        .replace(" ", "")
    )

    for value in available_values:

        cleaned_value = (
            str(value)
            .strip()
            .lower()
            .replace(" ", "")
        )

        if user_input == cleaned_value:
            return value

    return None


# ============================
# Load Model and Encoders
# ============================

model = joblib.load("yield_model.pkl")

crop_encoder = joblib.load("yield_crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")


# ============================
# YIELD PREDICTION SYSTEM
# ============================

print("=" * 50)
print("        🌾 CROP YIELD PREDICTION")
print("=" * 50)

print("\nPlease enter the following farm details.\n")


# ============================
# User Inputs
# ============================

crop = input(
    "🌱 Enter the crop name (example: Rice): "
)

year = int(
    input("📅 Enter the year of cultivation (example: 2026): ")
)

season = input(
    "🌦️ Enter the cultivation season (example: Kharif): "
)

state = input(
    "📍 Enter your state (example: Tamil Nadu): "
)

area = float(
    input("📐 Enter your farm area in hectares: ")
)

fertilizer = float(
    input("🧪 Enter the amount of fertilizer used in kg: ")
)

pesticide = float(
    input("🧴 Enter the amount of pesticide used in kg: ")
)


# ============================
# Match User Inputs
# ============================

crop = find_matching_value(
    crop,
    crop_encoder.classes_
)

season = find_matching_value(
    season,
    season_encoder.classes_
)

state = find_matching_value(
    state,
    state_encoder.classes_
)


# ============================
# Validate Inputs
# ============================

if crop is None:
    print("\n❌ Crop not found.")
    exit()

if season is None:
    print("\n❌ Season not found.")
    exit()

if state is None:
    print("\n❌ State not found.")
    exit()


# ============================
# Encode Values
# ============================

crop_encoded = crop_encoder.transform([crop])[0]

season_encoded = season_encoder.transform([season])[0]

state_encoded = state_encoder.transform([state])[0]


# ============================
# Prepare Model Input
# ============================

input_data = pd.DataFrame(
    [[
        crop_encoded,
        year,
        season_encoded,
        state_encoded,
        area,
        fertilizer,
        pesticide
    ]],
    columns=[
        "crop",
        "year",
        "season",
        "state",
        "area",
        "fertilizer",
        "pesticide"
    ]
)


# ============================
# Predict Yield
# ============================

predicted_yield = model.predict(input_data)[0]


# ============================
# Calculate Total Production
# ============================

total_production = predicted_yield * area


# ============================
# Display Result
# ============================

print("\n" + "=" * 50)
print("          🌾 YIELD PREDICTION RESULT")
print("=" * 50)

print(f"\nCrop: {crop}")
print(f"State: {state}")
print(f"Season: {season}")
print(f"Area: {area:.2f} hectares")

print(
    f"\n🌾 Predicted Yield: "
    f"{predicted_yield:.2f} tonnes/hectare"
)

print(
    f"📦 Predicted Total Production: "
    f"{total_production:.2f} tonnes"
)

print("\n" + "=" * 50)