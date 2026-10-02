import joblib
import numpy as np
import pandas as pd


# =========================================================
# 1. LOAD MODELS AND ENCODERS
# =========================================================

crop_model = joblib.load("crop_model.pkl")
crop_encoder = joblib.load("label_encoder.pkl")

yield_model = joblib.load("yield_model.pkl")
yield_crop_encoder = joblib.load("yield_crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")


# =========================================================
# 2. LOAD PRICE AND COST DATA
# =========================================================

price_df = pd.read_csv("dataset/combined_price_dataset.csv")
cost_df = pd.read_csv("dataset/crop_cost_dataset.csv")

# Make sure year is numeric
cost_df["year"] = pd.to_numeric(cost_df["year"], errors="coerce")



# =========================================================
# 3. HELPER FUNCTION
# =========================================================

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


# =========================================================
# 4. INPUTS
# =========================================================

print("=" * 60)
print("       🌾 SMART CROP RECOMMENDATION SYSTEM")
print("=" * 60)

print("\n🌱 SOIL AND WEATHER DETAILS")
print("-" * 40)

N = float(input("Nitrogen (N) kg/ha: "))
P = float(input("Phosphorus (P) kg/ha: "))
K = float(input("Potassium (K) kg/ha: "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("Soil pH (0-14): "))
rainfall = float(input("Rainfall (mm): "))


print("\n🚜 FARM DETAILS")
print("-" * 40)

year = int(input("Year of cultivation: "))

season = input(
    "Season (example: Kharif): "
)

state = input(
    "State (example: Tamil Nadu): "
)

area = float(
    input("Farm area (hectares): ")
)

fertilizer = float(
    input("Fertilizer used (kg): ")
)

pesticide = float(
    input("Pesticide used (kg): ")
)


# =========================================================
# 5. MATCH SEASON AND STATE
# =========================================================

season = find_matching_value(
    season,
    season_encoder.classes_
)

state = find_matching_value(
    state,
    state_encoder.classes_
)

if season is None:

    print("\n❌ Season not found.")

    print(
        "Available seasons:",
        list(season_encoder.classes_)
    )

    exit()


if state is None:

    print("\n❌ State not found.")

    print(
        "Available states:",
        list(state_encoder.classes_)
    )

    exit()


# =========================================================
# 6. TOP 3 CROP RECOMMENDATION
# =========================================================

crop_input = pd.DataFrame(
    [[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]],
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


probabilities = crop_model.predict_proba(
    crop_input
)[0]


# =========================================================
# GET CROP RECOMMENDATIONS
# Find 3 crops that are supported by the yield model
# =========================================================

# =========================================================
# GET CROP RECOMMENDATIONS
# Find 3 crops supported by the yield model
# =========================================================

ranked_indices = np.argsort(
    probabilities
)[::-1]

top3_crops = []

available_yield_crops = {
    str(crop).strip().lower().replace(" ", "")
    for crop in yield_crop_encoder.classes_
}

for index in ranked_indices:

    encoded_crop = crop_model.classes_[index]

    crop = crop_encoder.inverse_transform(
        [encoded_crop]
    )[0]

    confidence = probabilities[index] * 100

    crop_clean = (
        str(crop)
        .strip()
        .lower()
        .replace(" ", "")
    )

    if crop_clean in available_yield_crops:

        actual_crop = find_matching_value(
            crop,
            yield_crop_encoder.classes_
        )

        top3_crops.append(
            (actual_crop, confidence)
        )

        # IMPORTANT: break must be inside the for loop
        if len(top3_crops) == 3:
            break


# =========================================================
# 7. PRICE FUNCTION
# =========================================================

def get_price(crop):

    temp = price_df.copy()

    temp["crop_clean"] = temp[
        "crop"
    ].astype(str).str.lower().str.replace(
        " ", "", regex=False
    )

    crop_clean = (
        str(crop)
        .lower()
        .replace(" ", "")
    )

    matches = temp[
        temp["crop_clean"] == crop_clean
    ]

    if matches.empty:
        return None

    return matches[
        "price_per_tonne"
    ].mean()


# =========================================================
# 8. COST FUNCTION
# =========================================================
def get_cost(crop, state, year):

    matches = cost_df[
        (cost_df["crop"].str.lower() == crop.lower()) &
        (cost_df["state"].str.lower() == state.lower())
    ].copy()

    if matches.empty:
        return None

    matches["year"] = pd.to_numeric(
        matches["year"],
        errors="coerce"
    )

    exact = matches[matches["year"] == year]

    if not exact.empty:
        return exact.iloc[0]["cost_per_hectare"]

    previous = matches[matches["year"] <= year]

    if not previous.empty:
        latest = previous.sort_values("year").iloc[-1]
        return latest["cost_per_hectare"]

    return matches.iloc[0]["cost_per_hectare"]


# =========================================================
# 9. PREDICT YIELD + PROFIT
# =========================================================

results = []


for crop, confidence in top3_crops:

    try:

        crop_encoded = yield_crop_encoder.transform(
            [crop]
        )[0]

    except ValueError:

        print(
            f"\n⚠️ {crop} not available "
            "in yield dataset."
        )

        continue


    season_encoded = season_encoder.transform(
        [season]
    )[0]

    state_encoded = state_encoder.transform(
        [state]
    )[0]


    # Yield model input
    yield_input = pd.DataFrame(
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


    # Predict yield
    predicted_yield = yield_model.predict(
        yield_input
    )[0]


    # Production
    production = (
        predicted_yield * area
    )


    # Market price
    price = get_price(crop)


    # Cultivation cost
    cost_per_hectare = get_cost(
        crop,
        state,
        year
    )


    # Revenue
    if price is not None:

        revenue = production * price

    else:

        revenue = None


    # Total cost
    if cost_per_hectare is not None:

        total_cost = (
            cost_per_hectare * area
        )

    else:

        total_cost = None


    # Profit
    if (
        revenue is not None
        and total_cost is not None
    ):

        profit = revenue - total_cost

    else:

        profit = None


    results.append({
        "crop": crop,
        "confidence": confidence,
        "yield": predicted_yield,
        "production": production,
        "price": price,
        "cost": cost_per_hectare,
        "total_cost": total_cost,
        "revenue": revenue,
        "profit": profit
    })


# =========================================================
# 10. RANK BY PROFIT
# =========================================================

profit_results = [
    result
    for result in results
    if result["profit"] is not None
]


profit_results.sort(
    key=lambda x: x["profit"],
    reverse=True
)


# =========================================================
# 11. DISPLAY FINAL RESULT
# =========================================================

print("\n")
print("=" * 65)
print("       🏆 FINAL CROP RECOMMENDATION")
print("=" * 65)


for rank, result in enumerate(
    profit_results,
    start=1
):

    if rank == 1:
        medal = "🥇"

    elif rank == 2:
        medal = "🥈"

    else:
        medal = "🥉"


    print(f"\n{medal} RANK {rank}")
    print("-" * 50)

    print(
        f"🌱 Crop: "
        f"{result['crop'].capitalize()}"
    )

    print(
        f"📊 Recommendation confidence: "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"🌾 Predicted yield: "
        f"{result['yield']:.2f} tonnes/hectare"
    )

    print(
        f"📦 Estimated production: "
        f"{result['production']:.2f} tonnes"
    )

    print(
        f"💰 Market price: "
        f"₹{result['price']:,.2f}/tonne"
    )

    print(
        f"💸 Cost per hectare: "
        f"₹{result['cost']:,.2f}"
    )

    print(
        f"💵 Total cultivation cost: "
        f"₹{result['total_cost']:,.2f}"
    )

    print(
        f"📈 Estimated revenue: "
        f"₹{result['revenue']:,.2f}"
    )

    print(
        f"🏆 Estimated profit: "
        f"₹{result['profit']:,.2f}"
    )


# =========================================================
# 12. UNAVAILABLE PROFIT DATA
# =========================================================

unavailable = [
    result
    for result in results
    if result["profit"] is None
]


if unavailable:

    print("\n⚠️ PROFIT DATA UNAVAILABLE")
    print("-" * 40)

    for result in unavailable:

        print(
            f"{result['crop'].capitalize()} "
            "- price or cost data unavailable"
        )


print("\n" + "=" * 65)
print("✅ PREDICTION COMPLETED")
print("=" * 65)