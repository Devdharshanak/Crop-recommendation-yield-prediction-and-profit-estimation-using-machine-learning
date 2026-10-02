import pandas as pd

# ==============================
# 1. Load datasets
# ==============================

daily_price = pd.read_csv("dataset/daily_price.csv")
commodity_price = pd.read_csv("dataset/commodity_price.csv")

print("Daily price dataset:", daily_price.shape)
print("Commodity price dataset:", commodity_price.shape)


# ==============================
# 2. Clean column names
# ==============================

daily_price.columns = daily_price.columns.str.strip()
commodity_price.columns = commodity_price.columns.str.strip()

print("\nDaily price columns:")
print(list(daily_price.columns))

print("\nCommodity price columns:")
print(list(commodity_price.columns))


# ==============================
# 3. Rename columns
# ==============================

daily_price = daily_price.rename(columns={
    "Commodity": "commodity",
    "Min Price": "min_price",
    "Max Price": "max_price",
    "Modal Price": "modal_price"
})

commodity_price = commodity_price.rename(columns={
    "Commodity": "commodity",
    "Min_x0020_Price": "min_price",
    "Max_x0020_Price": "max_price",
    "Modal_x0020_Price": "modal_price"
})


# ==============================
# 4. Keep required columns
# ==============================

daily_price = daily_price[
    ["commodity", "min_price", "max_price", "modal_price"]
]

commodity_price = commodity_price[
    ["commodity", "min_price", "max_price", "modal_price"]
]


# ==============================
# 5. Combine datasets
# ==============================

price_data = pd.concat(
    [daily_price, commodity_price],
    ignore_index=True
)


# ==============================
# 6. Crop name mapping
# ==============================

def identify_crop(commodity):

    name = str(commodity).lower().strip()

    clean = (
        name.replace("(", " ")
            .replace(")", " ")
            .replace("-", " ")
            .replace("/", " ")
            .replace(",", " ")
    )

    if "black gram" in clean or "urd" in clean:
        return "blackgram"

    if "chickpea" in clean or "kabuli chana" in clean:
        return "chickpea"

    if "lentil" in clean or "masur" in clean:
        return "lentil"

    if "green gram" in clean or "moong" in clean:
        return "mungbean"

    if "arhar" in clean or "tur" in clean or "pigeon pea" in clean:
        return "pigeonpeas"

    if "musk melon" in clean or "muskmelon" in clean or "karbuja" in clean:
        return "muskmelon"

    if "water melon" in clean or "watermelon" in clean:
        return "watermelon"

    if "apple" in clean:
        return "apple"

    if "banana" in clean:
        return "banana"

    if "coconut" in clean:
        return "coconut"

    if "coffee" in clean:
        return "coffee"

    if "cotton" in clean:
        return "cotton"

    if "grape" in clean:
        return "grapes"

    if "jute" in clean:
        return "jute"

    if "maize" in clean:
        return "maize"

    if "mango" in clean:
        return "mango"

    if "orange" in clean:
        return "orange"

    if "papaya" in clean:
        return "papaya"

    if "pomegranate" in clean:
        return "pomegranate"

    if "rice" in clean:
        return "rice"

    return None


# Apply crop mapping
price_data["crop"] = price_data["commodity"].apply(identify_crop)


# ==============================
# 7. Keep matching crops
# ==============================

matched_data = price_data.dropna(subset=["crop"]).copy()


# ==============================
# 8. Convert ₹/quintal to ₹/tonne
# ==============================

matched_data["price_per_tonne"] = (
    pd.to_numeric(
        matched_data["modal_price"],
        errors="coerce"
    ) * 10
)


# Remove invalid prices
matched_data = matched_data.dropna(
    subset=["price_per_tonne"]
)


# ==============================
# 9. Average price for each crop
# ==============================

final_price_data = (
    matched_data
    .groupby("crop", as_index=False)["price_per_tonne"]
    .mean()
)

final_price_data["price_per_tonne"] = (
    final_price_data["price_per_tonne"].round(2)
)


# ==============================
# 10. All 22 crops
# ==============================

recommended_crops = [
    "rice",
    "maize",
    "chickpea",
    "kidneybeans",
    "pigeonpeas",
    "mothbeans",
    "mungbean",
    "blackgram",
    "lentil",
    "pomegranate",
    "banana",
    "mango",
    "grapes",
    "watermelon",
    "muskmelon",
    "apple",
    "orange",
    "papaya",
    "coconut",
    "cotton",
    "jute",
    "coffee"
]

available_crops = set(final_price_data["crop"])

missing_crops = [
    crop for crop in recommended_crops
    if crop not in available_crops
]


# ==============================
# 11. Save final dataset
# ==============================

final_price_data.to_csv(
    "dataset/combined_price_dataset.csv",
    index=False
)


# ==============================
# 12. Display result
# ==============================

print("\n" + "=" * 55)
print("       COMBINED PRICE DATASET")
print("=" * 55)

print("\nCrops with price data:")

for crop in sorted(available_crops):
    print("  ✅", crop)

print("\nMissing crops:")

for crop in missing_crops:
    print("  ❌", crop)

print("\nTotal recommended crops:", len(recommended_crops))
print("Crops with price data:", len(available_crops))
print("Missing crops:", len(missing_crops))

print("\n✅ Saved:")
print("dataset/combined_price_dataset.csv")

print("=" * 55)