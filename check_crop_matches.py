import pandas as pd

# Load datasets
crop_df = pd.read_csv("dataset/Crop_recommendation.csv")
price_df = pd.read_csv("dataset/daily_price.csv")

# Get crop names from both datasets
recommended_crops = set(
    crop_df["label"].str.strip().str.lower().unique()
)

price_crops = set(
    price_df["Commodity"].str.strip().str.lower().unique()
)

# Find exact matches
matches = recommended_crops.intersection(price_crops)

# Find crops without price data
missing = recommended_crops - price_crops

print("\n" + "=" * 50)
print("       CROP PRICE DATA MATCH CHECK")
print("=" * 50)

print(f"\nTotal recommended crops: {len(recommended_crops)}")
print(f"Crops with price data:   {len(matches)}")
print(f"Crops without price data:{len(missing)}")

print("\n✅ MATCHING CROPS:")
for crop in sorted(matches):
    print(" -", crop)

print("\n❌ MISSING CROPS:")
for crop in sorted(missing):
    print(" -", crop)

print("=" * 50)