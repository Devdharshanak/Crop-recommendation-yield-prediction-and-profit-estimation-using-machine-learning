import pandas as pd
import os

# -----------------------------------
# 1. Load Cost Dataset
# -----------------------------------

input_file = "dataset/cost-of-cultivation.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# -----------------------------------
# 2. Select Required Columns
# -----------------------------------

df = df[
    [
        "year",
        "state_name",
        "crop_name",
        "cul_cost_a2fl"
    ]
]

# Rename columns
df = df.rename(columns={
    "state_name": "state",
    "crop_name": "crop",
    "cul_cost_a2fl": "cost_per_hectare"
})

# -----------------------------------
# 3. Crop Name Mapping
# -----------------------------------

crop_mapping = {

    "Paddy": "rice",

    "Maize": "maize",

    "Gram": "chickpea",

    "Tur (Arhar)": "pigeonpeas",

    "Moong": "mungbean",

    "Urad": "blackgram",

    "Masur": "lentil",

    "Lentil": "lentil",

    "Cotton": "cotton",

    "Jute": "jute",

    "Ragi": "ragi"

}

df["crop"] = df["crop"].map(crop_mapping)

# -----------------------------------
# 4. Remove Crops We Don't Need
# -----------------------------------

df = df.dropna(subset=["crop"])

# Remove rows where cost is missing
df = df.dropna(subset=["cost_per_hectare"])

# -----------------------------------
# 5. Clean State Names
# -----------------------------------

df["state"] = (
    df["state"]
    .astype(str)
    .str.strip()
)

# -----------------------------------
# 6. Convert Cost to Numeric
# -----------------------------------

df["cost_per_hectare"] = pd.to_numeric(
    df["cost_per_hectare"],
    errors="coerce"
)

df = df.dropna(subset=["cost_per_hectare"])

# -----------------------------------
# 7. Save Clean Dataset
# -----------------------------------

output_file = "dataset/crop_cost_dataset.csv"

df.to_csv(output_file, index=False)

print("\nClean dataset shape:", df.shape)

print("\nAvailable crops:")
print(sorted(df["crop"].unique()))

print("\nNumber of states:")
print(df["state"].nunique())

print("\nSample data:")
print(df.head(10))

print("\nSaved successfully:")
print(output_file)