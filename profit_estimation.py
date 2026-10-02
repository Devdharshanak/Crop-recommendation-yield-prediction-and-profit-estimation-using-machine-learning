import pandas as pd

# -----------------------------------
# 1. Load Price and Cost Data
# -----------------------------------

price_file = "dataset/combined_price_dataset.csv"
cost_file = "dataset/crop_cost_dataset.csv"

price_df = pd.read_csv(price_file)
cost_df = pd.read_csv(cost_file)

# -----------------------------------
# 2. Clean Crop Names
# -----------------------------------

price_df["crop"] = (
    price_df["crop"]
    .astype(str)
    .str.strip()
    .str.lower()
)

cost_df["crop"] = (
    cost_df["crop"]
    .astype(str)
    .str.strip()
    .str.lower()
)

cost_df["state"] = (
    cost_df["state"]
    .astype(str)
    .str.strip()
)

# -----------------------------------
# 3. Get Farmer Inputs
# -----------------------------------

crop = input("Enter crop: ").strip().lower()
state = input("Enter state: ").strip()
year = int(input("Enter year: "))

area = float(input("Enter area in hectares: "))
predicted_yield = float(
    input("Enter predicted yield (tonnes/hectare): ")
)

# -----------------------------------
# 4. Find Market Price
# -----------------------------------

price_row = price_df[
    price_df["crop"] == crop
]

if price_row.empty:

    print("\n❌ Market price data unavailable for:", crop)
    exit()

price_per_tonne = price_row["price_per_tonne"].mean()

# -----------------------------------
# 5. Find Cultivation Cost
# -----------------------------------

state_rows = cost_df[
    cost_df["state"].str.lower() == state.lower()
]

crop_rows = state_rows[
    state_rows["crop"] == crop
]

# If exact state is unavailable,
# use crop-wise cost
if crop_rows.empty:

    crop_rows = cost_df[
        cost_df["crop"] == crop
    ]

if crop_rows.empty:

    print("\n❌ Cultivation cost data unavailable for:", crop)
    exit()

# -----------------------------------
# 6. Select Cost Year
# -----------------------------------

crop_rows = crop_rows.copy()

crop_rows["start_year"] = (
    crop_rows["year"]
    .astype(str)
    .str[:4]
)

crop_rows["start_year"] = pd.to_numeric(
    crop_rows["start_year"],
    errors="coerce"
)

# Prefer requested year
exact_year = crop_rows[
    crop_rows["start_year"] == year
]

if not exact_year.empty:

    selected_cost = exact_year[
        "cost_per_hectare"
    ].mean()

else:

    # Otherwise use latest available
    # cost before requested year
    previous_years = crop_rows[
        crop_rows["start_year"] <= year
    ]

    if not previous_years.empty:

        latest_year = previous_years[
            "start_year"
        ].max()

        selected_cost = previous_years[
            previous_years["start_year"] == latest_year
        ]["cost_per_hectare"].mean()

    else:

        selected_cost = crop_rows[
            "cost_per_hectare"
        ].mean()

# -----------------------------------
# 7. Calculate Production
# -----------------------------------

total_production = predicted_yield * area

# -----------------------------------
# 8. Calculate Revenue
# -----------------------------------

revenue = (
    total_production *
    price_per_tonne
)

# -----------------------------------
# 9. Calculate Total Cost
# -----------------------------------

total_cost = (
    selected_cost *
    area
)

# -----------------------------------
# 10. Calculate Profit
# -----------------------------------

profit = revenue - total_cost

# -----------------------------------
# 11. Display Results
# -----------------------------------

print("\n" + "=" * 45)
print("        CROP PROFIT ESTIMATION")
print("=" * 45)

print(f"Crop                 : {crop}")
print(f"State                : {state}")
print(f"Area                 : {area:.2f} hectares")
print(f"Predicted Yield      : {predicted_yield:.2f} tonnes/hectare")

print(f"\nMarket Price         : ₹{price_per_tonne:,.2f}/tonne")

print(f"Total Production     : {total_production:,.2f} tonnes")

print(f"Estimated Revenue    : ₹{revenue:,.2f}")

print(f"Cost per Hectare     : ₹{selected_cost:,.2f}")

print(f"Total Cultivation Cost: ₹{total_cost:,.2f}")

print("-" * 45)

print(f"ESTIMATED PROFIT     : ₹{profit:,.2f}")

print("=" * 45)