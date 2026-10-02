import pandas as pd

price_df = pd.read_csv("dataset/daily_price.csv")

missing_crops = [
    "blackgram",
    "chickpea",
    "kidneybeans",
    "lentil",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "pigeonpeas",
    "watermelon"
]

commodities = price_df["Commodity"].dropna().unique()

print("\n" + "=" * 60)
print("       CHECKING POSSIBLE CROP NAME MATCHES")
print("=" * 60)

for crop in missing_crops:
    print(f"\n🔎 {crop}")

    # Find commodities containing part of the crop name
    matches = [
        commodity
        for commodity in commodities
        if any(word in str(commodity).lower()
               for word in crop.lower().split())
    ]

    if matches:
        for match in matches[:10]:
            print("   →", match)
    else:
        print("   ❌ No possible match found")

print("\n" + "=" * 60)