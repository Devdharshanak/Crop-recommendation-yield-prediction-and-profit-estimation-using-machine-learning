


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("dataset/Crop_recommendation.csv")

print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())


#  Dataset Information

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


#  Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)


# Duplicate Values

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Shape After Removing Duplicates:", df.shape)


# Separate Features & Target

X = df.drop("label", axis=1)
y = df["label"]


# Encode Labels

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nEncoded Crop Labels:")

for crop, code in zip(encoder.classes_,
                      range(len(encoder.classes_))):
    print(f"{crop} --> {code}")


# Histogram

plt.figure(figsize=(12,8))
df.hist(figsize=(12,8))
plt.suptitle("Feature Distribution")
plt.show()


# Crop Distribution

plt.figure(figsize=(10,8))

sns.countplot(
    y=df["label"],
    order=df["label"].value_counts().index
)

plt.title("Crop Distribution")
plt.xlabel("Count")
plt.ylabel("Crop")

plt.show()


# Correlation Heatmap

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="YlGnBu"
)

plt.title("Correlation Heatmap")

plt.show()


# Boxplots

for column in X.columns:

    plt.figure(figsize=(6,4))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")

    plt.show()


# Final Output

print("\nFeatures:")
print(X.head())

print("\nEncoded Target:")
print(y_encoded[:10])

print("\nDataset is ready for Machine Learning!")

print("\nCompleted Tasks:")
print("✔ Dataset Loaded")
print("✔ Dataset Explored")
print("✔ Missing Values Checked")
print("✔ Duplicate Values Removed")
print("✔ Features & Labels Separated")
print("✔ Labels Encoded")
print("✔ Exploratory Data Analysis Completed")