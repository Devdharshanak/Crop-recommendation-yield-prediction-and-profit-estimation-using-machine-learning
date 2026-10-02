import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib



df = pd.read_csv("dataset/Crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)



X = df.drop("label", axis=1)
y = df["label"]



encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])




model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")



y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")



print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)



joblib.dump(model, "crop_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel saved as: crop_model.pkl")
print("Label encoder saved as: label_encoder.pkl")

print("\nMachine Learning Training Completed Successfully!")
