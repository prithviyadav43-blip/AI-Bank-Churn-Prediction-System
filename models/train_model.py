import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Dataset Load
data = pd.read_csv(r"C:\Users\PRITHVI\OneDrive\Banking Ai Dashboard\dataset\European_Bank.csv")

# Remove columns
data = data.drop(["CustomerId", "Surname"], axis=1)

# Convert text into numbers
label = LabelEncoder()

data["Geography"] = label.fit_transform(data["Geography"])
data["Gender"] = label.fit_transform(data["Gender"])

# Input Output
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Train Model
model = RandomForestClassifier()

model.fit(X, y)

# Save Model
joblib.dump(model, "churn_model.pkl")

print("AI Model Trained Successfully")