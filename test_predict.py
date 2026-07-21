import pandas as pd
from src.predict import predict_transaction

# Load sample data
df = pd.read_csv("data/creditcard.csv")

# Remove the target column
sample = df.drop("Class", axis=1).head(10)

# Predict
result = predict_transaction(sample)

print(result[["Time", "Amount", "Prediction"]])