import joblib

# Load the complete pipeline
pipeline = joblib.load("models/fraud_pipeline.pkl")

def predict_transaction(df):
    """
    Predict fraud or genuine transactions.
    """

    predictions = pipeline.predict(df)

    df["Prediction"] = predictions

    df["Prediction"] = df["Prediction"].map({
        0: "Genuine",
        1: "Fraud"
    })

    return df