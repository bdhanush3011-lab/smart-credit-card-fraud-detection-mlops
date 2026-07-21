from datetime import datetime
import os

from flask import Flask, render_template, request, send_file
import pandas as pd

from src.predict import predict_transaction


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file selected"


    file = request.files["file"]

    df = pd.read_csv(file)


    # Remove target column if dataset contains it
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)



    # Prediction
    result = predict_transaction(df.copy())


    # Save predictions
    result.to_csv(
        "predictions.csv",
        index=False
    )



    total = len(result)

    fraud = (result["Prediction"] == "Fraud").sum()

    genuine = (result["Prediction"] == "Genuine").sum()


    fraud_rate = round(
        (fraud / total) * 100,
        2
    )



    # Save prediction history

    history_data = pd.DataFrame([{

        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "Total Transactions": total,

        "Fraud": fraud,

        "Genuine": genuine,

        "Fraud Rate": fraud_rate

    }])



    if os.path.exists("history.csv"):

        old_history = pd.read_csv("history.csv")

        history_data = pd.concat(
            [old_history, history_data],
            ignore_index=True
        )



    history_data.to_csv(
        "history.csv",
        index=False
    )



    table = result.head(20).to_html(
        classes="table table-striped table-hover",
        index=False
    )



    return render_template(
        "index.html",
        table=table,
        total=total,
        fraud=fraud,
        genuine=genuine,
        fraud_rate=fraud_rate
    )



@app.route("/download")
def download():

    return send_file(
        "predictions.csv",
        as_attachment=True
    )



@app.route("/history")
def history():

    if os.path.exists("history.csv") and os.path.getsize("history.csv") > 0:

        data = pd.read_csv("history.csv")

        table = data.to_html(
            classes="table table-striped",
            index=False
        )

        return render_template(
            "history.html",
            table=table
        )


    return "No history available"


if __name__ == "__main__":
    app.run(debug=True)