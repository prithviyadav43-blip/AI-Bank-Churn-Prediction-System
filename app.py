from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DATASET PATH
dataset = os.path.join(
    BASE_DIR,
    "dataset",
    "European_Bank.csv"
)

# MODEL PATH
model_path = os.path.join(
    BASE_DIR,
    "models",
    "churn_model.pkl"
)

# Load Data
dataframe = pd.read_csv("European_Bank.csv")
# Load Model
model = joblib.load("models/churn_model.pkl")


# HOME PAGE
@app.route('/accuracy')
def accuracy():
    return render_template("accuracy.html")

@app.route('/')
def home():

    customers = dataframe.head(10).to_dict(orient='records')

    total_customers = len(dataframe)

    total_churn = dataframe['Exited'].sum()

    churn_rate = round(
        (total_churn / total_customers) * 100,
        2
    )

    avg_credit = round(
        dataframe['CreditScore'].mean(),
        2
    )

    avg_balance = round(
        dataframe['Balance'].mean(),
        2
    )

    young = len(
        dataframe[dataframe['Age'] < 30]
    )

    middle = len(
        dataframe[
            (dataframe['Age'] >= 30)
            &
            (dataframe['Age'] < 50)
            ]
    )

    senior = len(
        dataframe[dataframe['Age'] >= 50]
    )

    # Geography Wise Churn
    geography_churn = dataframe.groupby(
        'Geography'
    )['Exited'].sum().tolist()

    geography_labels = dataframe['Geography'].unique().tolist()

    # Credit Card Wise Churn
    card_churn = dataframe.groupby(
        'HasCrCard'
    )['Exited'].sum().tolist()

    # Active Member Churn
    active_churn = dataframe.groupby(
        'IsActiveMember'
    )['Exited'].sum().tolist()

    gender_churn = dataframe.groupby(
        'Gender'
    )['Exited'].sum().tolist()

    high_value = len(
        dataframe[dataframe['Balance'] > 100000]
    )

    medium_value = len(
        dataframe[
            (dataframe['Balance'] > 50000)
        ]
    )

    low_value = len(
        dataframe[dataframe['Balance'] < 50000]
    )

    return render_template(

        "index.html",

        customers=customers,

        total_customers=total_customers,
        churn_rate=churn_rate,
        avg_credit=avg_credit,
        avg_balance=avg_balance,

        geography_churn=geography_churn,
        geography_labels=geography_labels,

        card_churn=card_churn,

        active_churn=active_churn,

        gender_churn=gender_churn,

        young=young,
        middle=middle,
        senior=senior,


        high_value=high_value,
        medium_value=medium_value,
        low_value=low_value


    )


# MANUAL PAGE
@app.route('/manual')
def manual():
    return render_template(
        "manual.html",
        prediction=result,
        percentage=percentage,
        message=message
    )


# AUTO PAGE
@app.route('/auto')
def auto():
    return render_template("auto.html")


# LOGIN PAGE
@app.route('/login')
def login():
    return render_template("login.html")


# MANUAL PREDICTION
@app.route('/predict', methods=['POST'])
def predict():

    creditscore = int(request.form['creditscore'])
    age = int(request.form['age'])
    balance = float(request.form['balance'])
    salary = float(request.form['salary'])
    creditcard = int(request.form['creditcard'])

    activemember = int(request.form['activemember'])

    data = np.array([[
        2023,
        creditscore,
        1,
        1,
        age,
        5,
        balance,
        2,
        creditcard,
        activemember,
        salary
    ]])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    percentage = round(probability * 100, 2)

    if prediction[0] == 1:

        result = "⚠ High Churn Risk"

        message = """
        Customer is willing to leave the bank.
        Immediate retention strategy is recommended.
        """

    else:

        result = "✅ Customer Safe"

        message = """
        Customer is likely to stay with the bank.
        Relationship status is stable.
        """

    return render_template(
        "manual.html",
        prediction=result,
        percentage=percentage
    )


# AUTO SEARCH
@app.route('/search', methods=['POST'])
def search():

    customer_id = int(request.form['customerid'])

    customer = dataframe[
        dataframe['CustomerId'] == customer_id
    ]

    if customer.empty:

        return render_template(
            "auto.html",
            result="❌ Customer Not Found"
        )

    row = customer.iloc[0]

    # AI Model Data
    model_data = np.array([[
        row['Year'],
        row['CreditScore'],
        1,
        1,
        row['Age'],
        row['Tenure'],
        row['Balance'],
        row['NumOfProducts'],
        row['HasCrCard'],
        row['IsActiveMember'],
        row['EstimatedSalary']
    ]])

    prediction = model.predict(model_data)

    if prediction[0] == 1:
        churn = "⚠ High Churn Risk"

    else:
        churn = "✅ Customer Safe"

    details = f"""
    Name: {row['Surname']}
    | Age: {row['Age']}
    | Country: {row['Geography']}
    | Balance: {row['Balance']}
    | Salary: {row['EstimatedSalary']}
    """

    return render_template(
        "auto.html",
        result=churn,
        details=details
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=10000)