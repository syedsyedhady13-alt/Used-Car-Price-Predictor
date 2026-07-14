# ==========================================================
# Used Car Price Predictor
# Interactive Prediction Program
#
# Author: Syed Abdul Hadi
# ==========================================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================================
# Banner
# ==========================================================

print("=" * 65)
print("               USED CAR PRICE PREDICTOR")
print("=" * 65)
print("Loading saved files...\n")

# ==========================================================
# Required Files
# ==========================================================

required_files = {
    "Gradient Boosting Model": "gradient_boosting_model.pkl",
    "Ridge Regression Model": "ridge_model.pkl",
    "Lasso Regression Model": "lasso_model.pkl",
    "Scaler": "scaler.pkl",
    "Feature Columns": "feature_columns.pkl",
    "Top Models": "top_models.pkl",
    "Categorical Columns": "categorical_columns.pkl"
}

loaded = {}

for name, filename in required_files.items():

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"\nERROR: {filename} was not found."
        )

    loaded[name] = joblib.load(filename)
    print(f"✓ {name} loaded successfully.")

print("\nAll required files loaded successfully.")
print("=" * 65)

# ==========================================================
# Assign Variables
# ==========================================================

gbr_model = loaded["Gradient Boosting Model"]
ridge_model = loaded["Ridge Regression Model"]
lasso_model = loaded["Lasso Regression Model"]

scaler = loaded["Scaler"]

feature_columns = loaded["Feature Columns"]

top_models = loaded["Top Models"]

categorical_columns = loaded["Categorical Columns"]

# ==========================================================
# Helper Functions
# ==========================================================

def get_string(prompt):

    while True:

        value = input(prompt).strip()

        if value != "":
            return value

        print("Input cannot be empty.\n")


def get_integer(prompt):

    while True:

        try:
            return int(input(prompt))

        except ValueError:

            print("Please enter a valid integer.\n")


def get_float(prompt):

    while True:

        try:
            return float(input(prompt))

        except ValueError:

            print("Please enter a valid number.\n")

# ==========================================================
# Model Selection
# ==========================================================

print("\nAvailable Models\n")

print("1. Gradient Boosting Regression")
print("2. Ridge Regression")
print("3. Lasso Regression")

while True:

    choice = input("\nChoose model (1-3): ").strip()

    if choice == "1":
        selected_model = gbr_model
        selected_model_name = "Gradient Boosting Regression"
        break

    elif choice == "2":
        selected_model = ridge_model
        selected_model_name = "Ridge Regression"
        break

    elif choice == "3":
        selected_model = lasso_model
        selected_model_name = "Lasso Regression"
        break

    else:

        print("Invalid choice.\n")

print(f"\nSelected Model : {selected_model_name}")

print("=" * 65)
# ==========================================================
# Prediction Loop
# ==========================================================

while True:

    print("\n" + "=" * 65)
    print("ENTER VEHICLE INFORMATION")
    print("=" * 65)

    # ----------------------------
    # Numerical Features
    # ----------------------------

    condition = get_float("Condition: ")
    odometer = get_float("Odometer: ")
    mmr = get_float("MMR Value: ")
    sale_year = get_integer("Sale Year: ")

    # ----------------------------
    # Categorical Features
    # ----------------------------

    company = get_string("Company: ")
    model = get_string("Model: ")
    size = get_string("Vehicle Size: ")
    transmission = get_string("Transmission: ")
    state = get_string("State: ")
    color = get_string("Exterior Color: ")
    interior = get_string("Interior Color: ")

    sale_day = get_string("Sale Day: ")
    sale_month = get_string("Sale Month: ")

    # ---------------------------------
    # Convert rare models into "Other"
    # ---------------------------------

    if model not in top_models:
        model = "Other"

    # ---------------------------------
    # Create DataFrame
    # ---------------------------------

    user_df = pd.DataFrame({

        "condition": [condition],
        "odometer": [odometer],
        "mmr": [mmr],
        "Sale year": [sale_year],

        "COMPANY": [company],
        "MODEL": [model],
        "SIZE": [size],
        "transmission": [transmission],
        "state": [state],
        "color": [color],
        "interior": [interior],
        "sale Day": [sale_day],
        "Sale month": [sale_month]

    })

    # ---------------------------------
    # One-Hot Encoding
    # ---------------------------------

    user_df = pd.get_dummies(

        user_df,

        columns=categorical_columns,

        drop_first=True,

        dtype=int

    )

    # ---------------------------------
    # Match Training Columns
    # ---------------------------------

    user_df = user_df.reindex(

        columns=feature_columns,

        fill_value=0

    )

    # ---------------------------------
    # Scaling
    # ---------------------------------

    if selected_model_name in [

        "Ridge Regression",

        "Lasso Regression"

    ]:

        prediction_data = scaler.transform(user_df)

    else:

        prediction_data = user_df
    # ==========================================================
    # Prediction
    # ==========================================================

    prediction = selected_model.predict(prediction_data)

    predicted_price = float(prediction[0])

    print("\n" + "=" * 65)
    print("PREDICTION RESULT")
    print("=" * 65)

    print(f"Selected Model : {selected_model_name}")
    print(f"Estimated Selling Price : ${predicted_price:,.2f}")

    print("=" * 65)

    # ==========================================================
    # Another Prediction?
    # ==========================================================

    while True:

        again = input("\nWould you like to predict another car? (Y/N): ").strip().lower()

        if again in ["y", "yes"]:

            break

        elif again in ["n", "no"]:

            print("\nThank you for using the Used Car Price Predictor!")
            print("Goodbye!")

            exit()

        else:

            print("Please enter Y or N.")