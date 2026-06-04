from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from config import *
import numpy as np
import matplotlib.pyplot as plt


def run_with_error_switch(y_true, y_pred, threshold_percent=80):
    mode = "X"
    x_results = []
    y_results = []

    for i in range(len(y_true)):
        if y_true[i] == 0:
            continue
            
        error_percent = (abs(y_true[i] - y_pred[i]) / y_true[i]) * 100

        if mode == "X":
            x_results.append((i, error_percent))
            if error_percent > threshold_percent:
                mode = "Y"
        elif mode == "Y":
            y_results.append((i, error_percent))
            if error_percent > threshold_percent:
                break

    return x_results, y_results


def plot_error_phases(x_phase, y_phase, threshold):
    """تابع مخصوص رسم نمودار خطای فاز X و Y"""
    plt.figure(figsize=(12, 6))

    x_indices = [item[0] for item in x_phase]
    x_errors = [item[1] for item in x_phase]

    y_indices = [item[0] for item in y_phase]
    y_errors = [item[1] for item in y_phase]

    plt.plot(x_indices, x_errors, label="Phase X (Error < 80%)", color="blue", marker='o', markersize=4)
    
    plt.plot(y_indices, y_errors, label="Phase Y (Switched)", color="red", marker='o', markersize=4)

    plt.axhline(y=threshold, color="black", linestyle="--", label=f"Threshold ({threshold}%)")

    if x_phase:
        plt.scatter(x_indices[-1], x_errors[-1], color="orange", s=150, zorder=5, label="Switch Point")


    plt.title("Model Error Analysis - Phase Switch (X to Y)", fontsize=14)
    plt.xlabel("Data Index", fontsize=12)
    plt.ylabel("Error Percentage (%)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.show()


def train_model(df):
    X = df[FEATURES]
    y = df["قیمت"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    threshold = 80
    x_phase, y_phase = run_with_error_switch(
        y_test.values,
        y_pred,
        threshold_percent=threshold
    )

    plot_error_phases(x_phase, y_phase, threshold)

    return model, X_test