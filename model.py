import numpy as np
import joblib
import os
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import FEATURES

MODEL_PATH = "models/xgb_model.joblib"
TOLERANCE = 0.20



def run_with_error_switch(y_true, y_pred, threshold_percent=20):
    
    mode = "X"
    x_results = []
    y_results = []
    passed = True

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
                passed = False
                break

    return x_results, y_results, passed



def train_model(df, save=True):
    X = df[FEATURES]
    y = df["قیمت"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="mae"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    y_true = y_test.values


    errors = np.abs(y_true - y_pred) / np.where(y_true == 0, 1, y_true) * 100
    sorted_idx = np.argsort(errors)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    x_phase, y_phase, passed = run_with_error_switch(y_true_sorted, y_pred_sorted, threshold_percent=20)
    acc = (errors < 20).mean() * 100

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    print("نتایج ارزیابی مدل")
    print(f"   R²                    : {r2:.4f}")
    print(f"   RMSE                  : {rmse:,.2f} میلیون تومان")
    print(f"   MAE                   : {mae:,.2f} میلیون تومان")
    print(f"   دقت (خطا زیر ۲۰٪)    : {acc:.1f}%")
    print(f"   نمونه‌های فاز X       : {len(x_phase)}")
    print(f"   نمونه‌های فاز Y       : {len(y_phase)}")
    if passed:
        print("    مدل تولرانس ۲۰٪ را پاس کرد!")
    else:
        print("     مدل در فاز Y دو خطای پشت سر هم داشت")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    model.shap_feature_weights_ = dict(zip(FEATURES, shap_importance))

    if save:
        save_model(model)

    return model



def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"مدل ذخیره شد ← {path}")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "مدل آموزش‌دیده‌ای پیدا نشد.\n"
            "   ابتدا train.py را اجرا کنید."
        )
    return joblib.load(path)
