import pandas as pd
import numpy as np
from model import load_model, TOLERANCE
from config import FEATURES, AREA_COL


def get_yes_no(prompt):
    while True:
        val = input(f"  {prompt} (دارد/ندارد): ").strip()
        if val == "دارد":
            return 1
        elif val == "ندارد":
            return 0
        print("   لطفاً فقط 'دارد' یا 'ندارد' وارد کنید.")

def get_float(prompt):
    while True:
        try:
            return float(input(f"  {prompt}: ").strip())
        except ValueError:
            print("   لطفاً یک عدد معتبر وارد کنید.")

def get_int(prompt):
    while True:
        try:
            return int(input(f"  {prompt}: ").strip())
        except ValueError:
            print("   لطفاً یک عدد صحیح وارد کنید.")


FEATURE_PROMPTS = {
    "سال ساخت":       ("int",  "سال ساخت (مثلاً ۱۳۹۵)"),
    "اتاق":           ("int",  "تعداد اتاق"),
    "موقعیت طبقه":    ("int",  "موقعیت طبقه (مثلاً ۱، ۲، ۳...)"),
    "تعداد طبقات":    ("int",  "تعداد کل طبقات ساختمان"),
    "تعداد واحد ها":  ("int",  "تعداد واحدهای ساختمان"),
    "آسانسور":        ("bool", "آسانسور"),
    "پارکینگ":        ("bool", "پارکینگ"),
    "انباری":         ("bool", "انباری"),
}


def collect_input():
    print("    فرم پیش‌بینی قیمت خانه")

    area = get_float("متراژ خانه (متر مربع)")

    data = {}
    for feature in FEATURES:
        if feature not in FEATURE_PROMPTS:
            data[feature] = get_float(feature)
            continue
        dtype, prompt = FEATURE_PROMPTS[feature]
        if dtype == "float":
            data[feature] = get_float(prompt)
        elif dtype == "int":
            data[feature] = get_int(prompt)
        elif dtype == "bool":
            data[feature] = get_yes_no(prompt)

    return data, area


def predict_price():
    model = load_model()
    user_input, area = collect_input()

    input_df = pd.DataFrame([user_input])[FEATURES]

    
    price_per_m2 = model.predict(input_df)[0]
    total_price = price_per_m2 * area

    low  = total_price * (1 - TOLERANCE)
    high = total_price * (1 + TOLERANCE)

    print("    نتیجه پیش‌بینی:")
    print(f"   متراژ             : {area:,.0f} متر مربع")
    print(f"   قیمت هر متر مربع : {price_per_m2:,.1f} میلیون تومان")
    print(f"   قیمت کل ملک      : {total_price:,.1f} میلیون تومان")
    print(f"   معادل             : {total_price / 1000:,.2f} میلیارد تومان")

    return price_per_m2


if __name__ == "__main__":
    predict_price()
