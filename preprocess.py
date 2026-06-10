import pandas as pd
import numpy as np
from config import PRICE_PER_M2_COL, AREA_COL, FEATURES

def convert_bool(value):
    if value == "دارد":
        return 1
    elif value == "ندارد":
        return 0
    return 0

def remove_outliers(df, col):

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

def load_data():
    df = pd.read_excel("data/housedata.xlsx")

    df["آسانسور"] = df["آسانسور"].apply(convert_bool)
    df["پارکینگ"]  = df["پارکینگ"].apply(convert_bool)
    df["انباری"]   = df["انباری"].apply(convert_bool)

    df["قیمت"] = df[PRICE_PER_M2_COL] 

    df = df.dropna(subset=FEATURES + ["قیمت"])

    df = remove_outliers(df, "قیمت")
    df = remove_outliers(df, AREA_COL)

    print(f" {len(df)} ردیف بعد از پاک‌سازی باقی ماند.")
    return df
