import pandas as pd
from config import *

def convert_bool(value):
    if value == "دارد":
        return 1
    elif value == "ندارد":
        return 0
    return 0

def load_data():
    df = pd.read_excel("data/YOUR_DATABASE_NAME")

    
    df["آسانسور"] = df["آسانسور"].apply(convert_bool)
    df["پارکینگ"] = df["پارکینگ"].apply(convert_bool)
    df["انباری"] = df["انباری"].apply(convert_bool)

    
    df["قیمت"] = df[PRICE_PER_M2_COL] * df[AREA_COL]

    return df