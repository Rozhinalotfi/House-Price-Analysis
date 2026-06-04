from preprocess import load_data
from model import train_model
from analysis import feature_effects

df = load_data()

model, X_test = train_model(df)

result = feature_effects(model, X_test, df)

print("\n اثر هر ویژگی روی قیمت هر متر:")
print(result)