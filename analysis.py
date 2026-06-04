import numpy as np
import pandas as pd
import shap
from config import *

def feature_effects(model, X_test, df):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    impact = np.abs(shap_values.values).mean(axis=0)

    result = pd.DataFrame({
        "ویژگی": X_test.columns,
        "اثر کل": impact
    })

    avg_area = df[AREA_COL].mean()
    result["اثر هر متر"] = result["اثر کل"] / avg_area

    return result.sort_values("اثر هر متر", ascending=False)