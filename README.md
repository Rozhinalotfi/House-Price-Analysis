# House Price Analysis
A machine learning project that analyzes house prices based on different property features such as number of rooms, parking, and elevator availability. It uses XGBoost for feature importance and Matplotlib for visualization.


#  House Price Prediction

A machine learning project for predicting house prices per square meter using XGBoost, built for small real estate datasets.

---

##  Project Structure

```
House Price Analysis/
├── data/
│   └── housedata.xlsx       # Training dataset
├── models/
│   └── xgb_model.joblib     # Saved model (auto-generated after training)
├── config.py                # Feature names and column config
├── preprocess.py            # Data loading and cleaning
├── model.py                 # Training, evaluation, save/load
├── predict.py               # Interactive price prediction (manual input)
├── train.py                 # Entry point for training
└── requirements.txt
```

---

##  Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install xgboost scikit-learn pandas numpy openpyxl shap joblib
```

---

##  Usage

### 1. Train the model
Run this once before predicting:
```bash
python train.py
```

Output example:
```
 بارگذاری داده‌ها...
 420 ردیف بارگذاری شد.
 شروع آموزش مدل...

 نتایج ارزیابی مدل
   R²                    : 0.8741
   RMSE                  : 3.21 میلیون تومان
   MAE                   : 2.10 میلیون تومان
   دقت (خطا زیر ۲۰٪)    : 83.4%
   نمونه‌های فاز X       : 68
   نمونه‌های فاز Y       : 2
    مدل تولرانس ۲۰٪ را پاس کرد!

 مدل ذخیره شد ← models/xgb_model.joblib
```

### 2. Predict a price
```bash
python predict.py
```

You will be prompted to enter house details:
```

    فرم پیش‌بینی قیمت خانه

  متراژ خانه (متر مربع): 90
  سال ساخت (مثلاً ۱۳۹۵): 1398
  تعداد اتاق: 2
  موقعیت طبقه (مثلاً ۱، ۲، ۳...): 3
  تعداد کل طبقات ساختمان: 6
  تعداد واحدهای ساختمان: 12
  آسانسور (دارد/ندارد): دارد
  پارکینگ (دارد/ندارد): دارد
  انباری (دارد/ندارد): ندارد


    نتیجه پیش‌بینی:
   متراژ             : 90 متر مربع
   قیمت هر متر مربع : 36.4 میلیون تومان
   قیمت کل ملک      : 3,276.0 میلیون تومان
   معادل             : 3.28 میلیارد تومان

```

---

##  Dataset Format

The training file `data/housedata.xlsx` must contain these columns:

| Column | Type | Example |
|--------|------|---------|
| قیمت متراژ | float | 36 *(million tomans per m²)* |
| متراژ | float | 90 |
| سال ساخت | int | 1398 |
| اتاق | int | 2 |
| موقعیت طبقه | int | 3 |
| تعداد طبقات | int | 6 |
| تعداد واحد ها | int | 12 |
| آسانسور | str | دارد / ندارد |
| پارکینگ | str | دارد / ندارد |
| انباری | str | دارد / ندارد |

> **Note:** Prices are stored as million tomans per square meter (e.g. `36` = 36 million tomans/m²), not full prices.

---

##  How It Works

| Step | Description |
|------|-------------|
| **Preprocessing** | Loads Excel data, converts boolean columns, removes outliers via IQR |
| **Training** | XGBoost regressor with regularization tuned for small datasets |
| **SHAP** | Computed silently after training to understand feature impact |
| **Evaluation** | Uses a custom X/Y phase-switch logic — model passes if it doesn't hit two consecutive errors above 20% |
| **Prediction** | Predicts price per m², multiplies by area, shows ±20% confidence range |

### X/Y Phase Switch Logic
After training, predictions on the test set are sorted by error (low to high):
- **Phase X** — predictions with error under 20% (the good ones)
- **Switch** — first prediction that exceeds 20% error
- **Phase Y** — one more chance; if error exceeds 20% again → model fails

This ensures at least 80% of predictions stay within the 20% tolerance band.

---

##  Dependencies

| Package | Purpose |
|---------|---------|
| xgboost | Core prediction model |
| scikit-learn | Train/test split, metrics |
| pandas | Data handling |
| numpy | Numerical operations |
| openpyxl | Reading Excel files |
| shap | Feature importance (silent) |
| joblib | Saving and loading the model |