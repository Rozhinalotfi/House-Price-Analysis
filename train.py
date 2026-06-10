from preprocess import load_data
from model import train_model

if __name__ == "__main__":
    print(" بارگذاری داده‌ها...")
    df = load_data()
    print(f" {len(df)} ردیف بارگذاری شد.")

    print("\n شروع آموزش مدل...")
    train_model(df)
