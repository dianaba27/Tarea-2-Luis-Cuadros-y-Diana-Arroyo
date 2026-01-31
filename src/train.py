import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def split_train_val(df: pd.DataFrame, val_block: int = 33):
    train_df = df[df["date_block_num"] < val_block].copy()
    val_df = df[df["date_block_num"] == val_block].copy()

    X_train = train_df.drop(columns=["item_cnt_month"])
    y_train = train_df["item_cnt_month"]

    X_val = val_df.drop(columns=["item_cnt_month"])
    y_val = val_df["item_cnt_month"]

    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prep/monthly_sales.csv")
    parser.add_argument("--model_out", default="artifacts/model.joblib")
    parser.add_argument("--val_block", type=int, default=33)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    X_train, y_train, X_val, y_val = split_train_val(df, val_block=args.val_block)

    # Parámetros
    model = xgb.XGBRegressor(
        max_depth=8,
        n_estimators=100,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        learning_rate=0.3,
        seed=53,
        eval_metric="rmse",
        early_stopping_rounds=10,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds_val = model.predict(X_val).clip(0, 20)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))
    print(f"[RMSE val] {rmse:.4f}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    print(f"Modelo guardado en: {model_out}")


if __name__ == "__main__":
    main()
