import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/inference/test.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--out", default="data/predictions/Prediccion_Equipo2.csv")
    parser.add_argument("--date_block_num", type=int, default=34)
    args = parser.parse_args()

    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)

    model = joblib.load(args.model)

    test["date_block_num"] = args.date_block_num
    X_test = test[["date_block_num", "shop_id", "item_id"]]

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 20)

    submission = pd.DataFrame({"ID": sample["ID"], "item_cnt_month": preds})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print(f"Predicciones guardadas en: {out_path}  (n={len(submission)})")


if __name__ == "__main__":
    main()
