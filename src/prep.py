import argparse
from pathlib import Path
import pandas as pd


def build_monthly_sales(sales_path: Path) -> pd.DataFrame:
    sales = pd.read_csv(sales_path)

    # Agregación mensual
    monthly = (
        sales.groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["date_block_num", "shop_id", "item_id", "item_cnt_month"]

    monthly["item_cnt_month"] = monthly["item_cnt_month"].clip(0, 20)
    return monthly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="data/raw/sales_train.csv")
    parser.add_argument("--out", default="data/prep/monthly_sales.csv")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    monthly = build_monthly_sales(Path(args.sales))
    monthly.to_csv(out_path, index=False)

    print(f"Guardado: {out_path}  (shape={monthly.shape})")


if __name__ == "__main__":
    main()
