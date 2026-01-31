# Tarea 2. Notebook a Scripts (XGBoost)

Este repositorio contiene el refactor de un notebook de Machine Learning a un flujo reproducible en scripts de Python, usando el framework uv para manejar el entorno y dependencias.

## Estructura del repositorio

```text
.
├── notebooks/
│   └── ML_Model.ipynb
├── src/
│   ├── __init__.py
│   ├── prep.py
│   ├── train.py
│   └── inference.py
├── data/
│   ├── raw/
│   │   ├── sales_train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   ├── prep/
│   │   └── monthly_sales.csv
│   ├── inference/
│   │   └── test.csv
│   └── predictions/
│       └── Prediccion_Equipo2.csv
├── artifacts/
│   └── model.joblib
├── pyproject.toml
└── uv.lock
