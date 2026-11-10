import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "run_001.csv"

df = pd.read_csv(csv_path)

e = np.sqrt((df["x_ref"] - df["x"])**2 + (df["y_ref"] - df["y"])**2)

mask = df["t"] >= 10.0

print("max error after 10s:", e[mask].max())
print("final error:", e.iloc[-1])
