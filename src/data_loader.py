import os
import pandas as pd

def list_csvs(folder: str):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                out.append(os.path.join(root, f))
    return out

def load_csv(path: str):
    return pd.read_csv(path)

def pick_largest_csv(paths):
    return sorted(paths, key=lambda p: os.path.getsize(p), reverse=True)[0]
