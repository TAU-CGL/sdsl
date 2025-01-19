import os
import json
import sqlite3
import itertools
import subprocess

import tqdm
import pandas as pd

from common import *


KS = range(4, 33, 2)
NS = [1, 2, 4, 8, 16, 32, 64]
OUTPUT_DB = os.path.join(RESULTS_DIR, "exp_k_k_prime_uniform.db")


def get_available_maps():
    return [f for f in os.listdir(MAPS_DIR) if f.endswith(".poly")]

def get_combinations():
    return itertools.product(get_available_maps(), KS, NS)

def get_empty_df():
    results_df = pd.DataFrame(columns=["map", "n", "k", "k_", "seed"])

def run_experiment(map, k, n):
    metadata = {
            "map": map,
            "n": n,
            "k": k,
            "seed": SEED
        }
    df = get_empty_df()
    result = subprocess.run(
        [
            "python", "experiments/exp_k_k_prime_uniform.py",
            "--map", os.path.join(MAPS_DIR, map),
            "--k", str(k),
            "--obst-num", str(n),
            "--seed", str(SEED),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        return None
    output_data = json.loads(result.stdout)

    for k_ in output_data["k_"]:
        row = {
            **metadata,
            "k_": k_
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

if __name__ == "__main__":
    conn = sqlite3.connect(OUTPUT_DB)
    for map, k, n in tqdm.tqdm(list(get_combinations())):
        print(map, k, n)
        df = run_experiment(map, k, n)
        if df is not None:
            df.to_sql("results", conn, if_exists="append", index=False)
    conn.close()