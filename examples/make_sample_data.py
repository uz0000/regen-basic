"""
Write a small demo table — a stand-in for "some real table you have" — so the
examples and the CLI have something to run against. This is *input* to the
simulator, not its output.

Deliberately covers every column kind the simulator has to handle, since a
table of nothing but floats would exercise about half the code:

  station_id   identifier  — near-unique key (re-minted, never sampled)
  region       categorical — a handful of repeated labels
  temp_c       continuous  — correlated with humidity, so there is real joint
  humidity     continuous    structure for the simulator to preserve or lose
  hour         continuous  — integer-valued, must come back as an integer
  sensor_fault binary      — rare-ish flag, correlated with humidity
"""

import argparse

import numpy as np
import pandas as pd


def make_dataset(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "coastal", "inland"], size=n,
                        p=[0.3, 0.3, 0.25, 0.15])
    # Baseline temperature shifts by region, so region and temp_c are related —
    # a categorical/continuous relationship, the kind a per-column sampler
    # silently drops.
    base = np.select(
        [region == "north", region == "south", region == "coastal"],
        [8.0, 22.0, 15.0], default=18.0,
    )
    temp_c = base + rng.normal(0, 4, n)
    # Humidity falls as temperature rises: a strong negative correlation the
    # simulator is expected to reproduce.
    humidity = (95.0 - 1.6 * temp_c + rng.normal(0, 6, n)).clip(5, 100)
    hour = rng.integers(0, 24, n).astype(float)
    # Faults concentrate at high humidity rather than occurring at random.
    fault_p = 1.0 / (1.0 + np.exp(-(humidity - 80.0) / 5.0))
    sensor_fault = (rng.uniform(size=n) < fault_p).astype(int)

    return pd.DataFrame({
        "station_id": np.arange(1, n + 1),
        "region": region,
        "temp_c": temp_c,
        "humidity": humidity,
        "hour": hour,
        "sensor_fault": sensor_fault,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/readings.csv")
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()

    df = make_dataset(n=args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows → {args.out}")
