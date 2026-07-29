from pathlib import Path
from itertools import product
import pandas as pd

comp_alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ce_alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cty_alpha_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

all_dfs = []

for state in ["co", "tn", "pa", "ut", "ma"]:
    for comp_alpha, ce_alpha, cty_alpha, num in product(
        comp_alpha_range,
        ce_alpha_range,
        cty_alpha_range,
        [1, 2],
    ):
        base = Path("Output")
        folders = base.glob(
            f"{state}_{comp_alpha}-{ce_alpha}-{cty_alpha}_*_{num}"
        )
        for folder in folders:
            files = sorted(
                folder.glob(f"ensemble_{num}_DemPercs_*.csv"),
                key=lambda f: int(f.stem.split("_")[-1]),
            )

            if not files:
                print(f"No files found in {folder}")
                continue

            for file in files:
                try:
                    all_dfs.append(pd.read_csv(file))
                except pd.errors.EmptyDataError:
                    print(f"Empty CSV: {file}")
all_dem_percs = pd.concat(all_dfs, ignore_index=True)

destination_df = pd.read_csv("Output/cube_df.csv")
destination_df["Lowest_DP"] = all_dem_percs.min(axis=1)
destination_df.to_csv("Output/cube_df.csv", index=False)

print(f"Saved {len(all_dem_percs):,} rows.")
