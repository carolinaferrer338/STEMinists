from pathlib import Path
import pandas as pd

comp_alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ce_alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cty_alpha_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
all_dfs = []

for state in ['co', 'tn', 'pa', 'ut']:
    for comp_alpha in comp_alpha_range:
        for ce_alpha in ce_alpha_range:
            for cty_alpha in cty_alpha_range:
                for num in [1, 2]:

                    folder = Path(
                        f"STEMinists/Output/{state}_{comp_alpha}-{ce_alpha}-{cty_alpha}_200000_{num}"
                    )

                    files = sorted(
                        folder.glob(f"ensemble_{num}chain_outputs_*.csv"),
                        key=lambda f: int(f.stem.split("_")[-1])
                    )

                    if not files:
                        print(f"No files found in {folder}")
                        continue

                    dfs = []

                    for file in files:
                        df = pd.read_csv(file)

                        step = int(file.stem.split("_")[-1])

                        df["state"] = state
                        df["comp_alpha"] = comp_alpha
                        df["ce_alpha"] = ce_alpha
                        df["cty_alpha"] = cty_alpha
                        df["ensemble"] = num
                        df["step"] = step

                        dfs.append(df)

                    df = pd.concat(dfs, ignore_index=True)

                    if "CES" in df.columns:
                        df["CES"] = df["CES"].apply(len)

                    all_dfs.append(df)

cube_df = pd.concat(all_dfs, ignore_index=True)
cube_df.to_csv("STEMinists/Output/cube_df.csv", index=False)

print(f"Saved {len(cube_df):,} rows.")
