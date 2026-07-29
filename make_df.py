from pathlib import Path
import pandas as pd

destination_df = pd.read_csv("Output/cube_df.csv")

lowest_dps = []

for _, row in destination_df.iterrows():
    state = row["state"]
    comp_alpha = row["comp_alpha"]
    ce_alpha = row["ce_alpha"]
    cty_alpha = row["cty_alpha"]
    ensemble = row["ensemble"]
    step = int(row["step"])

    folders = list(
        Path("Output").glob(
            f"{state}_{comp_alpha}-{ce_alpha}-{cty_alpha}_*_{ensemble}"
        )
    )

    if len(folders) != 1:
        lowest_dps.append(pd.NA)
        continue

    file = folders[0] / f"ensemble_{ensemble}_DemPercs_{step}.csv"

    if not file.exists():
        lowest_dps.append(pd.NA)
        continue

    try:
        df = pd.read_csv(file)

        if step < len(df):
            lowest_dps.append(df.iloc[step].min())
        else:
            lowest_dps.append(pd.NA)

    except pd.errors.EmptyDataError:
        lowest_dps.append(pd.NA)

destination_df["Lowest_DP"] = lowest_dps
destination_df.to_csv("Output/cube_df.csv", index=False)
