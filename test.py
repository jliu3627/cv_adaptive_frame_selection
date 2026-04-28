import pandas as pd
import glob
import os

csvs = glob.glob("data/interim/labels_gt/*.csv")
for csv in csvs:
    df = pd.read_csv(csv)
    print(os.path.basename(csv))
    print(df["label"].value_counts(normalize=True))