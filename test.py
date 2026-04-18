import pandas as pd

df = pd.read_csv("data/interim/labels/MOT17-10-SDP_labels.csv")
print(df["label"].value_counts(normalize=True))