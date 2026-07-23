import pandas as pd
meta = pd.read_csv("data/echonext_metadata_100k.csv")
val_meta = meta[meta["split"] == "val"]

# All 12 labels are 0
label_cols = [c for c in val_meta.columns if "flag" in c]
perfectly_normal = val_meta[(val_meta[label_cols] == 0).all(axis=1)]
print(f"Perfectly normal: {len(perfectly_normal)} / {len(val_meta)}")