# %%
# §0 — Setup (already cloned IntroECG manually, so just wire up the path)
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "IntroECG/7-EchoNext Minimodel"))

print("cradlenet exists:", os.path.exists(os.path.join(PROJECT_ROOT, "IntroECG/7-EchoNext Minimodel/cradlenet")))
#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
# %%
meta = pd.read_csv(os.path.join(PROJECT_ROOT, "data/echonext_metadata_100k.csv"))
print(meta.shape)
print(meta.columns.tolist())
print(meta.head(3))
# %%
val_meta = meta[meta['split'] == 'val'].reset_index(drop=True)
print(f"val_meta rows: {len(val_meta)}")

val_waveforms = np.load(os.path.join(PROJECT_ROOT, "data/EchoNext_val_waveforms.npy"))
val_tabular   = np.load(os.path.join(PROJECT_ROOT, "data/EchoNext_val_tabular_features.npy"))
print(f"val_waveforms shape: {val_waveforms.shape}")
print(f"val_tabular shape  : {val_tabular.shape}") 
# %%
# §1 (cont.) — reshape waveforms: (N,1,2500,12) -> (N,12,2500)
waveforms_pt = val_waveforms[:, 0, :, :]           # (N, 2500, 12)
waveforms_pt = waveforms_pt.transpose(0, 2, 1)     # (N, 12, 2500)
print(f"waveforms_pt shape: {waveforms_pt.shape}")

# %%
# §2 — Labels + class imbalance
LABEL_COLS = [
    'lvef_lte_45_flag', 'lvwt_gte_13_flag', 'aortic_stenosis_moderate_or_greater_flag',
    'aortic_regurgitation_moderate_or_greater_flag', 'mitral_regurgitation_moderate_or_greater_flag',
    'tricuspid_regurgitation_moderate_or_greater_flag', 'pulmonary_regurgitation_moderate_or_greater_flag',
    'rv_systolic_dysfunction_moderate_or_greater_flag', 'pericardial_effusion_moderate_large_flag',
    'pasp_gte_45_flag', 'tr_max_gte_32_flag', 'shd_moderate_or_greater_flag'
]

labels_arr = val_meta[LABEL_COLS].values.astype(np.float32)
print(f"labels_arr shape: {labels_arr.shape}")
print(f"any NaNs in labels: {np.isnan(labels_arr).any()}")

prevalence = np.nanmean(labels_arr, axis=0)
for name, p in zip(LABEL_COLS, prevalence):
    print(f"{name:<55} {p:.4f}")

# %%
# §3 — "Split" — using val data for all three (verification only, not real metrics)
N = len(val_meta)
all_idx = np.arange(N)
train_idx = val_idx = test_idx = all_idx
print(f"Using {N} rows for train/val/test (identical — pipeline check only)")

# %%
# §4 (corrected) — Dataset uses ORIGINAL val_waveforms (B,1,2500,12), not waveforms_pt
class EchoNextDataset(Dataset):
    def __init__(self, indices, waveforms, tabular, labels):
        self.indices   = indices
        self.waveforms = waveforms   # expects (N, 1, 2500, 12) — model reshapes internally
        self.tabular   = tabular
        self.labels    = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            torch.tensor(self.waveforms[idx], dtype=torch.float32),
            torch.tensor(self.tabular[idx],   dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.float32),
            idx,
        )

BATCH_SIZE = 64

train_ds = EchoNextDataset(train_idx, val_waveforms, val_tabular, labels_arr)  # val_waveforms, not waveforms_pt
val_ds   = EchoNextDataset(val_idx,   val_waveforms, val_tabular, labels_arr)
test_ds  = EchoNextDataset(test_idx,  val_waveforms, val_tabular, labels_arr)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

waves, tabs, lbls, idxs = next(iter(train_loader))
print(f"Batch waveforms: {waves.shape}")  # should now be [64, 1, 2500, 12]

# %%
# §5 (corrected) — Load the pretrained EchoNext Mini-Model
from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular

model = ResNet1dWithTabular(
    len_tabular_feature_vector=7,
    filter_size=16,
    input_channels=12,
    num_classes=12,
    dropout_value=0.5,
)

MODEL_CKPT = os.path.join(
    PROJECT_ROOT,
    "IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt"
)
checkpoint = torch.load(MODEL_CKPT, map_location=DEVICE, weights_only=True)
state_dict = checkpoint['model']   # the real weights live here, no prefix stripping needed

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
if missing:
    print("Sample missing:", missing[:5])
if unexpected:
    print("Sample unexpected:", unexpected[:5])

model = model.to(DEVICE)
model.eval()

print('Model loaded successfully')
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')




# %%
# §6 (corrected) — model expects a (waveform, tabular) tuple, not two separate args
def run_inference(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for waves, tabs, lbls, _ in loader:
            waves = waves.to(device)
            tabs  = tabs.to(device)
            logits = model((waves, tabs))   # tuple, matches forward(self, x_and_tabular)
            probs  = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(lbls.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

print("Running inference...")
test_probs, test_labels = run_inference(model, test_loader, DEVICE)

print("\nPer-label AUROC (val data used as stand-in test — not a real metric):")
print("-" * 60)
aurocs = {}
for i, name in enumerate(LABEL_COLS):
    if test_labels[:, i].sum() > 0:
        auc = roc_auc_score(test_labels[:, i], test_probs[:, i])
        aurocs[name] = auc
        print(f"{name:<55} {auc:.3f}")
mean_auc = np.mean(list(aurocs.values()))
print("-" * 60)
print(f"Mean AUROC: {mean_auc:.3f}")