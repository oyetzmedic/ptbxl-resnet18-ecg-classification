"""
finetune.py — fine-tune an open-source pretrained model (torchvision ResNet18,
ImageNet weights) on lead-II ECG spectrograms for PTB-XL superclasses.

Why this design (interviewers ask):
- "Fine-tuning an open-source model" is the criterion. ResNet18/ImageNet is a
  genuinely pretrained open model; converting ECG to a time-frequency image is
  a standard, well-understood transfer route for 1-D biosignals.
- Spectrograms are NOT medical imaging, and this project never claims imaging
  experience. It claims biosignal processing + fine-tuning, precisely.
- Same patient-disjoint folds and hard leakage guard as the baseline, so the
  two models are compared on identical, honest terms.

Run AFTER download_data.py:  python3 finetune.py --data ./ptbxl --epochs 5
CPU works (slow); a modest GPU finishes in minutes.
"""
import argparse, ast, json
import numpy as np
import pandas as pd

from ecg_lib import (bandpass_filter, znormalise, spectrogram_image,
                     single_superclass_labels, patientwise_fold_split,
                     assert_patient_disjoint, SUPERCLASSES, LEAD_II)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./ptbxl")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    import torch, wfdb
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import resnet18, ResNet18_Weights
    from sklearn.metrics import f1_score, recall_score, accuracy_score
    torch.manual_seed(42); np.random.seed(42)

    df = pd.read_csv(f"{args.data}/ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(f"{args.data}/scp_statements.csv", index_col=0)
    df = single_superclass_labels(df, agg)
    train, val, test = patientwise_fold_split(df)
    assert_patient_disjoint(train, test); assert_patient_disjoint(train, val)
    cls_to_idx = {c: i for i, c in enumerate(SUPERCLASSES)}
    weights_spec = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights_spec.transforms()

    class SpecDS(Dataset):
        def __init__(self, frame):
            self.f = frame.reset_index(drop=True)
        def __len__(self):
            return len(self.f)
        def __getitem__(self, i):
            row = self.f.iloc[i]
            sig, _ = wfdb.rdsamp(f"{args.data}/{row.filename_lr}")
            lead = znormalise(bandpass_filter(sig))[:, LEAD_II]
            img = spectrogram_image(lead)                      # (224,224) in [0,1]
            x = torch.from_numpy(np.stack([img, img, img]))    # 3-channel
            x = preprocess(x)
            return x, cls_to_idx[row.superclass]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18(weights=weights_spec)   # the open-source
    model.fc = nn.Linear(model.fc.in_features, len(SUPERCLASSES))  # pretrained model
    model = model.to(device)

    # Class weights from the training distribution — NORM dominates PTB-XL.
    counts = train.superclass.value_counts().reindex(SUPERCLASSES).values.astype(float)
    weights = torch.tensor((counts.sum() / (len(counts) * counts)),
                           dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tr_dl = DataLoader(SpecDS(train), batch_size=args.batch, shuffle=True, num_workers=0)
    te_dl = DataLoader(SpecDS(test), batch_size=args.batch, num_workers=0)

    for epoch in range(args.epochs):
        model.train(); tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), torch.as_tensor(y).to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += float(loss)
        print(f"epoch {epoch+1}/{args.epochs}  mean_loss={tot/len(tr_dl):.4f}")

    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for x, y in te_dl:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
            ys.extend(list(y))
    idx_to_cls = {v: k for k, v in cls_to_idx.items()}
    pred_lbl = [idx_to_cls[p] for p in preds]
    true_lbl = [idx_to_cls[int(t)] for t in ys]

    metrics = {
        "accuracy": float(accuracy_score(true_lbl, pred_lbl)),
        "macro_f1": float(f1_score(true_lbl, pred_lbl, average="macro")),
        "per_class_recall": {c: float(r) for c, r in zip(
            SUPERCLASSES, recall_score(true_lbl, pred_lbl,
                                       labels=SUPERCLASSES, average=None))},
        "epochs": args.epochs,
    }
    print(json.dumps(metrics, indent=2))
    json.dump(metrics, open("finetune_metrics.json", "w"), indent=2)
    torch.save(model.state_dict(), "finetune_resnet18_state_dict.pt")


if __name__ == "__main__":
    main()
