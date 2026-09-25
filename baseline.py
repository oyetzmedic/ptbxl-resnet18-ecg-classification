"""
baseline.py — classical-features baseline on PTB-XL superclasses.

Run AFTER download_data.py:  python3 baseline.py --data ./ptbxl
Order of operations: labels -> recommended patient-disjoint folds -> HARD
leakage guard -> filter/normalise -> features -> class-weighted logistic
regression -> honest metrics (macro F1 and per-class recall, not accuracy
alone — NORM dominates and accuracy would flatter).
"""
import argparse, ast, json
import numpy as np
import pandas as pd

from ecg_lib import (FS, bandpass_filter, znormalise, classical_feature_vector,
                     single_superclass_labels, patientwise_fold_split,
                     assert_patient_disjoint, SUPERCLASSES)


def load_meta(data_dir):
    df = pd.read_csv(f"{data_dir}/ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(f"{data_dir}/scp_statements.csv", index_col=0)
    return single_superclass_labels(df, agg)


def load_signals(df, data_dir):
    import wfdb  # only needed on the machine that has the data
    X = np.zeros((len(df), 1000, 12), dtype=np.float32)
    for i, fname in enumerate(df.filename_lr):
        sig, _ = wfdb.rdsamp(f"{data_dir}/{fname}")
        X[i] = znormalise(bandpass_filter(sig))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./ptbxl")
    args = ap.parse_args()

    df = load_meta(args.data)
    train, val, test = patientwise_fold_split(df)
    assert_patient_disjoint(train, test); assert_patient_disjoint(train, val)
    print(f"records: train {len(train)} | val {len(val)} | test {len(test)}")

    ytr = train.superclass.values; yte = test.superclass.values
    Xtr = np.stack([classical_feature_vector(r) for r in load_signals(train, args.data)])
    Xte = np.stack([classical_feature_vector(r) for r in load_signals(test, args.data)])

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, recall_score, accuracy_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=42).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))

    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "per_class_recall": {c: float(r) for c, r in zip(
            SUPERCLASSES, recall_score(yte, pred, labels=SUPERCLASSES, average=None))},
    }
    print(json.dumps(metrics, indent=2))
    json.dump(metrics, open("baseline_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
