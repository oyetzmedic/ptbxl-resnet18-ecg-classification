# PTB-XL ResNet18 ECG Classification

A reproducible deep-learning pipeline for classifying single-superclass PTB-XL ECG records using lead-II time-frequency representations and an ImageNet-pretrained ResNet18.

The project converts ECG signals into 224 x 224 log-spectrogram images, fine-tunes ResNet18 for five PTB-XL diagnostic superclasses, and evaluates performance using the official patient-disjoint PTB-XL stratified folds.

A CPU logistic-regression baseline uses handcrafted features from all 12 leads. The saved results below compare both pipelines.

## Classes

The classifier predicts one of five PTB-XL diagnostic superclasses:

- `NORM` - Normal ECG
- `MI` - Myocardial infarction
- `STTC` - ST/T change
- `CD` - Conduction disturbance
- `HYP` - Hypertrophy

Only records whose diagnostic codes resolve to exactly one diagnostic superclass are retained, so this repository implements a single-superclass classification task rather than multilabel classification.

## ResNet18 pipeline

1. Load PTB-XL metadata and 100 Hz waveform records.
2. Map diagnostic SCP codes to PTB-XL diagnostic superclasses.
3. Retain records belonging to exactly one supported superclass.
4. Use lead II from each ECG.
5. Apply a 0.5-40 Hz band-pass filter and z-normalisation.
6. Convert the signal to a 224 x 224 log-spectrogram scaled to `[0, 1]`.
7. Fine-tune an ImageNet-pretrained ResNet18.
8. Use inverse-frequency class weights with weighted cross-entropy loss.
9. Evaluate on the held-out PTB-XL test fold.

## Patient-disjoint split

The project uses the official PTB-XL `strat_fold` assignments:

- Folds 1-8: training
- Fold 9: reserved validation fold
- Fold 10: test

The code explicitly checks for patient overlap between partitions and raises an error if leakage is detected.

The current training script reserves fold 9 but does not use it for early stopping or model selection.

## ResNet18 model

- Architecture: ResNet18
- Initialisation: ImageNet pretrained weights
- Input: lead-II ECG log-spectrogram
- Input size: 224 x 224
- Output classes: 5
- Loss: weighted cross-entropy
- Optimiser: AdamW
- Default learning rate: `3e-4`
- Default epochs: `5`
- Default batch size: `64`
- Device: CUDA when available, otherwise CPU

## Classical baseline

`baseline.py` applies the same band-pass filtering and per-lead z-normalisation before extracting 57 features: six lead-II rate and rhythm features, four statistical features from each of the 12 leads, and three lead-II spectral band powers.

`StandardScaler` is fitted on the training features. Logistic regression uses balanced class weights, `max_iter=2000` and `random_state=42`. The baseline runs on CPU.

## Archived ResNet18 experiment

The archived run used:

```text
python finetune.py --data Z:\\ --epochs 5 --batch 8
```

Environment used for the archived run:

```text
Python: 3.12.8
PyTorch: 2.6.0+cu124
torchvision: 0.21.0+cu124
CUDA runtime: 12.4
GPU: NVIDIA GeForce GTX 1050 Ti
```

## Test results and comparison

Values come from [baseline_metrics.json](baseline_metrics.json) and [finetune_metrics.json](finetune_metrics.json), rounded to four decimal places.

| Metric | Baseline | ResNet18 (5 epochs) |
| --- | ---: | ---: |
| Accuracy | 0.5448 | 0.6352 |
| Macro F1 | 0.4665 | 0.4565 |
| NORM recall | 0.5669 | 0.7884 |
| MI recall | 0.4492 | 0.3047 |
| STTC recall | 0.5165 | 0.5455 |
| CD recall | 0.6033 | 0.6250 |
| HYP recall | 0.5536 | 0.0714 |

The baseline has higher macro F1 and higher MI and HYP recall. ResNet18 has higher accuracy and higher NORM, STTC and CD recall.

Both scripts use the same label-filtering and fold-splitting utilities. The baseline uses features from all 12 leads, while ResNet18 uses lead-II spectrograms. This is a comparison of two pipelines with different inputs. The results do not isolate the effect of model architecture.

The ResNet18 results come from a five-epoch portfolio experiment. The saved metrics contain no confidence intervals or repeated-run estimates. Neither pipeline has clinical validation.

## Installation

Create a Python environment and install the pinned project dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` records the CUDA 12.4 PyTorch build used for the archived experiment.

A complete environment snapshot is also retained in `requirements_frozen.txt`.

## Download PTB-XL

Run:

```bash
python download_data.py
```

The helper uses WFDB to download waveform records into:

```text
ptbxl/
```

It requests all available records, including 500 Hz recordings, and does not explicitly fetch the two metadata CSVs. Download `ptbxl_database.csv` and `scp_statements.csv` from the PhysioNet release page below into the same directory.

The expected structure includes:

```text
ptbxl_database.csv
scp_statements.csv
records100/
```

PTB-XL is available from PhysioNet:

[PTB-XL v1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

## Running the models

### CPU baseline

With PTB-XL stored in `./ptbxl`:

```bash
python baseline.py --data ./ptbxl
```

For a Windows dataset stored at the root of drive Z:

```powershell
python baseline.py --data Z:/
```

`--data` points to the folder containing `ptbxl_database.csv`, `scp_statements.csv` and `records100/`.

Each baseline run writes `baseline_metrics.json` in the current working directory, replacing an existing file with the same name.

### ResNet18 training

With PTB-XL stored in the default `./ptbxl` directory:

```bash
python finetune.py
```

Example with explicit options:

```bash
python finetune.py --data ./ptbxl --epochs 5 --batch 8 --lr 3e-4
```

The script writes:

```text
finetune_metrics.json
finetune_resnet18_state_dict.pt
```

Model weight files are intentionally excluded from Git by `.gitignore`.

## Repository files

```text
baseline.py             CPU logistic-regression baseline
baseline_metrics.json   Saved baseline test metrics
finetune.py              ResNet18 training and evaluation pipeline
ecg_lib.py               ECG preprocessing, labels and split utilities
download_data.py         PTB-XL download helper
finetune_metrics.json    Archived ResNet18 test metrics
requirements.txt        Minimal pinned runtime dependencies
requirements_frozen.txt Full archived Python environment
environment.txt         Recorded Python/PyTorch/CUDA environment
RUN_COMMAND.txt         Command used for the archived training run
```

## Reproducibility and data handling

Raw PTB-XL waveform data and trained model files are intentionally excluded from version control. The repository contains the code, environment information, run configuration and saved metrics required to understand and reproduce the experiment.

This project is intended for machine-learning research and portfolio demonstration only. It is not a medical device and must not be used for clinical diagnosis or patient-care decisions.
