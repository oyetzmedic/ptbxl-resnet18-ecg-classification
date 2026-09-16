\# PTB-XL ResNet18 ECG Classification



A reproducible deep-learning pipeline for classifying single-superclass PTB-XL ECG records using lead-II time-frequency representations and an ImageNet-pretrained ResNet18.



The project converts ECG signals into 224 x 224 log-spectrogram images, fine-tunes ResNet18 for five PTB-XL diagnostic superclasses, and evaluates performance using the official patient-disjoint PTB-XL stratified folds.



\## Classes



The classifier predicts one of five PTB-XL diagnostic superclasses:



\- `NORM` - Normal ECG

\- `MI` - Myocardial infarction

\- `STTC` - ST/T change

\- `CD` - Conduction disturbance

\- `HYP` - Hypertrophy



Only records whose diagnostic codes resolve to exactly one diagnostic superclass are retained, so this repository implements a single-superclass classification task rather than multilabel classification.



\## Pipeline



1\. Load PTB-XL metadata and 100 Hz waveform records.

2\. Map diagnostic SCP codes to PTB-XL diagnostic superclasses.

3\. Retain records belonging to exactly one supported superclass.

4\. Use lead II from each ECG.

5\. Apply a 0.5-40 Hz band-pass filter and z-normalisation.

6\. Convert the signal to a 224 x 224 log-spectrogram scaled to `\[0, 1]`.

7\. Fine-tune an ImageNet-pretrained ResNet18.

8\. Use inverse-frequency class weights with weighted cross-entropy loss.

9\. Evaluate on the held-out PTB-XL test fold.



\## Patient-disjoint split



The project uses the official PTB-XL `strat\_fold` assignments:



\- Folds 1-8: training

\- Fold 9: reserved validation fold

\- Fold 10: test



The code explicitly checks for patient overlap between partitions and raises an error if leakage is detected.



The current training script reserves fold 9 but does not use it for early stopping or model selection.



\## Model



\- Architecture: ResNet18

\- Initialisation: ImageNet pretrained weights

\- Input: lead-II ECG log-spectrogram

\- Input size: 224 x 224

\- Output classes: 5

\- Loss: weighted cross-entropy

\- Optimiser: AdamW

\- Default learning rate: `3e-4`

\- Default epochs: `5`

\- Default batch size: `64`

\- Device: CUDA when available, otherwise CPU



\## Archived experiment



The archived run used:



```text

python finetune.py --data Z:\\ --epochs 5 --batch 8

````



Environment used for the archived run:



```text

Python: 3.12.8

PyTorch: 2.6.0+cu124

torchvision: 0.21.0+cu124

CUDA runtime: 12.4

GPU: NVIDIA GeForce GTX 1050 Ti

```



\### Test results



| Metric      | Result |

| ----------- | -----: |

| Accuracy    | 0.6352 |

| Macro F1    | 0.4565 |

| NORM recall | 0.7884 |

| MI recall   | 0.3047 |

| STTC recall | 0.5455 |

| CD recall   | 0.6250 |

| HYP recall  | 0.0714 |



These results are from a five-epoch portfolio experiment rather than a clinically validated model. Performance varies substantially by class, particularly for the minority HYP class.



\## Installation



Create a Python environment and install the pinned project dependencies:



```bash

pip install -r requirements.txt

```



`requirements.txt` records the CUDA 12.4 PyTorch build used for the archived experiment.



A complete environment snapshot is also retained in `requirements\_frozen.txt`.



\## Download PTB-XL



Run:



```bash

python download\_data.py

```



The script uses WFDB to download the PTB-XL 100 Hz records and metadata into:



```text

ptbxl/

```



The expected structure includes:



```text

ptbxl\_database.csv

scp\_statements.csv

records100/

```



PTB-XL is available from PhysioNet:



\[https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)



\## Training



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

finetune\_metrics.json

finetune\_resnet18\_state\_dict.pt

```



Model weight files are intentionally excluded from Git by `.gitignore`.



\## Repository files



```text

finetune.py              Training and evaluation pipeline

ecg\_lib.py               ECG preprocessing, labels and split utilities

download\_data.py         PTB-XL download helper

finetune\_metrics.json    Archived test metrics

requirements.txt         Minimal pinned runtime dependencies

requirements\_frozen.txt  Full archived Python environment

environment.txt          Recorded Python/PyTorch/CUDA environment

RUN\_COMMAND.txt          Command used for the archived training run

```



\## Reproducibility and data handling



Raw PTB-XL waveform data and trained model files are intentionally excluded from version control. The repository contains the code, environment information, run configuration and saved metrics required to understand and reproduce the experiment.



This project is intended for machine-learning research and portfolio demonstration only. It is not a medical device and must not be used for clinical diagnosis or patient-care decisions.

