# PANDA

**PANDA (Probabilistic Annotator-aware Nonlinear Deep Approach)** is a probabilistic deep-learning framework for multiclass classification from labels provided by multiple annotators. It combines a neural-network classifier with an Expectation-Maximization (EM) training procedure to jointly model latent class probabilities and annotator behavior.

PANDA supports three annotator-model parameterizations:

- **PANDA-S** (`eta_type="symmetric"`): symmetric annotator model.
- **PANDA-D** (`eta_type="diagonal"`): diagonal annotator model.
- **PANDA-F** (`eta_type="full"`): full annotator model, initialized through a symmetric warm-start stage.

The three variants share the `MultiAnnotatorEM` implementation and can be selected through the `eta_type` argument.

## Repository structure

```text
PANDA/
├── panda/
│   ├── panda.py                     # MultiAnnotatorEM model
│   ├── training.py                  # Main EM training procedure
│   ├── training_timing.py           # Instrumented EM timing procedure
│   └── utils.py                     # Annotator simulation and initialization
├── experiments/
│   ├── data_loader.py               # Unified dataset loading
│   ├── hyperparameter_tuning.py     # Bayesian hyperparameter search
│   ├── build_summary.py             # Aggregate tuning results
│   ├── cross_validation.py          # Evaluation across training folds
│   ├── confidence_recovery.py       # Annotator-confidence experiments
│   ├── plot_confidence.py           # Confidence heatmaps
│   ├── timing_em.py                 # E-step / M-step timing experiments
│   ├── histories_training.py        # Training and validation histories
│   └── export_histories_normalized.py # Normalize and aggregate histories
├── data/
│   └── MADatasets/                  # Dataset files (see below)
├── results/
│   ├── resumen_global.json          # Selected hyperparameters
│   ├── histories_normalized_tikz.csv
│   ├── confidence_recovery/
│   │   └── confidence_recovery_summary_0.5_es.csv
│   └── timing/
│       └── timing_summary_em_breakdown.csv
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The tree highlights the main files needed for the reported experiments; other supporting files or directories may also be present.

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/wcardenasb/PANDA.git
cd PANDA
python -m pip install -r requirements.txt
```

Run the commands below **from the repository root**. Python 3.10 was used for the local development workflow; compatible TensorFlow and dependency versions depend on the operating system and hardware.

## Datasets and loading

The experimental pipeline supports datasets from the `MADatasets` collection, including binary, multiclass, and datasets with simulated annotators. Configured dataset names include:

```text
Breast, Bupa, Ionosphere, Pima, TicTacToe, Occupancy, Skin_NonSkin,
Iris, Wine, Segmentation, Western, Voice, Polarity, Music,
blobs, circles, spirals
```

Dataset files are expected in:

```text
data/MADatasets/MADatasets/
```

Make sure the `.mat` files required by your experiment are available at that location. Listing a dataset here does not imply that its raw file is distributed with this repository.

```python
from pathlib import Path
from experiments.data_loader import load_dataset

base_path = Path("data/MADatasets/MADatasets")
datasets = load_dataset("Breast", str(base_path), seed=42, p_errors=0.3)
dataset = datasets[0]

train_data = dataset["train_data"]
val_data = dataset["val_data"]
input_dim = dataset["input_dim"]
K = dataset["K"]
T = dataset["T"]
```

Each training or validation tuple has the form:

```python
(x, Y, M, Z)
```

where `x` contains input features, `Y` contains annotator-label indicators, `M` is the observation mask, and `Z` contains reference labels used for model assessment. Additional metadata, including sample indices and simulated annotator clusters when applicable, is returned by the loader.

**Note on fixed-split datasets:** For datasets that provide separate training/test sets but no test-time annotator labels, the loader constructs placeholder validation annotation tensors to match the common interface. Accordingly, annotation-based validation Q-loss on those datasets should not be interpreted as an observed-label validation loss; evaluation using reference-label metrics is distinct from this diagnostic.

## PANDA training

The standard training entry point is `panda.training.train_em`:

```python
from panda.training import train_em

model, acc, auc, f1, history, val_history = train_em(
    train=train_data,
    val=val_data,
    input_dim=input_dim,
    K=K,
    T=T,
    d=16,
    eta_type="symmetric",  # "symmetric", "diagonal", or "full"
)
```

`train_em` returns the trained model, validation accuracy, ROC AUC, F1-score, and the recorded training and validation loss histories. The implementation selects/restores the best weights using validation AUC and supports `min_em_iters`, `patience`, and `min_delta` for early stopping. The histories track the EM auxiliary objective (`q_loss`), while optional L2 weight decay is applied during optimization. PANDA-F first runs a symmetric initialization phase.

## Hyperparameter optimization

`experiments/hyperparameter_tuning.py` uses **Keras Tuner Bayesian optimization** to select model settings, including network depth, layer widths, latent dimension `d`, learning rate, and batch size.

Trial artifacts are organized under `results/tuning/` and are excluded from version control because of their size.

After tuning, aggregate the selected trials with:

```bash
python -m experiments.build_summary
```

The resulting configuration file is:

```text
results/resumen_global.json
```

A version of this JSON file is included in the repository so that experiments can use the selected hyperparameters without rerunning the full search.

## Evaluation experiments

### Cross-validation protocol

```bash
python -m experiments.cross_validation
```

For datasets with an internally generated split, the pipeline first creates an external training/validation partition. It then constructs **five stratified subsets of the training partition**. A new model is trained on each subset and evaluated against the **same fixed external validation set**. This is not the conventional five-fold protocol in which a different fold becomes the test set on each iteration.

Reported metrics include accuracy (**ACC**), area under the ROC curve (**AUC**), and **F1-score**. The full local experiment histories may be stored in `results/metrics_cv_histories.pkl`; such intermediate files are not committed.

### Annotator-confidence recovery

```bash
python -m experiments.confidence_recovery
python -m experiments.plot_confidence
```

These scripts evaluate estimated annotator confidence in datasets with simulated annotator expertise and generate confidence matrices/heatmaps. The published summary for an induced-error setting of `p_errors=0.5` is:

```text
results/confidence_recovery/confidence_recovery_summary_0.5_es.csv
```

The error-rate setting and selected datasets can be adjusted in the experiment script.

### Computational cost of E- and M-steps

```bash
python -m experiments.timing_em
```

`panda/training_timing.py` instruments EM execution with `time.perf_counter()` to record total and mean E-step/M-step times and the remaining execution time. PANDA-F measurements include its symmetric warm-start phase. The committed aggregate timing results are in:

```text
results/timing/timing_summary_em_breakdown.csv
```

Measurements depend on hardware, TensorFlow execution, and experiment settings; they should not be treated as hardware-independent benchmarks.

### Training and validation histories

```bash
python -m experiments.histories_training
python -m experiments.export_histories_normalized
```

The first script records per-M-epoch training and validation histories across datasets and PANDA variants. The export script normalizes each fold's loss by its initial value, interpolates its relative training progress onto 101 points, and reports fold-level means and standard deviations:

```text
results/histories_normalized_tikz.csv
```

**Historical-results note:** The export script reads the historical local file `results/metrics_cv_histories_training2.pkl`. This filename is retained for compatibility with the previously generated figures; it does **not** imply that `panda/training2.py` remains part of the codebase. The current `panda/training.py` uses a revised loss-logging convention, so rerunning the histories can produce curves that differ from the committed historical summary. To export newly generated histories, update the `INPUT` path in `export_histories_normalized.py` to match the output of the new run.

## Results included in Git

Only compact, manuscript-related outputs are committed:

| File | Content |
| --- | --- |
| `results/resumen_global.json` | Selected hyperparameter settings |
| `results/histories_normalized_tikz.csv` | Normalized training and validation curves |
| `results/timing/timing_summary_em_breakdown.csv` | Aggregate E/M timing measurements |
| `results/confidence_recovery/confidence_recovery_summary_0.5_es.csv` | Annotator-confidence summary |

Per-fold timing files, training-history pickles, generated figures, and other intermediate outputs are intentionally excluded. All generated results should be interpreted in conjunction with the configuration and protocol used to obtain them.

## Reproducibility notes

- Dataset splitting and annotator simulation use fixed seeds where specified in the experimental scripts; other sources of nondeterminism may remain, especially in TensorFlow execution.
- Experiment scripts should be run from the repository root and require the dataset files to be available locally.
- `results/resumen_global.json` stores the selected tuning configurations.
- The timing and history experiments are separate instrumented procedures; their metrics and loss logs should not be assumed interchangeable without checking the individual script settings.
- Published aggregate CSVs document previous experimental runs; successful import or compilation of the code alone does not establish exact numerical reproducibility.


## Citation

If you use PANDA in your research, please cite the associated manuscript:

```bibtex
@misc{cardenasbedoya_panda,
  title  = {Probabilistic Annotator-aware Nonlinear Deep Approach for Multi-class Learning},
  author = {C{\'a}rdenas-Bedoya, Wilfor and C{\'a}rdenas-Pe{\~n}a, David and Gil-Gonzalez, Julián },
  url    = {https://github.com/wcardenasb/PANDA}
}
```

The bibliographic information will be updated once the manuscript is published.


## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
