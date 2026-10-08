# PANDA

**PANDA (Probabilistic Annotator-aware Nonlinear Deep Approach)** is a unified probabilistic framework for classification with multiple annotators. The framework combines deep learning with an Expectation-Maximization (EM)-based learning procedure to model the latent class distribution and the behavior of multiple annotators.

PANDA includes three annotator models with different levels of flexibility:

- **PANDA-S**: symmetric annotator model.
- **PANDA-D**: diagonal annotator model.
- **PANDA-F**: full annotator model.

The three variants are implemented within the same model and selected through the `eta_type` parameter.

## Repository structure

```text
PANDA/
├── panda/
│   ├── panda.py
│   ├── training.py
│   └── utils.py
│
├── experiments/
│   ├── data_loader.py
│   ├── hyperparameter_tuning.py
│   ├── cross_validation.py
│   └── build_summary.py
│
├── data/
│   └── MADatasets/
│
├── models/
│
├── results/
│   └── resumen_global.json
│
├── notebooks/
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
└── test_data_loader.py
```

## Main components

### `panda/`

Contains the core implementation of the PANDA framework.

- `panda.py`: implementation of the `MultiAnnotatorEM` model.
- `training.py`: EM training procedure and neural network backbone.
- `utils.py`: auxiliary functions for evaluation, annotator simulation and model initialization.

### `experiments/`

Contains the experimental pipeline.

- `data_loader.py`: unified loading and preprocessing of the datasets.
- `hyperparameter_tuning.py`: hyperparameter optimization using Keras Tuner.
- `build_summary.py`: identifies the best trial for each dataset and PANDA variant and generates `results/resumen_global.json`.
- `cross_validation.py`: evaluates the models using the hyperparameters selected during tuning.

## Installation

The required Python dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Datasets

The experiments use datasets from the MADatasets collection, including binary, multiclass, real-world and synthetic datasets.

The datasets used by the experimental pipeline are:

```text
Breast
Bupa
Ionosphere
Pima
TicTacToe
Occupancy
Skin_NonSkin
Iris
Wine
Segmentation
Western
Voice
Polarity
Music
blobs
circles
spirals
```

The datasets are expected to be located under:

```text
data/MADatasets/MADatasets/
```

## Loading datasets

All datasets are loaded through the unified interface implemented in `data_loader.py`.

Example:

```python
from experiments.data_loader import load_dataset

base_path = r"C:\path\to\PANDA\data\MADatasets\MADatasets"

datasets = load_dataset("Breast", base_path)
```

Each dataset is returned using a common structure containing:

- training data
- validation data
- input dimensionality
- number of classes
- number of annotators

The training data follow the structure:

```python
(x_train, Y_train, M_train, Z_train)
```

where:

- `x_train`: input samples.
- `Y_train`: annotator labels.
- `M_train`: annotation mask.
- `Z_train`: reference labels used for evaluation.

## Hyperparameter tuning

Hyperparameter optimization is performed using **Keras Tuner** with Bayesian optimization.

The tuner searches over:

- number of neural network layers
- number of units per layer
- latent representation dimension `d`
- learning rate
- batch size

Detailed tuning results are organized by PANDA variant:

```text
results/
└── tuning/
    ├── sym/
    ├── diag/
    └── full/
```

The tuning directories contain the trial histories and can become large. They are therefore excluded from version control through `.gitignore`.

## Building the global summary

After hyperparameter tuning, `build_summary.py` identifies the best trial for each dataset and PANDA variant and combines the results into a single JSON file.

Run:

```bash
python -m experiments.build_summary
```

This generates:

```text
results/resumen_global.json
```

The global summary contains:

- best AUC score
- best trial identifier
- selected hyperparameters

The resulting JSON file is used by the cross-validation experiment.

## Cross-validation

The selected hyperparameters are used in the cross-validation experiments implemented in `cross_validation.py`.

Run:

```bash
python -m experiments.cross_validation
```

The experimental protocol first performs a train-validation split. The training portion is then divided into five stratified folds.

For each fold:

1. A new PANDA model is initialized.
2. The model is trained using the selected hyperparameters.
3. The model is evaluated on the fixed external validation set.

The reported metrics are:

- **ACC**: Accuracy.
- **AUC**: Area Under the ROC Curve.
- **F1**: F1-score.

The complete cross-validation results are saved locally as:

```text
results/metrics_cv_histories.pkl
```

This file is excluded from version control because it can contain large experiment histories.

## PANDA variants

The three PANDA variants are selected through the `eta_type` parameter.

For PANDA-S:

```python
eta_type="symmetric"
```

For PANDA-D:

```python
eta_type="diagonal"
```

For PANDA-F:

```python
eta_type="full"
```

## Example

A PANDA model can be trained through the `train_em` function:

```python
from panda.training import train_em

model, acc, auc, f1, history, val_history = train_em(
    train=train_data,
    val=val_data,
    input_dim=input_dim,
    K=K,
    T=T,
    d=d,
    eta_type="symmetric"
)
```

## Reproducibility

The experiments use fixed random seeds for dataset splitting and annotator simulation.

The selected hyperparameters used in the experiments are provided in:

```text
results/resumen_global.json
```

The source code required to reproduce the hyperparameter tuning and cross-validation procedures is provided in the `experiments/` directory.

## Citation

If you use PANDA in your research, please cite the associated paper.

```bibtex
@article{cardenasprobabilistic,
  title={Probabilistic Annotator-aware Nonlinear Deep Approach for Multi-class Learning},
  author={C{\'a}rdenas-Bedoya, Wilfor and Cardenas Pe{\~n}a, David Augusto and others},
  journal={...}
}
```

Citation information will be updated with the final publication details.

## License

This project is released under the MIT License.

See the `LICENSE` file for details.
