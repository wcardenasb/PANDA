
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "metrics_cv_histories_training2.pkl"
OUTPUT = ROOT / "results" / "histories_normalized_tikz.csv"

# ============================================================
# CONFIGURATION
# ============================================================

N_POINTS = 101
N_FOLDS = 5

DATABASES = [
    "Bupa", "Ionosphere", "Breast", "Pima", "TicTacToe",
    "Occupancy", "Skin_NonSkin", "Iris", "Wine", "Segmentation",
    "Western", "YG", "YR", "YB", "Polarity"
]

DISPLAY_NAMES = {
    "Skin_NonSkin": "Skin",
    "YG": "Voice (G)",
    "YR": "Voice (R)",
    "YB": "Voice (B)"
}

MODELS = ["sym", "diag", "full"]

SPLITS = {
    "validation": "history_val",
    "train": "history"
}

# ============================================================
# LOAD AND CHECK
# ============================================================

df = pd.read_pickle(INPUT)

required = {"database", "model", "history", "history_val"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Columnas ausentes: {sorted(missing)}")

if df.duplicated(subset=["database", "model"]).any():
    raise ValueError("Hay combinaciones (database, model) duplicadas.")

x_common = np.linspace(0.0, 1.0, N_POINTS)
rows = []

# ============================================================
# NORMALIZE EACH FOLD, INTERPOLATE AND AGGREGATE
# ============================================================

for model in MODELS:

    for database in DATABASES:

        match = df.loc[
            (df["model"] == model) &
            (df["database"] == database)
        ]

        if len(match) != 1:
            raise ValueError(
                f"Combinación ausente o duplicada: {database}, {model}"
            )

        row = match.iloc[0]

        for split, column in SPLITS.items():

            histories = row[column]

            if len(histories) != N_FOLDS:
                raise ValueError(
                    f"{database}/{model}/{split}: "
                    f"se esperaban {N_FOLDS} folds, "
                    f"pero hay {len(histories)}."
                )

            interpolated = []
            lengths = []

            for fold_number, history in enumerate(histories, start=1):

                loss = np.asarray(
                    history, dtype=float
                ).reshape(-1)

                if (
                    len(loss) < 2
                    or not np.isfinite(loss).all()
                    or loss[0] <= 0
                ):
                    raise ValueError(
                        f"Curva inválida: {database}/{model}/{split}, "
                        f"fold {fold_number}."
                    )

                # Normalize X: relative training progress
                x_fold = np.linspace(0.0, 1.0, len(loss))

                # Normalize Y: relative loss
                y_fold = loss / loss[0]

                # Interpolate on a common grid
                interpolated.append(
                    np.interp(x_common, x_fold, y_fold)
                )

                lengths.append(len(loss))

            # Average across folds
            curves = np.vstack(interpolated)

            means = curves.mean(axis=0)
            stds = curves.std(axis=0, ddof=1)

            for x, mean, std in zip(x_common, means, stds):

                rows.append({
                    "database": database,
                    "database_label": DISPLAY_NAMES.get(
                        database, database
                    ),
                    "model": model,
                    "split": split,
                    "x_norm": x,
                    "loss_mean_norm": mean,
                    "loss_std_norm": std,
                    "n_folds": len(histories),
                    "min_epochs": min(lengths),
                    "max_epochs": max(lengths)
                })

# ============================================================
# EXPORT
# ============================================================

result = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(
    OUTPUT,
    index=False,
    float_format="%.8f"
)

print(f"Archivo: {OUTPUT}")
print(f"Bases: {result['database'].nunique()}")
print(f"Modelos: {result['model'].unique().tolist()}")
print(f"Conjuntos: {result['split'].unique().tolist()}")
print(f"Filas exportadas: {len(result)}")
print(f"Curvas medias: {len(result) // N_POINTS}")
