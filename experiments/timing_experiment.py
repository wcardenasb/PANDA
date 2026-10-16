
import os
import json
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from experiments.data_loader import load_dataset
from panda.training import train_em
from panda.dlcl_nn import build_crowd_nn


# =========================
# Configuración
# =========================
BASE_PATH = os.path.join("data", "MADatasets", "MADatasets")
RESULTS_PATH = os.path.join("results", "timing")
SUMMARY_PATH = os.path.join("results", "resumen_global.json")
OUTPUT_PATH = os.path.join(RESULTS_PATH, "timing_folds.csv")

N_SPLITS = 5
SEED = 42
BASELINE_EPOCHS = 100
EM_ITERS = 10
M_STEPS = 10
BASELINE_BATCH_SIZE = 8

FIXED_SPLIT_DATASETS = {"Polarity", "Music"}

os.makedirs(RESULTS_PATH, exist_ok=True)


# =========================
# Bases de datos
# =========================
SEMI_BINARY = [
    # "Breast", "Bupa", "Ionosphere", "Pima", "TicTacToe",
    # "Occupancy", "Skin_NonSkin"
]

SEMI_MULTI = [
    # "Iris", "Wine", "Segmentation", "Western"
]

REAL_BINARY = [
    "Voice",
      "Polarity"
      ]
REAL_MULTI = ["Music"]

DATASETS = (
    [(db, "semi_synthetic", None) for db in SEMI_BINARY + SEMI_MULTI]
    + [(db, "real", None) for db in REAL_BINARY if db != "Voice"]
    + [("Voice", "real", protocol) for protocol in ["YG", "YR", "YB"]]
    + [(db, "real", None) for db in REAL_MULTI]
)

MODELS = {
    "PANDA-S": ("sym", "symmetric"),
    "PANDA-D": ("diag", "diagonal"),
    "PANDA-F": ("full", "full"),
}


# =========================
# Cargar hiperparámetros
# =========================
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)


# =========================
# Funciones auxiliares
# =========================
def to_numpy(data):
    return np.asarray(data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_key(db, protocol, model_name, fold):
    protocol_key = str(protocol) if db == "Voice" and protocol is not None and not pd.isna(protocol) else ""
    return (str(db), protocol_key, str(model_name), int(fold))


def save_result(row, key):
    pd.DataFrame([row]).to_csv(OUTPUT_PATH, mode="a", header=not os.path.exists(OUTPUT_PATH), index=False)
    completed.add(key)



def get_panda_params(dataset, model_code, protocol=None):
    dataset_key = protocol if dataset == "Voice" else dataset
    hp = summary[dataset_key][model_code]["hyperparameters"]
    return {"d": int(hp.get("d", 16)), "lr": float(hp.get("Learning_rate", 1e-3)),
             "n_layers": int(hp.get("n_layers", 3)), "nn_list": [int(hp.get("units_0", 100)),
              int(hp.get("units_1", 90)), int(hp.get("units_2", 64))], "batch_size": int(hp.get("batch_size", 128))}



# =========================
# Recuperar resultados previos
# =========================
completed = set()

if os.path.exists(OUTPUT_PATH):
    previous = pd.read_csv(OUTPUT_PATH)
    completed = {make_key(str(row["dataset"]), row["protocol"], str(row["model"]), int(row["fold"])) for _, row in previous.iterrows()}

print(f"Combinaciones ya completadas: {len(completed)}")


# =========================
# Experimento
# =========================
for db, category, protocol in DATASETS:
    print(f"\n{'=' * 60}\nBase: {db} | Tipo: {category} | Protocolo: {protocol}")

    data = load_dataset(db, base_path=BASE_PATH, protocol=protocol)
    dataset = data[0]

    x_train, Y_train, M_train, Z_train = map(to_numpy, dataset["train_data"])
    x_val, Y_val, M_val, Z_val = map(to_numpy, dataset["val_data"])

    input_dim, K, T = dataset["input_dim"], dataset["K"], dataset["T"]
    val_data = (tf.constant(x_val, dtype=tf.float32), tf.constant(Y_val, dtype=tf.float32), tf.constant(M_val, dtype=tf.float32), tf.constant(Z_val, dtype=tf.float32))

    # Polarity y Music usan Xtrain/Xtest originales, sin validación cruzada.
    if db in FIXED_SPLIT_DATASETS:
        folds = [(1, np.arange(len(x_train)))]
    else:
        train_labels = np.argmax(Z_train, axis=1)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = [(fold, train_idx) for fold, (train_idx, _) in enumerate(skf.split(x_train, train_labels), start=1)]

    for fold, train_idx in folds:
        print(f"\nEjecución {fold}/{1 if db in FIXED_SPLIT_DATASETS else N_SPLITS}")

        train_fold = (
            tf.constant(x_train[train_idx], dtype=tf.float32),
            tf.constant(Y_train[train_idx], dtype=tf.float32),
            tf.constant(M_train[train_idx], dtype=tf.float32),
            tf.constant(Z_train[train_idx], dtype=tf.float32),
        )

        # -------------------------
        # Red neuronal base
        # -------------------------
        model_name = "Baseline-NN"
        key = make_key(db, protocol, model_name, fold)

        if key in completed:
            print(f"{model_name}: ya completado; se omite.")
        else:
            tf.keras.backend.clear_session()
            set_seed(SEED + fold)

            x_fold, Y_fold = train_fold[0], train_fold[1]
            Y_train_dict = {f"annotator_{t}": Y_fold[:, t, :] for t in range(T)}

            start = time.perf_counter()
            model = build_crowd_nn(input_dim=input_dim, K=K, T=T, units_list=(100, 90), lr=1e-3)
            model.fit(x_fold, Y_train_dict, epochs=BASELINE_EPOCHS, batch_size=BASELINE_BATCH_SIZE, verbose=0)
            elapsed = time.perf_counter() - start

            row = {
                "dataset": db, "category": category, "protocol": protocol,
                "model": model_name, "fold": fold,
                "train_time_seconds": elapsed,
                "epochs": BASELINE_EPOCHS, "em_iters": 0, "m_steps": 0,
            }

            save_result(row, key)
            print(f"{model_name}: {elapsed:.2f} s")

            del model
            tf.keras.backend.clear_session()

        # -------------------------
        # PANDA-S, PANDA-D, PANDA-F
        # -------------------------
        for model_name, (model_code, eta_type) in MODELS.items():
            key = make_key(db, protocol, model_name, fold)

            if key in completed:
                print(f"{model_name}: ya completado; se omite.")
                continue

            tf.keras.backend.clear_session()
            set_seed(SEED + fold)

            params = get_panda_params(db, model_code, protocol)

            start = time.perf_counter()
            train_em(
                train=train_fold,
                val=val_data,
                input_dim=input_dim,
                K=K, T=T,
                d=params["d"],
                lr=params["lr"],
                n_layers=params["n_layers"],
                nn_list=params["nn_list"],
                batch_size=params["batch_size"],
                em_iters=EM_ITERS,
                m_steps=M_STEPS,
                eta_type=eta_type,
            )
            elapsed = time.perf_counter() - start

            row = {
                "dataset": db, "category": category, "protocol": protocol,
                "model": model_name, "fold": fold,
                "train_time_seconds": elapsed,
                "epochs": EM_ITERS * M_STEPS,
                "em_iters": EM_ITERS, "m_steps": M_STEPS,
            }

            save_result(row, key)
            print(f"{model_name}: {elapsed:.2f} s")

            tf.keras.backend.clear_session()

print(f"\nExperimento terminado. Resultados: {OUTPUT_PATH}")

# ============================================================
# RESUMEN DE TIEMPOS
# ============================================================

def build_timing_summary():
    if not os.path.exists(OUTPUT_PATH):
        print("No existe el archivo de tiempos:", OUTPUT_PATH)
        return

    df = pd.read_csv(OUTPUT_PATH)
    df["protocol"] = df["protocol"].fillna("")
    df["train_time_seconds"] = pd.to_numeric(
        df["train_time_seconds"], errors="coerce"
    )

    models = ["Baseline-NN", "PANDA-S", "PANDA-D", "PANDA-F"]
    rows = []

    for (dataset, category, protocol), group in df.groupby(
        ["dataset", "category", "protocol"], dropna=False, sort=True
    ):
        row = {
            "dataset": dataset,
            "category": category,
            "protocol": protocol,
        }

        for model in models:
            times = group.loc[
                group["model"] == model, "train_time_seconds"
            ].dropna()

            row[f"{model}_n"] = len(times)
            row[f"{model}_mean_seconds"] = times.mean() if len(times) else np.nan
            row[f"{model}_std_seconds"] = times.std(ddof=1) if len(times) > 1 else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_PATH, "timing_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    print("\n" + "=" * 60)
    print("RESUMEN DE TIEMPOS GENERADO")
    print("=" * 60)
    print(f"Archivo: {summary_path}")
    print(f"Bases/protocolos resumidos: {len(summary)}")
    print(summary.to_string(index=False))


build_timing_summary()