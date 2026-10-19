
import os
import json
import time
import random
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from experiments.data_loader import load_dataset
from panda.training import train_em
from panda.dlcl_nn import build_crowd_nn

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# Configuración
# =========================
BASE_PATH = os.path.join("data", "MADatasets", "MADatasets")
RESULTS_PATH = os.path.join("results", "timing")
SUMMARY_PATH = os.path.join("results", "resumen_global.json")

# Archivos nuevos: no mezclar con el experimento anterior
OUTPUT_PATH = os.path.join(RESULTS_PATH, "timing_folds_night.csv")
TIMING_SUMMARY_PATH = os.path.join(RESULTS_PATH, "timing_summary_night.csv")

N_SPLITS = 5
SEED = 42
BASELINE_EPOCHS = 100
EM_ITERS = 10
M_STEPS = 10
BASELINE_BATCH_SIZE = 8

# Calentamiento no contabilizado en los resultados
DO_WARMUP = True
WARMUP_BASELINE_EPOCHS = 1
WARMUP_EM_ITERS = 1
WARMUP_M_STEPS = 1

FIXED_SPLIT_DATASETS = {"Polarity", "Music"}

os.makedirs(RESULTS_PATH, exist_ok=True)


# =========================
# Bases de datos
# =========================
SEMI_BINARY = [
    "Breast", "Bupa", "Ionosphere", "Pima", "TicTacToe",
    "Occupancy", "Skin_NonSkin"
]

SEMI_MULTI = [
    "Iris", "Wine", "Segmentation", "Western"
]

REAL_BINARY = ["Voice", "Polarity"]
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

# Incluimos el modelo base en la rotación
METHODS = [
    "Baseline-NN", "PANDA-S", "PANDA-D", "PANDA-F"
]


# =========================
# Hiperparámetros
# =========================
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    hp_summary = json.load(f)


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
    protocol_key = (
        str(protocol)
        if db == "Voice" and protocol is not None and not pd.isna(protocol)
        else ""
    )
    return (str(db), protocol_key, str(model_name), int(fold))


def get_panda_params(dataset, model_code, protocol=None):
    dataset_key = protocol if dataset == "Voice" else dataset
    hp = hp_summary[dataset_key][model_code]["hyperparameters"]

    return {
        "d": int(hp.get("d", 16)),
        "lr": float(hp.get("Learning_rate", 1e-3)),
        "n_layers": int(hp.get("n_layers", 3)),
        "nn_list": [
            int(hp.get("units_0", 100)),
            int(hp.get("units_1", 90)),
            int(hp.get("units_2", 64)),
        ],
        "batch_size": int(hp.get("batch_size", 128)),
    }


def save_result(row, key):
    pd.DataFrame([row]).to_csv(
        OUTPUT_PATH,
        mode="a",
        header=not os.path.exists(OUTPUT_PATH),
        index=False,
    )
    completed.add(key)


def run_baseline(train_fold, input_dim, K, T, epochs):
    x_fold, Y_fold = train_fold[0], train_fold[1]
    Y_train_dict = {
        f"annotator_{t}": Y_fold[:, t, :]
        for t in range(T)
    }

    model = build_crowd_nn(
        input_dim=input_dim,
        K=K,
        T=T,
        units_list=(100, 90),
        lr=1e-3,
    )

    model.fit(
        x_fold,
        Y_train_dict,
        epochs=epochs,
        batch_size=BASELINE_BATCH_SIZE,
        verbose=0,
    )

    del model
    tf.keras.backend.clear_session()


def run_panda(train_fold, val_data, input_dim, K, T,
              db, protocol, model_name, em_iters, m_steps):
    model_code, eta_type = MODELS[model_name]
    params = get_panda_params(db, model_code, protocol)

    train_em(
        train=train_fold,
        val=val_data,
        input_dim=input_dim,
        K=K,
        T=T,
        d=params["d"],
        lr=params["lr"],
        n_layers=params["n_layers"],
        nn_list=params["nn_list"],
        batch_size=params["batch_size"],
        em_iters=em_iters,
        m_steps=m_steps,
        eta_type=eta_type,
    )

    tf.keras.backend.clear_session()


def measure_training(fn):
    """Mide tiempo real y, si es posible, CPU y memoria del proceso."""
    stats = {
        "process_cpu_max_percent": np.nan,
        "system_cpu_max_percent": np.nan,
        "process_memory_max_mb": np.nan,
    }

    stop_monitor = threading.Event()
    process = psutil.Process(os.getpid()) if psutil else None

    def monitor():
        if process is None:
            return

        process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        while not stop_monitor.wait(0.5):
            try:
                process_cpu = process.cpu_percent(interval=None)
                system_cpu = psutil.cpu_percent(interval=None)
                memory_mb = process.memory_info().rss / (1024 ** 2)

                stats["process_cpu_max_percent"] = max(
                    stats["process_cpu_max_percent"]
                    if not np.isnan(stats["process_cpu_max_percent"]) else 0,
                    process_cpu,
                )
                stats["system_cpu_max_percent"] = max(
                    stats["system_cpu_max_percent"]
                    if not np.isnan(stats["system_cpu_max_percent"]) else 0,
                    system_cpu,
                )
                stats["process_memory_max_mb"] = max(
                    stats["process_memory_max_mb"]
                    if not np.isnan(stats["process_memory_max_mb"]) else 0,
                    memory_mb,
                )
            except (psutil.Error, OSError):
                pass

    thread = None
    if process is not None:
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    start = time.perf_counter()
    try:
        fn()
    finally:
        elapsed = time.perf_counter() - start
        stop_monitor.set()
        if thread is not None:
            thread.join(timeout=2)

    return elapsed, stats


def run_warmup(db, protocol, train_fold, val_data, input_dim, K, T):
    """Calentamiento breve; sus tiempos no se guardan."""
    print("Calentamiento inicial (no contabilizado)...")

    for model_name in METHODS:
        tf.keras.backend.clear_session()
        set_seed(SEED)

        if model_name == "Baseline-NN":
            run_baseline(
                train_fold, input_dim, K, T,
                epochs=WARMUP_BASELINE_EPOCHS,
            )
        else:
            run_panda(
                train_fold, val_data, input_dim, K, T,
                db, protocol, model_name,
                em_iters=WARMUP_EM_ITERS,
                m_steps=WARMUP_M_STEPS,
            )

        print(f"  Warm-up {model_name}: listo")


# =========================
# Recuperar resultados previos
# =========================
completed = set()

if os.path.exists(OUTPUT_PATH):
    previous = pd.read_csv(OUTPUT_PATH)

    for _, row in previous.iterrows():
        completed.add(
            make_key(
                str(row["dataset"]),
                row["protocol"],
                str(row["model"]),
                int(row["fold"]),
            )
        )

print(f"Combinaciones ya completadas: {len(completed)}")
if psutil is None:
    print("Aviso: psutil no está instalado; no se registrarán CPU ni memoria.")


# =========================
# Experimento
# =========================
global_fold_counter = 0

for db, category, protocol in DATASETS:
    print(f"\n{'=' * 65}")
    print(f"Base: {db} | Tipo: {category} | Protocolo: {protocol}")

    data = load_dataset(db, base_path=BASE_PATH, protocol=protocol)
    dataset = data[0]

    x_train, Y_train, M_train, Z_train = map(
        to_numpy, dataset["train_data"]
    )
    x_val, Y_val, M_val, Z_val = map(
        to_numpy, dataset["val_data"]
    )

    input_dim = dataset["input_dim"]
    K = dataset["K"]
    T = dataset["T"]

    val_data = (
        tf.constant(x_val, dtype=tf.float32),
        tf.constant(Y_val, dtype=tf.float32),
        tf.constant(M_val, dtype=tf.float32),
        tf.constant(Z_val, dtype=tf.float32),
    )

    if db in FIXED_SPLIT_DATASETS:
        folds = [(1, np.arange(len(x_train)))]
    else:
        train_labels = np.argmax(Z_train, axis=1)
        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED,
        )
        folds = [
            (fold, train_idx)
            for fold, (train_idx, _) in enumerate(
                skf.split(x_train, train_labels), start=1
            )
        ]

    # Calentamiento una sola vez por base/protocolo, si quedan
    # ejecuciones pendientes. Usa los datos del primer pliegue.
    pending_keys = [
        make_key(db, protocol, method, fold)
        for fold, _ in folds
        for method in METHODS
    ]
    needs_run = any(key not in completed for key in pending_keys)

    if DO_WARMUP and needs_run:
        first_fold, first_idx = folds[0]
        warmup_train = (
            tf.constant(x_train[first_idx], dtype=tf.float32),
            tf.constant(Y_train[first_idx], dtype=tf.float32),
            tf.constant(M_train[first_idx], dtype=tf.float32),
            tf.constant(Z_train[first_idx], dtype=tf.float32),
        )

        run_warmup(
            db, protocol, warmup_train, val_data,
            input_dim, K, T,
        )

    for fold, train_idx in folds:
        global_fold_counter += 1

        print(
            f"\nPliegue {fold}/"
            f"{1 if db in FIXED_SPLIT_DATASETS else N_SPLITS}"
        )

        train_fold = (
            tf.constant(x_train[train_idx], dtype=tf.float32),
            tf.constant(Y_train[train_idx], dtype=tf.float32),
            tf.constant(M_train[train_idx], dtype=tf.float32),
            tf.constant(Z_train[train_idx], dtype=tf.float32),
        )

        # Rota el orden para evitar ejecutar siempre los métodos
        # en la misma posición temporal.
        offset = (global_fold_counter - 1) % len(METHODS)
        method_order = METHODS[offset:] + METHODS[:offset]

        print("Orden:", " -> ".join(method_order))

        for model_name in method_order:
            key = make_key(db, protocol, model_name, fold)

            if key in completed:
                print(f"{model_name}: ya completado; se omite.")
                continue

            tf.keras.backend.clear_session()
            set_seed(SEED + global_fold_counter)

            if model_name == "Baseline-NN":
                fn = lambda: run_baseline(
                    train_fold, input_dim, K, T,
                    epochs=BASELINE_EPOCHS,
                )
                epochs = BASELINE_EPOCHS
                em_iters = 0
                m_steps = 0
            else:
                fn = lambda m=model_name: run_panda(
                    train_fold, val_data, input_dim, K, T,
                    db, protocol, m,
                    em_iters=EM_ITERS,
                    m_steps=M_STEPS,
                )
                epochs = EM_ITERS * M_STEPS
                em_iters = EM_ITERS
                m_steps = M_STEPS

            elapsed, stats = measure_training(fn)

            row = {
                "dataset": db,
                "category": category,
                "protocol": protocol,
                "model": model_name,
                "fold": fold,
                "train_time_seconds": elapsed,
                "epochs": epochs,
                "em_iters": em_iters,
                "m_steps": m_steps,
                **stats,
            }

            save_result(row, key)
            print(f"{model_name}: {elapsed:.2f} s")

            tf.keras.backend.clear_session()


print(f"\nExperimento terminado. Resultados: {OUTPUT_PATH}")


# =========================
# Resumen de tiempos
# =========================
def build_timing_summary():
    if not os.path.exists(OUTPUT_PATH):
        print("No existe el archivo de tiempos:", OUTPUT_PATH)
        return

    df = pd.read_csv(OUTPUT_PATH)
    df["protocol"] = df["protocol"].fillna("")
    df["train_time_seconds"] = pd.to_numeric(
        df["train_time_seconds"], errors="coerce"
    )

    rows = []

    for (dataset, category, protocol), group in df.groupby(
        ["dataset", "category", "protocol"],
        dropna=False,
        sort=True,
    ):
        row = {
            "dataset": dataset,
            "category": category,
            "protocol": protocol,
        }

        for model in METHODS:
            times = group.loc[
                group["model"] == model, "train_time_seconds"
            ].dropna()

            row[f"{model}_n"] = len(times)
            row[f"{model}_mean_seconds"] = (
                times.mean() if len(times) else np.nan
            )
            row[f"{model}_std_seconds"] = (
                times.std(ddof=1) if len(times) > 1 else np.nan
            )

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        TIMING_SUMMARY_PATH,
        index=False,
        float_format="%.6f",
    )

    print("\n" + "=" * 65)
    print("RESUMEN DE TIEMPOS GENERADO")
    print("=" * 65)
    print(f"Archivo: {TIMING_SUMMARY_PATH}")
    print(f"Bases/protocolos resumidos: {len(summary_df)}")
    print(summary_df.to_string(index=False))


build_timing_summary()
