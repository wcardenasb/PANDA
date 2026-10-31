import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from experiments.data_loader import load_dataset
from panda.training_timing import train_em_timing


# ============================================================
# CONFIGURACIÓN
# ============================================================

nonreal_db_binary = [
    'Breast', 'Bupa', 'Ionosphere', 'Pima',
    'TicTacToe', 'Occupancy', 'Skin_NonSkin'
]

nonreal_db_multi = [
    'Iris', 'Wine', 'Segmentation', 'Western'
]

real_db_binary = ['Voice', 'Polarity']
real_db_multi = ['Music']
# synthetic_db = ['blobs', 'circles', 'spirals']

data_bases = (
    nonreal_db_binary
    + nonreal_db_multi
    + real_db_binary
    + real_db_multi
    # + synthetic_db
)

eta_model = ['sym', 'diag', 'full']

simulated_db = nonreal_db_binary + nonreal_db_multi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

base_path = os.path.join(ROOT, "data", "MADatasets", "MADatasets")
results_path = os.path.join(ROOT, "results")

timing_path = os.path.join(results_path, 'timing')

os.makedirs(results_path, exist_ok=True)
os.makedirs(timing_path, exist_ok=True)


# ============================================================
# ARCHIVOS DE ENTRADA Y SALIDA
# ============================================================

json_path = os.path.join(results_path, 'resumen_global.json')

histories_path = os.path.join(
    results_path,
    'metrics_cv_histories_timing.pkl'
)

timing_folds_path = os.path.join(
    timing_path,
    'timing_folds_em_breakdown.csv'
)

timing_summary_path = os.path.join(
    timing_path,
    'timing_summary_em_breakdown.csv'
)


# ============================================================
# HIPERPARÁMETROS
# ============================================================

with open(json_path, 'r') as f:
    resumen = json.load(f)

p_errors_list = [0.3]


# ============================================================
# CONTENEDORES DE RESULTADOS
# ============================================================

rows = []
timing_rows = []


# ============================================================
# EXPERIMENTO
# ============================================================

for db in data_bases:

    print(f"\n{'=' * 60}")
    print(f"Procesando base de datos: {db}")

    error_values = p_errors_list if db in simulated_db else [None]

    for p_error in error_values:

        if p_error is not None:
            print(f"\n--- p_error = {p_error:.1f} ---")

            datasets = load_dataset(
                db,
                base_path,
                seed=42,
                p_errors=p_error
            )
        else:
            datasets = load_dataset(
                db,
                base_path,
                seed=42
            )

        for data in datasets:

            dataset_name = data["name"] or db

            train_data = data["train_data"]
            val_data = data["val_data"]

            input_dim = data["input_dim"]
            K = data["K"]
            T = data["T"]

            x_train, Y_train, M_train, Z_train = train_data

            print(
                f"\nDataset: {dataset_name} | "
                f"input_dim={input_dim} | K={K} | T={T}"
            )

            skf = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            for eta in eta_model:

                print(f"\n=== Modelo: {eta} ===")

                hyperparams = resumen[dataset_name][eta][
                    "hyperparameters"
                ]

                n_layers = hyperparams["n_layers"]

                nn_list = [
                    hyperparams["units_0"],
                    hyperparams["units_1"],
                    hyperparams["units_2"]
                ]

                lr = hyperparams["Learning_rate"]
                d = hyperparams["d"]

                # Listas para métricas e histories
                acc_list = []
                auc_list = []
                f1_list = []

                history_list = []
                history_val_list = []

                # ====================================================
                # CROSS VALIDATION
                # ====================================================

                for fold_idx, (train_idx, _) in enumerate(
                    skf.split(
                        x_train,
                        Z_train.numpy().argmax(axis=1)
                    )
                ):

                    print(f"\nFold {fold_idx + 1}/5")

                    train_fold = tuple(
                        tf.gather(x, train_idx)
                        for x in train_data
                    )

                    # Parámetros específicos de cada modelo
                    if eta == 'sym':

                        eta_type = 'symmetric'
                        em_iters = 25
                        m_steps = 10
                        batch_size = 128

                    elif eta == 'diag':

                        eta_type = 'diagonal'
                        em_iters = 25
                        m_steps = 10
                        batch_size = 128

                    else:

                        eta_type = 'full'
                        em_iters = 15
                        m_steps = 10
                        batch_size = hyperparams["batch_size"]

                    # =================================================
                    # ENTRENAMIENTO + HISTORIES + TIEMPOS
                    # =================================================

                    (
                        model,
                        acc,
                        auc,
                        f1,
                        history,
                        history_val,
                        timing
                    ) = train_em_timing(
                        train=train_fold,
                        val=val_data,
                        input_dim=input_dim,
                        K=K,
                        T=T,
                        d=d,
                        lr=lr,
                        eta_type=eta_type,
                        batch_size=batch_size,
                        n_layers=n_layers,
                        nn_list=nn_list,
                        em_iters=em_iters,
                        m_steps=m_steps
                    )

                    # =================================================
                    # GUARDAR MÉTRICAS E HISTORIES
                    # =================================================

                    acc_list.append(acc)
                    auc_list.append(auc)
                    f1_list.append(f1)

                    history_list.append(history)
                    history_val_list.append(history_val)

                    # =================================================
                    # GUARDAR TIEMPOS POR FOLD
                    # =================================================

                    timing_row = {
                        "database": dataset_name,
                        "model": eta,
                        "fold": fold_idx + 1,

                        "total_time_seconds":
                            timing["total_time_seconds"],

                        "e_total_seconds":
                            timing["e_total_seconds"],

                        "e_mean_seconds":
                            timing["e_mean_seconds"],

                        "e_steps_count":
                            timing["e_steps_count"],

                        "m_total_seconds":
                            timing["m_total_seconds"],

                        "m_mean_seconds":
                            timing["m_mean_seconds"],

                        "m_steps_count":
                            timing["m_steps_count"],

                        "other_time_seconds":
                            timing["other_time_seconds"]
                    }

                    if p_error is not None:
                        timing_row["p_error"] = p_error

                    timing_rows.append(timing_row)

                    # Guardado progresivo de tiempos
                    pd.DataFrame(timing_rows).to_csv(
                        timing_folds_path,
                        index=False
                    )

                    print(
                        f"ACC={acc:.4f} | "
                        f"AUC={auc:.4f} | "
                        f"F1={f1:.4f}"
                    )

                    print(
                        f"Tiempo total: "
                        f"{timing['total_time_seconds']:.2f} s | "
                        f"E-step: "
                        f"{timing['e_total_seconds']:.2f} s | "
                        f"M-step: "
                        f"{timing['m_total_seconds']:.2f} s"
                    )

                    # Liberar memoria entre folds
                    del model
                    tf.keras.backend.clear_session()

                # ====================================================
                # RESUMEN DE MÉTRICAS E HISTORIES
                # ====================================================

                row = {
                    "database": dataset_name,
                    "model": eta,

                    "ACC_mean": np.mean(acc_list),
                    "ACC_std": np.std(acc_list),

                    "AUC_mean": np.mean(auc_list),
                    "AUC_std": np.std(auc_list),

                    "F1_mean": np.mean(f1_list),
                    "F1_std": np.std(f1_list),

                    "history": history_list,
                    "history_val": history_val_list
                }

                if p_error is not None:
                    row["p_error"] = p_error

                rows.append(row)

                # Guardado progresivo de histories
                pd.DataFrame(rows).to_pickle(histories_path)

                print(
                    f"\nResultados de {dataset_name} - {eta} guardados."
                )


# ============================================================
# RESUMEN FINAL DE TIEMPOS
# ============================================================

df_timing = pd.DataFrame(timing_rows)

if not df_timing.empty:

    group_columns = ["database", "model"]

    if "p_error" in df_timing.columns:
        group_columns.append("p_error")

    timing_summary = (
        df_timing
        .groupby(group_columns)
        .agg(
            total_time_mean=("total_time_seconds", "mean"),
            total_time_std=("total_time_seconds", "std"),

            e_total_mean=("e_total_seconds", "mean"),
            e_total_std=("e_total_seconds", "std"),

            e_mean=("e_mean_seconds", "mean"),

            e_steps_mean=("e_steps_count", "mean"),

            m_total_mean=("m_total_seconds", "mean"),
            m_total_std=("m_total_seconds", "std"),

            m_mean=("m_mean_seconds", "mean"),

            m_steps_mean=("m_steps_count", "mean"),

            other_time_mean=("other_time_seconds", "mean"),
            other_time_std=("other_time_seconds", "std")
        )
        .reset_index()
    )

    timing_summary.to_csv(
        timing_summary_path,
        index=False
    )


# ============================================================
# GUARDADO FINAL
# ============================================================

df_resultados = pd.DataFrame(rows)

df_resultados.to_pickle(histories_path)

print("\n" + "=" * 60)
print("EXPERIMENTO FINALIZADO")
print("=" * 60)

print(f"\nHistories y métricas:\n{histories_path}")

print(f"\nTiempos por fold:\n{timing_folds_path}")

print(f"\nResumen de tiempos:\n{timing_summary_path}")

print("\n=== MÉTRICAS FINALES ===")

if not df_resultados.empty:
    print(
        df_resultados[
            ["database", "model", "ACC_mean", "AUC_mean", "F1_mean"]
        ]
    )