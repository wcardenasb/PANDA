import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from experiments.data_loader import load_dataset
from panda.training import train_em


# ============================================================
# CONFIGURACIÓN
# ============================================================

nonreal_db_binary = [
    'Breast',
    'Bupa',
    'Ionosphere',
    'Pima',
    'TicTacToe', 'Occupancy', 'Skin_NonSkin'
]

nonreal_db_multi = [
    'Iris',
    'Wine', 'Segmentation', 'Western'
]

real_db_binary = ['Voice', 'Polarity']
real_db_multi = ['Music']

# synthetic_db = ['blobs', 'circles', 'spirals']

data_bases = (
    nonreal_db_binary
    + nonreal_db_multi
    + real_db_binary
    # + real_db_multi
)

eta_model = ['sym', 'diag', 'full']

simulated_db = nonreal_db_binary + nonreal_db_multi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

base_path = os.path.join(ROOT, "data", "MADatasets", "MADatasets")
results_path = os.path.join(ROOT, "results")

os.makedirs(results_path, exist_ok=True)


# ============================================================
# ARCHIVOS
# ============================================================

json_path = os.path.join(
    results_path,
    'resumen_global.json'
)

histories_path = os.path.join(
    results_path,
    'metrics_cv_histories_training2.pkl'
)


# ============================================================
# HIPERPARÁMETROS
# ============================================================

with open(json_path, 'r') as f:
    resumen = json.load(f)

p_errors_list = [0.3]


# ============================================================
# CONTENEDOR DE RESULTADOS
# ============================================================

rows = []


# ============================================================
# EXPERIMENTO
# ============================================================

for db in data_bases:

    print(f"\n{'=' * 60}")
    print(f"Procesando base de datos: {db}")

    error_values = (
        p_errors_list
        if db in simulated_db
        else [None]
    )

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

                # Listas de resultados
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

                    # Parámetros por modelo
                    if eta == 'sym':

                        eta_type = 'symmetric'
                        em_iters = 25
                        m_steps = 5
                        batch_size = 128

                    elif eta == 'diag':

                        eta_type = 'diagonal'
                        em_iters = 25
                        m_steps = 5
                        batch_size = 128

                    else:

                        eta_type = 'full'
                        em_iters = 15
                        m_steps = 5
                        batch_size = hyperparams["batch_size"]

                    # =================================================
                    # ENTRENAMIENTO
                    # =================================================

                    (
                        model,
                        acc,
                        auc,
                        f1,
                        history,
                        history_val
                    ) = train_em(
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
                        m_steps=m_steps,
                        weight_decay=1e-2,
                        min_em_iters=5,
                        patience=3,
                        min_delta=0.001,
                    )

                    # =================================================
                    # GUARDAR MÉTRICAS E HISTORIES
                    # =================================================

                    acc_list.append(acc)
                    auc_list.append(auc)
                    f1_list.append(f1)

                    history_list.append(history)
                    history_val_list.append(history_val)

                    print(
                        f"ACC={acc:.4f} | "
                        f"AUC={auc:.4f} | "
                        f"F1={f1:.4f} | "
                        f"Épocas M={len(history)}"
                    )

                    # Liberar memoria
                    del model
                    tf.keras.backend.clear_session()

                # ====================================================
                # RESUMEN POR BASE DE DATOS Y MODELO
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

                # Guardado progresivo
                pd.DataFrame(rows).to_pickle(
                    histories_path
                )

                print(
                    f"\nResultados de {dataset_name} - {eta} guardados."
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

print("\n=== MÉTRICAS FINALES ===")

if not df_resultados.empty:

    print(
        df_resultados[
            [
                "database",
                "model",
                "ACC_mean",
                "AUC_mean",
                "F1_mean"
            ]
        ]
    )