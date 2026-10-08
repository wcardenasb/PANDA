import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from experiments.data_loader import load_dataset
from panda.training import train_em


nonreal_db_binary = ['Breast', 'Bupa', 'Ionosphere', 'Pima', 'TicTacToe', 'Occupancy', 'Skin_NonSkin']
nonreal_db_multi = ['Iris', 'Wine', 'Segmentation', 'Western']
real_db_binary = ['Voice', 'Polarity']
real_db_multi = ['Music']
synthetic_db = ['blobs', 'circles', 'spirals']

# data_bases = nonreal_db_binary + nonreal_db_multi + real_db_binary + real_db_multi + synthetic_db
# eta_model = ['sym', 'diag', 'full']

data_bases = ['Breast']
eta_model = ['sym']

simulated_db = nonreal_db_binary + nonreal_db_multi

base_path = r'C:\Users\Usuario UTP\Documents\PANDA\data\MADatasets\MADatasets'
results_path = r'C:\Users\Usuario UTP\Documents\PANDA\results'
# models_path = r'C:\Users\Usuario UTP\Documents\PANDA\models'

os.makedirs(results_path, exist_ok=True)

json_path = os.path.join(results_path, 'resumen_global.json')

with open(json_path, 'r') as f:
    resumen = json.load(f)

rows = []
p_errors_list = [0.3]


for db in data_bases:

    print(f"\n{'='*60}\nProcesando base de datos: {db}")

    error_values = p_errors_list if db in simulated_db else [None]

    for p_error in error_values:

        if p_error is not None:
            print(f"\n--- p_error = {p_error:.1f} ---")
            datasets = load_dataset(db, base_path, seed=42, p_errors=p_error)
        else:
            datasets = load_dataset(db, base_path, seed=42)

        for data in datasets:

            dataset_name = data["name"] or db
            train_data = data["train_data"]
            val_data = data["val_data"]

            input_dim = data["input_dim"]
            K = data["K"]
            T = data["T"]

            x_train, Y_train, M_train, Z_train = train_data

            print(f"Dataset: {dataset_name} | input_dim={input_dim} | K={K} | T={T}")

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for eta in eta_model:

                print(f"\n=== Modelo: {eta} ===")

                hyperparams = resumen[dataset_name][eta]["hyperparameters"]

                n_layers = hyperparams["n_layers"]
                nn_list = [hyperparams["units_0"], hyperparams["units_1"], hyperparams["units_2"]]
                lr = hyperparams["Learning_rate"]
                d = hyperparams["d"]

                acc_list, auc_list, f1_list = [], [], []
                history_list, history_val_list = [], []

                for fold_idx, (train_idx, _) in enumerate(skf.split(x_train, Z_train.numpy().argmax(axis=1))):

                    print(f"Fold {fold_idx + 1}/5")

                    train_fold = tuple(tf.gather(x, train_idx) for x in train_data)

                    if eta == 'sym':

                        model, acc, auc, f1, history, history_val = train_em(
                            train=train_fold, val=val_data, input_dim=input_dim, K=K, T=T, d=d,
                            lr=lr, eta_type='symmetric', n_layers=n_layers, nn_list=nn_list,
                            em_iters=25, m_steps=20
                        )

                    elif eta == 'diag':

                        model, acc, auc, f1, history, history_val = train_em(
                            train=train_fold, val=val_data, input_dim=input_dim, K=K, T=T, d=d,
                            lr=lr, eta_type='diagonal', n_layers=n_layers, nn_list=nn_list,
                            em_iters=25, m_steps=20
                        )

                    else:

                        batch_size = hyperparams["batch_size"]

                        model, acc, auc, f1, history, history_val = train_em(
                            train=train_fold, val=val_data, input_dim=input_dim, K=K, T=T, d=d,
                            lr=lr, eta_type='full', batch_size=batch_size,
                            n_layers=n_layers, nn_list=nn_list, em_iters=15, m_steps=15
                        )

                    acc_list.append(acc)
                    auc_list.append(auc)
                    f1_list.append(f1)
                    history_list.append(history)
                    history_val_list.append(history_val)

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


df_resultados = pd.DataFrame(rows)

output_path = os.path.join(results_path, "metrics_cv_histories.pkl")
df_resultados.to_pickle(output_path)

print(f"\nResultados guardados en:\n{output_path}")
print("\n=== VALIDACIÓN CRUZADA FINALIZADA ===")
print(df_resultados[["database", "model", "ACC_mean", "AUC_mean", "F1_mean"]])