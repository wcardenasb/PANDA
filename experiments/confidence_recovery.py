
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from panda.training import train_em
from experiments.data_loader import load_dataset, DATASETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "MADatasets", "MADatasets")
SUMMARY_PATH = os.path.join(ROOT, "results", "resumen_global.json")
OUTPUT_DIR = os.path.join(ROOT, "results", "confidence_recovery")

DATASETS_TO_RUN = [
                    "Breast",
                    "Bupa", "Ionosphere",
                    "Pima", "TicTacToe",
                    "Occupancy", "Skin_NonSkin",
                    "Iris", "Wine",
                    "Segmentation",
                    "Western",
                    ]
VARIANTS = {
            "PANDA-S": "symmetric",
             "PANDA-D": "diagonal",
            "PANDA-F": "full"
             }
SEED = 42
P_ERRORS_LIST = [
                # 0.3,
                0.5,
                # 0.8
                ]
EM_ITERS = 50
M_STEPS = 20
BATCH_SIZE = 128
LR = 1e-5


def get_hyperparameters(dataset, variant):
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"No se encontró el resumen de hiperparámetros: {SUMMARY_PATH}")
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    variant_key = {"PANDA-S": "sym", "PANDA-D": "diag", "PANDA-F": "full"}.get(variant, variant)
    hp = summary[dataset][variant_key]["hyperparameters"]
    batch_size = hp["batch_size"] if variant_key == "full" else BATCH_SIZE
    return {"d": hp["d"], "n_layers": hp["n_layers"], "nn_list": [hp["units_0"], hp["units_1"], hp["units_2"]],
            "batch_size": batch_size, "lr": hp["Learning_rate"]}


def estimate_confidence(model, X, clusters, K, T):
    Pi, P = model.predict_proba(X)
    Pi, P = np.asarray(Pi), np.asarray(P)
    if model.eta_type == "symmetric":
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        features = model.backbone(X_tensor)
        eta = tf.sigmoid(tf.matmul(features, model.W, transpose_b=True) - model.gamma).numpy()
        print("\nConfianza Eta por clúster:")
        for c in range(T):
            mask = clusters == c
            if np.any(mask):
                experto = eta[mask, c].mean()
                otros = np.delete(eta[mask], c, axis=1).mean()
                print(f"Cluster {c+1}: experto={experto:.4f}, otros={otros:.4f}, diferencia={experto-otros:+.4f}")
    diagonal = np.diagonal(P, axis1=2, axis2=3)
    print("Rango diagonal P:", np.min(diagonal), np.max(diagonal))
    print("Pi media por clase:", Pi.mean(axis=0))
    print("Confianza diagonal media por anotador:", diagonal.mean(axis=(0, 2)))
    confidence = np.sum(Pi[:, None, :] * diagonal, axis=2)
    for c in range(T):
        mask = clusters == c
        if np.any(mask):
            experto = confidence[mask, c].mean()
            otros = np.delete(confidence[mask], c, axis=1).mean()
            print(f"Cluster {c+1}: experto={experto:.4f}, otros={otros:.4f}, diferencia={experto-otros:+.4f}")
    table = np.full((T, T), np.nan)
    for cluster_id in range(T):
        mask = clusters == cluster_id
        if np.any(mask):
            table[cluster_id, :] = confidence[mask].mean(axis=0)
    return pd.DataFrame(table, index=[f"Cluster_{i+1}" for i in range(T)], columns=[f"Annotator_{i+1}" for i in range(T)])


def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for p_errors in P_ERRORS_LIST:
        error_tag = f"{p_errors:g}"
        print(f"\n{'#' * 60}\nP_ERRORS: {p_errors}\n{'#' * 60}")
        all_results = []
        for dataset_name in DATASETS_TO_RUN:
            if dataset_name not in DATASETS or DATASETS[dataset_name]["type"] != "simulated":
                print(f"Se omite {dataset_name}: no es un dataset semisintético configurado.")
                continue
            print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
            dataset = load_dataset(dataset_name, DATA_PATH, seed=SEED, p_errors=p_errors)[0]
            X_all, clusters, train_idx = dataset["X_all"], dataset["clusters"], dataset["train_idx"]
            X_train, clusters_train = X_all[train_idx], clusters[train_idx]
            K, T = dataset["K"], dataset["T"]
            train_data, val_data = dataset["train_data"], dataset["val_data"]
            for variant_name, eta_type in VARIANTS.items():
                print(f"Entrenando {variant_name} con p_errors={p_errors}...")
                hp = get_hyperparameters(dataset_name, variant_name)
                tf.keras.utils.set_random_seed(SEED)
                model, best_acc, best_auc, best_f1, history, val_history = train_em(
                    train=train_data, val=val_data, input_dim=dataset["input_dim"], d=hp["d"], K=K, T=T,
                    em_iters=EM_ITERS, m_steps=M_STEPS, batch_size=hp["batch_size"], lr=hp["lr"],
                    n_layers=hp["n_layers"], nn_list=hp["nn_list"], eta_type=eta_type)
                table = estimate_confidence(model, X_train, clusters_train, K, T)
                print(f"\nACC: {best_acc:.4f} | AUC: {best_auc:.4f} | F1: {best_f1:.4f}")
                print(table.round(4))
                for cluster in table.index:
                    for annotator in table.columns:
                        all_results.append({"dataset": dataset_name, "variant": variant_name, "ACC": best_acc,
                                            "AUC": best_auc, "F1": best_f1, "cluster": cluster,
                                            "annotator": annotator, "confidence": table.loc[cluster, annotator]})
        summary_path = os.path.join(OUTPUT_DIR, f"confidence_recovery_summary_{error_tag}_es.csv")
        pd.DataFrame(all_results).assign(p_errors=p_errors).to_csv(summary_path, index=False)
        print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    run_experiment()
