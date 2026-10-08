import numpy as np
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tensorflow.keras import layers, Model

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment


# ============================================================
# Utilities
# ============================================================

def macro_auc(probs, Z_onehot):
    """Macro one-vs-rest AUC over K classes."""
    K = probs.shape[1]
    aucs = []
    for k in range(K):
        auc = tf.keras.metrics.AUC(curve="ROC", from_logits=False)
        auc.update_state(Z_onehot[:, k], probs[:, k])
        aucs.append(auc.result().numpy())
    return float(np.mean(aucs))

def symmetric_eta_mean(sym_model, x_tr):
    # average agreement prob per annotator under symmetric model
    X = sym_model.backbone(x_tr)
    logits_eta = tf.matmul(X, sym_model.W, transpose_b=True) - sym_model.gamma
    Eta = tf.nn.sigmoid(logits_eta).numpy()  # [N,T]
    return Eta.mean(axis=0)                  # [T]      

def init_full_from_symmetric(full_model, sym_model, x_tr, K, clip=1e-4):
    # copy backbone + prior head
    for v_full, v_sym in zip(full_model.backbone.trainable_variables, sym_model.backbone.trainable_variables):
        v_full.assign(v_sym)
    full_model.V.assign(sym_model.V)
    full_model.b.assign(sym_model.b)

    # symmetric probabilities per annotator
    eta_mean = symmetric_eta_mean(sym_model, x_tr)  # [T]
    p_diag = np.clip(eta_mean, clip, 1. - clip)
    p_off  = np.clip((1. - eta_mean) / (K - 1), clip, 1. - clip)

    # U = 0; set Gamma so that softmax(-Gamma) = symmetric probs
    full_model.U.assign(tf.zeros_like(full_model.U))
    G = np.zeros(full_model.Gamma.shape, dtype=np.float32)  # [T,K,K]
    for t in range(G.shape[0]):
        G[t, :, :] = -np.log(p_off[t])
        np.fill_diagonal(G[t], -np.log(p_diag[t]))
    full_model.Gamma.assign(G)

def simular_etiquetadores(X, y, T=3, cluster="kmeans", seed=42, p_errors=0.0):
    # np.random.seed(seed)

    # Clustering para asignar expertise
    if cluster == "kmeans":
        modelo = KMeans(n_clusters=T, random_state=seed)
        clusters = modelo.fit_predict(X)
    elif cluster == "gmm":
        modelo = GaussianMixture(n_components=T, random_state=seed)
        clusters = modelo.fit_predict(X)
    else:
        raise ValueError("agrupamiento debe ser 'kmeans' o 'gmm'")

    clases_unicas = np.unique(y)
    K = len(clases_unicas)
    N = X.shape[0]
    M = np.ones((N,T))
    y = y.ravel()
    Z = tf.one_hot(y, depth=K).numpy()
    
    y_labelers = np.full((len(X), T), None, dtype=object)  # usar None en vez de np.nan

    for i in range(T): 
        is_expert = clusters == i
        not_expert = ~is_expert

        y_labelers[is_expert, i] = y[is_expert]
        y_labelers[not_expert, i] = y[not_expert]

        if p_errors > 0.0:
            n_errors = int(p_errors * np.sum(not_expert))
            error_indices = np.random.choice(np.where(not_expert)[0], size=n_errors, replace=False)
            for idx in error_indices:
                clases_posibles = [c for c in clases_unicas if c != y[idx]]
                y_labelers[idx, i] = int(np.random.choice(clases_posibles))

    assert None not in y_labelers, "Hay etiquetas sin asignar"
    return tf.one_hot(y_labelers, depth=K).numpy(), M, Z