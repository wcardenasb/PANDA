import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from .panda import MultiAnnotatorEM
from .utils import init_full_from_symmetric

# =========================
# EM training loop
# =========================
def train_em(
    train, val,
    input_dim=20, d=16, K=3, T=4,
    em_iters=10, m_steps=2, batch_size=128,
    lr=1e-5, weight_decay=0.0,
    n_layers=3, nn_list=[10,10,10],
    eta_type='symmetric', model_pretrained=None
    # perfect_annotators=True
    ):
    # data
    x_tr, Y_tr, M_tr, Z_tr = train[0], train[1], train[2], train[3]
    x_va, Y_va, M_va, Z_va = val[0], val[1], val[2], val[3]
    N_train = tf.shape(x_tr)[0]
    N_val = tf.shape(x_va)[0]

    # print(symmetric)

    # model & opt
    backbone = make_backbone(input_dim, d, n_layers=n_layers, nn_list=nn_list)
    # backbone = make_backbone(input_dim, d)

    if model_pretrained is not None:
        model = model_pretrained
    else:
        model = MultiAnnotatorEM(d, K, T, backbone, eta_type=eta_type)

    optimizer = tf.keras.optimizers.Adam(lr)
    history = []
    val_history = []

    # weight decay (optional)
    def add_wd(loss):
        if weight_decay <= 0: return loss
        wd = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables]) * weight_decay
        return loss + wd

    # batching helpers
    def batch_indices(N, batch_size):
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            yield s, e

    best_auc = -np.inf
    best_acc = 0.0
    best_weights = None
    # patience = 2
    # patience_counter = 0

    if eta_type=='full':

        # ----- Phase A: symmetric EM warm start -----
        model_sym = MultiAnnotatorEM(d, K, T, backbone,eta_type='symmetric')
        opt_sym = tf.keras.optimizers.Adam(lr)
        sym_em_iters = 8
        sym_m_steps = 8

        for em in range(1, sym_em_iters + 1):
            # ---- E-step ----
            r_tr = model_sym.e_step(x_tr, Y_tr, M_tr, batch_size=batch_size)  # [N_train,K]

            # ---- M-step ----
            for m_iter in range(sym_m_steps):
                for s, e in batch_indices(N_train, batch_size):
                    xb = x_tr[s:e]; yb = Y_tr[s:e]; mb = M_tr[s:e]; rb = r_tr[s:e]
                    with tf.GradientTape() as tape:
                        loss = model_sym.q_loss(xb, yb, mb, rb)
                        loss = add_wd(loss)
                    grads = tape.gradient(loss, model_sym.trainable_variables)
                    opt_sym.apply_gradients(zip(grads, model_sym.trainable_variables))
                    history.append(loss)

                # ---- Metrics on fixed validation set ----
                Pi_val, _ = model_sym.predict_proba(x_va)
                # y_pred = y_prob.numpy().argmax(axis=1)
                y_pred = np.argmax(Pi_val, axis=1)
                y_true = Z_va.numpy().argmax(axis=1)

                # ===== Alinear etiquetas =====
                cm = confusion_matrix(y_true, y_pred)
                row_ind, col_ind = linear_sum_assignment(-cm)
                mapping = {col: row for row, col in zip(row_ind, col_ind)}
                y_pred_aligned = np.vectorize(mapping.get)(y_pred)

                inv_mapping = {v: k for k, v in mapping.items()}
                indices = [inv_mapping[k] for k in range(K)]
                y_prob_aligned = tf.gather(Pi_val, indices=indices, axis=1)

                acc = accuracy_score(y_true, y_pred_aligned)
                if K == 2:
                    # Caso binario:
                    auc = roc_auc_score(y_true, y_prob_aligned[:, 1])
                    f1 = f1_score(y_true, y_pred_aligned, average='binary')
                else:
                    # Caso multiclase:
                    auc = roc_auc_score(y_true, y_prob_aligned, multi_class='ovr')
                    f1 = f1_score(y_true, y_pred_aligned, average='macro')

                # --- check best ---
                if auc > best_auc:
                    best_auc = auc
                    best_acc = acc
                    best_f1 = f1
                    best_weights_sym = model_sym.get_weights()

        if best_weights_sym is not None:
            model_sym.set_weights(best_weights_sym)
        # ----- Phase B: build full head and initialize from symmetric -----
        init_full_from_symmetric(model, model_sym, x_tr, K)

        # ----- Phase C: EM on full head -----

        full_em_iters = em_iters

        for em in range(1, em_iters + 1):
            # ---- E-step ----
            r_tr = model.e_step(x_tr, Y_tr, M_tr, batch_size=batch_size)  # [N_train,K]

            # ---- M-step ----
            for m_iter in range(m_steps):
                for s, e in batch_indices(N_train, batch_size):
                    xb = x_tr[s:e]; yb = Y_tr[s:e]; mb = M_tr[s:e]; rb = r_tr[s:e]
                    with tf.GradientTape() as tape:
                        loss = model.q_loss(xb, yb, mb, rb)
                        loss = add_wd(loss)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                history.append(loss)

                # ---- Metrics on fixed validation set ----
                Pi_val, _ = model.predict_proba(x_va)
                # y_pred = y_prob.numpy().argmax(axis=1)
                y_pred = np.argmax(Pi_val, axis=1)
                y_true = Z_va.numpy().argmax(axis=1)

                # ===== Alinear etiquetas =====
                cm = confusion_matrix(y_true, y_pred)
                row_ind, col_ind = linear_sum_assignment(-cm)
                mapping = {col: row for row, col in zip(row_ind, col_ind)}
                y_pred_aligned = np.vectorize(mapping.get)(y_pred)

                inv_mapping = {v: k for k, v in mapping.items()}
                indices = [inv_mapping[k] for k in range(K)]
                y_prob_aligned = tf.gather(Pi_val, indices=indices, axis=1)

                acc = accuracy_score(y_true, y_pred_aligned)
                if K == 2:
                    # Caso binario:
                    auc = roc_auc_score(y_true, y_prob_aligned[:, 1])
                    f1 = f1_score(y_true, y_pred_aligned, average='binary')
                else:
                    # Caso multiclase:
                    auc = roc_auc_score(y_true, y_prob_aligned, multi_class='ovr')
                    f1 = f1_score(y_true, y_pred_aligned, average='macro')

                # --- check best ---
                if auc > best_auc:
                    best_auc = auc
                    best_acc = acc
                    best_f1 = f1
                    best_weights = model.get_weights()

            #   print(f"[Full] EM {em:02d}: Q-loss={loss.numpy():.4f} | val acc={acc:.3f} | val auc={auc:.3f}")

    else: #diagonal or symmetric

        for em in range(1, em_iters + 1):
            # ---- E-step ----
            r_tr = model.e_step(x_tr, Y_tr, M_tr, batch_size=batch_size)  # [N_train,K]

            r_val = model.e_step(x_va, Y_va, M_va, batch_size=batch_size)

            # ---- M-step ----
            for m_iter in range(m_steps):
                epoch_loss = 0
                n_batches = 0
                for s, e in batch_indices(N_train, batch_size):
                    xb = x_tr[s:e]; yb = Y_tr[s:e]; mb = M_tr[s:e]; rb = r_tr[s:e]
                    with tf.GradientTape() as tape:
                        loss = model.q_loss(xb, yb, mb, rb)
                        loss = add_wd(loss)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    epoch_loss += loss.numpy()
                    n_batches += 1
                # history.append(loss)
                epoch_loss /= n_batches
                history.append(epoch_loss)

                
                val_loss = model.q_loss(x_va, Y_va, M_va, r_val)
                val_history.append(val_loss.numpy())

                # # ---- Metrics on fixed validation set ----
                # Pi_val, _ = model.predict_proba(x_va)
                # y_pred = Pi_val.numpy().argmax(axis=1)
                # y_true = Z_va.numpy().argmax(axis=1)
                # acc = (y_pred == y_true).mean()

                Pi_val, _ = model.predict_proba(x_va)
                # y_pred = y_prob.numpy().argmax(axis=1)
                y_pred = np.argmax(Pi_val, axis=1)
                y_true = Z_va.numpy().argmax(axis=1)

                # ===== Alinear etiquetas =====
                cm = confusion_matrix(y_true, y_pred)
                row_ind, col_ind = linear_sum_assignment(-cm)
                mapping = {col: row for row, col in zip(row_ind, col_ind)}
                y_pred_aligned = np.vectorize(mapping.get)(y_pred)

                inv_mapping = {v: k for k, v in mapping.items()}
                indices = [inv_mapping[k] for k in range(K)]
                y_prob_aligned = tf.gather(Pi_val, indices=indices, axis=1)

                acc = accuracy_score(y_true, y_pred_aligned)
                if K == 2:
                    # Caso binario:
                    auc = roc_auc_score(y_true, y_prob_aligned[:, 1])
                    f1 = f1_score(y_true, y_pred_aligned, average='binary')
                else:
                    # Caso multiclase:
                    auc = roc_auc_score(y_true, y_prob_aligned, multi_class='ovr')
                    f1 = f1_score(y_true, y_pred_aligned, average='macro')

                # if K == 2:
                #     # Caso binario:
                #     auc = roc_auc_score(y_true, Pi_val[:, 1])
                #     f1 = f1_score(y_true, y_pred, average='binary')
                # else:
                #     # Caso multiclase:
                #     auc = roc_auc_score(y_true, Pi_val, multi_class='ovr')
                #     f1 = f1_score(y_true, y_pred, average='macro')

            # print(f"EM iter {em:02d} | M iter {m_iter:02d} | Q-loss={loss.numpy():.4f} | val_acc (prior p(z|x))={acc:.3f}| val_auc={auc:.3f}")

                # --- check best ---
                if auc > best_auc:
                    best_auc = auc
                    best_acc = acc
                    best_f1 = f1
                    best_weights = model.get_weights()

            #   print(f"EM iter {em:02d} | Q-loss={loss.numpy():.4f} | val_acc={acc:.3f} | val_auc={auc:.3f}")

    # --- al final restaurar mejor modelo ---
    if best_weights is not None:
        model.set_weights(best_weights)
    return model, best_acc, best_auc, best_f1, np.array(history), np.array(val_history)

# @register_keras_serializable()
# Example backbone: simple 2-layer MLP
def make_backbone(input_dim, d, n_layers, nn_list):
    inputs = layers.Input(shape=(input_dim,))
    h = inputs
    for i in range(n_layers):
        h = layers.Dense(nn_list[i], activation='relu', use_bias=False)(h)
    outputs = layers.Dense(d)(h)
    return Model(inputs, outputs)