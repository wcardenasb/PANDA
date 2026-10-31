import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from .panda import MultiAnnotatorEM
from .utils import init_full_from_symmetric


def train_em_timing(
    train, val,
    input_dim=20, d=16, K=3, T=4,
    em_iters=10, m_steps=2, batch_size=128,
    lr=1e-5, weight_decay=1e-4,
    n_layers=3, nn_list=[10, 10, 10],
    eta_type='symmetric', model_pretrained=None,
    min_em_iters=10, patience=5, min_delta=0.001
):
    # =========================
    # Data
    # =========================
    x_tr, Y_tr, M_tr, Z_tr = train
    x_va, Y_va, M_va, Z_va = val

    N_train = tf.shape(x_tr)[0]

    # =========================
    # Model and optimizer
    # =========================
    backbone = make_backbone(input_dim, d, n_layers, nn_list)

    if model_pretrained is not None:
        model = model_pretrained
    else:
        model = MultiAnnotatorEM(
            d, K, T, backbone, eta_type=eta_type
        )

    optimizer = tf.keras.optimizers.Adam(lr)

    history = []
    val_history = []
    history_phase = []

    # =========================
    # Timing
    # =========================
    timing = {
        "total_time_seconds": 0.0,
        "e_total_seconds": 0.0,
        "e_mean_seconds": 0.0,
        "e_steps_count": 0,
        "m_total_seconds": 0.0,
        "m_mean_seconds": 0.0,
        "m_steps_count": 0,
        "other_time_seconds": 0.0
    }

    total_start = time.perf_counter()

    def synchronize(model):
        """Fuerza la sincronización de operaciones pendientes de TensorFlow."""
        if model.trainable_variables:
            _ = model.trainable_variables[0].numpy()

    def run_e_step(current_model):
        start = time.perf_counter()

        responsibilities = current_model.e_step(
            x_tr, Y_tr, M_tr, batch_size=batch_size
        )

        # Sincronizar antes de detener el cronómetro
        _ = responsibilities.numpy()

        elapsed = time.perf_counter() - start

        timing["e_total_seconds"] += elapsed
        timing["e_steps_count"] += 1

        return responsibilities

    def run_m_step(current_model, current_optimizer, responsibilities):
        start = time.perf_counter()

        batch_losses = []

        for s, e in batch_indices(N_train, batch_size):
            xb = x_tr[s:e]
            yb = Y_tr[s:e]
            mb = M_tr[s:e]
            rb = responsibilities[s:e]

            with tf.GradientTape() as tape:
                loss = current_model.q_loss(xb, yb, mb, rb)

                if weight_decay > 0:
                    wd = tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in current_model.trainable_variables
                    ]) * weight_decay

                    loss = loss + wd

            grads = tape.gradient(
                loss, current_model.trainable_variables
            )

            current_optimizer.apply_gradients(
                zip(grads, current_model.trainable_variables)
            )

            batch_losses.append(float(loss.numpy()))

        synchronize(current_model)

        elapsed = time.perf_counter() - start

        timing["m_total_seconds"] += elapsed
        timing["m_steps_count"] += 1

        return float(np.mean(batch_losses))

    def evaluate_validation_loss(current_model):
        return validation_nll(
            current_model, x_va, Y_va, M_va
        )

    def batch_indices(N, batch_size):
        for s in range(0, int(N), batch_size):
            e = min(s + batch_size, int(N))
            yield s, e

    def calculate_metrics(current_model):
        Pi_val, _ = current_model.predict_proba(x_va)

        y_pred = np.argmax(Pi_val, axis=1)
        y_true = Z_va.numpy().argmax(axis=1)

        cm = confusion_matrix(
            y_true, y_pred, labels=np.arange(K)
        )

        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {
            col: row for row, col in zip(row_ind, col_ind)
        }

        y_pred_aligned = np.vectorize(mapping.get)(y_pred)

        inv_mapping = {
            v: k for k, v in mapping.items()
        }

        indices = [
            inv_mapping[k] for k in range(K)
        ]

        y_prob_aligned = tf.gather(
            Pi_val, indices=indices, axis=1
        )

        acc = accuracy_score(y_true, y_pred_aligned)

        if K == 2:
            auc = roc_auc_score(
                y_true, y_prob_aligned[:, 1]
            )
            f1 = f1_score(
                y_true, y_pred_aligned, average='binary'
            )
        else:
            auc = roc_auc_score(
                y_true,
                y_prob_aligned,
                multi_class='ovr'
            )
            f1 = f1_score(
                y_true,
                y_pred_aligned,
                average='macro'
            )

        return acc, auc, f1

    # =========================
    # Best model variables
    # =========================
    best_auc = -np.inf
    best_acc = 0.0
    best_f1 = 0.0
    best_weights = None
    patience_counter = 0

    # =========================
    # PANDA-F
    # =========================
    if eta_type == 'full':

        # ----- Phase A: symmetric EM warm start -----
        model_sym = MultiAnnotatorEM(
            d, K, T, backbone, eta_type='symmetric'
        )

        opt_sym = tf.keras.optimizers.Adam(lr)

        sym_em_iters = 8
        sym_m_steps = 8

        best_weights_sym = None

        for em in range(1, sym_em_iters + 1):

            # E-step
            r_tr = run_e_step(model_sym)

            # M-steps
            for m_iter in range(sym_m_steps):

                epoch_loss = run_m_step(
                    model_sym, opt_sym, r_tr
                )

                history.append(epoch_loss)
                history_phase.append("symmetric_warmup")

                # Validation loss
                val_loss = evaluate_validation_loss(model_sym)
                val_history.append(val_loss)

                # Metrics on validation set
                acc, auc, f1 = calculate_metrics(model_sym)

                if auc > best_auc:
                    best_auc = auc
                    best_acc = acc
                    best_f1 = f1
                    best_weights_sym = model_sym.get_weights()

        if best_weights_sym is not None:
            model_sym.set_weights(best_weights_sym)

        # ----- Phase B: initialize full model -----
        init_full_from_symmetric(
            model, model_sym, x_tr, K
        )

        # Reset best model selection for full phase
        best_auc = -np.inf
        best_acc = 0.0
        best_f1 = 0.0
        best_weights = None
        patience_counter = 0

        # ----- Phase C: full EM -----
        for em in range(1, em_iters + 1):

            # E-step
            r_tr = run_e_step(model)

            # M-steps
            for m_iter in range(m_steps):

                epoch_loss = run_m_step(
                    model, optimizer, r_tr
                )

                history.append(epoch_loss)
                history_phase.append("full")

                # Validation loss
                val_loss = evaluate_validation_loss(model)
                val_history.append(val_loss)

            # Metrics at the end of each EM iteration
            acc, auc, f1 = calculate_metrics(model)

            # Best model selection and patience
            if auc > best_auc + min_delta:
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_weights = model.get_weights()
                patience_counter = 0

            elif em >= min_em_iters:
                patience_counter += 1

            if (
                em >= min_em_iters
                and patience_counter >= patience
            ):
                print(
                    f"[Full] Early stopping en EM {em}. "
                    f"Mejor AUC={best_auc:.4f}"
                )
                break

    # =========================
    # PANDA-S and PANDA-D
    # =========================
    else:

        for em in range(1, em_iters + 1):

            # E-step
            r_tr = run_e_step(model)

            # M-steps
            for m_iter in range(m_steps):

                epoch_loss = run_m_step(
                    model, optimizer, r_tr
                )

                history.append(epoch_loss)
                history_phase.append(eta_type)

                # Validation loss
                val_loss = evaluate_validation_loss(model)
                val_history.append(val_loss)

            # Metrics at the end of each EM iteration
            acc, auc, f1 = calculate_metrics(model)

            # Best model selection and patience
            if auc > best_auc + min_delta:
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_weights = model.get_weights()
                patience_counter = 0

            elif em >= min_em_iters:
                patience_counter += 1

            if (
                em >= min_em_iters
                and patience_counter >= patience
            ):
                print(
                    f"Early stopping en EM {em}. "
                    f"Mejor AUC={best_auc:.4f}"
                )
                break

    # =========================
    # Restore best model
    # =========================
    if best_weights is not None:
        model.set_weights(best_weights)

    # =========================
    # Final timing
    # =========================
    timing["total_time_seconds"] = (
        time.perf_counter() - total_start
    )

    timing["e_mean_seconds"] = (
        timing["e_total_seconds"] / timing["e_steps_count"]
        if timing["e_steps_count"] > 0 else 0.0
    )

    timing["m_mean_seconds"] = (
        timing["m_total_seconds"] / timing["m_steps_count"]
        if timing["m_steps_count"] > 0 else 0.0
    )

    timing["other_time_seconds"] = max(
        0.0,
        timing["total_time_seconds"]
        - timing["e_total_seconds"]
        - timing["m_total_seconds"]
    )

    return (
        model,
        best_acc,
        best_auc,
        best_f1,
        np.array(history),
        np.array(val_history),
        timing
    )


# =========================
# Backbone
# =========================
def make_backbone(input_dim, d, n_layers, nn_list):
    inputs = layers.Input(shape=(input_dim,))
    h = inputs

    for i in range(n_layers):
        h = layers.Dense(
            nn_list[i],
            activation='relu',
            use_bias=False
        )(h)

    outputs = layers.Dense(d)(h)

    return Model(inputs, outputs)


# =========================
# Validation NLL
# =========================
def validation_nll(model, x, Y, M):
    Pi, P = model.predict_proba(x)

    logP = tf.math.log(
        tf.clip_by_value(P, 1e-12, 1.0)
    )

    log_pi = tf.math.log(
        tf.clip_by_value(Pi, 1e-12, 1.0)
    )

    log_py_k = tf.einsum(
        'ntl,ntkl->ntk', Y, logP
    )

    log_py_k = tf.reduce_sum(
        log_py_k * M[:, :, None], axis=1
    )

    return -tf.reduce_mean(
        tf.reduce_logsumexp(
            log_pi + log_py_k, axis=1
        )
    ).numpy()