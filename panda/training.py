import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from .panda import MultiAnnotatorEM
from .utils import init_full_from_symmetric


def train_em(
    train, val,
    input_dim=20, d=16, K=3, T=4,
    em_iters=10, m_steps=2, batch_size=128,
    lr=1e-5, weight_decay=0.0,
    n_layers=3, nn_list=[10, 10, 10],
    eta_type='symmetric', model_pretrained=None,
    min_em_iters=10, patience=5, min_delta=0.0
):
    """
    EM training with per-M-epoch training and validation Q-loss histories.
    Early stopping and best-weight restoration use validation AUC only.
    """

    # ========================================================
    # DATA AND MODEL
    # ========================================================

    x_tr, Y_tr, M_tr, Z_tr = train
    x_va, Y_va, M_va, Z_va = val

    N_train = int(tf.shape(x_tr)[0])
    N_val = int(tf.shape(x_va)[0])

    backbone = make_backbone(
        input_dim, d, n_layers=n_layers, nn_list=nn_list
    )

    model = model_pretrained if model_pretrained is not None else MultiAnnotatorEM(
        d, K, T, backbone, eta_type=eta_type
    )

    optimizer = tf.keras.optimizers.Adam(lr)

    history = []
    val_history = []

    # ========================================================
    # HELPERS
    # ========================================================

    def batch_indices(n):
        for s in range(0, n, batch_size):
            yield s, min(s + batch_size, n)

    def add_wd(q_loss, current_model):
        if weight_decay <= 0:
            return q_loss

        penalty = tf.add_n([
            tf.nn.l2_loss(v)
            for v in current_model.trainable_variables
        ])

        return q_loss + weight_decay * penalty

    def validation_q_loss(current_model, responsibilities):
        total = 0.0

        for s, e in batch_indices(N_val):
            q = current_model.q_loss(
                x_va[s:e], Y_va[s:e],
                M_va[s:e], responsibilities[s:e]
            )
            total += float(q.numpy()) * (e - s)

        return total / N_val

    def m_epoch(current_model, current_optimizer, responsibilities):
        total_q = 0.0

        for s, e in batch_indices(N_train):

            xb = x_tr[s:e]
            yb = Y_tr[s:e]
            mb = M_tr[s:e]
            rb = responsibilities[s:e]

            with tf.GradientTape() as tape:
                q_loss = current_model.q_loss(xb, yb, mb, rb)
                objective = add_wd(q_loss, current_model)

            grads = tape.gradient(
                objective, current_model.trainable_variables
            )

            current_optimizer.apply_gradients(
                zip(grads, current_model.trainable_variables)
            )

            # Record Q-loss without L2 penalty.
            total_q += float(q_loss.numpy()) * (e - s)

        return total_q / N_train

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

        y_pred_aligned = np.array([
            mapping[k] for k in y_pred
        ])

        inv_mapping = {v: k for k, v in mapping.items()}
        indices = [inv_mapping[k] for k in range(K)]

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
                y_true, y_prob_aligned, multi_class='ovr'
            )
            f1 = f1_score(
                y_true, y_pred_aligned, average='macro'
            )

        return acc, auc, f1

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_auc = -np.inf
    best_acc = 0.0
    best_f1 = 0.0
    best_weights = None
    patience_counter = 0

    # ========================================================
    # PANDA-F
    # ========================================================

    if eta_type == 'full':

        # ----------------------------------------------------
        # PHASE A: SYMMETRIC WARM START
        # ----------------------------------------------------

        model_sym = MultiAnnotatorEM(
            d, K, T, backbone, eta_type='symmetric'
        )

        opt_sym = tf.keras.optimizers.Adam(lr)

        best_auc_sym = -np.inf
        best_weights_sym = None

        sym_em_iters = 8
        sym_m_steps = 8

        for em in range(1, sym_em_iters + 1):

            r_tr = model_sym.e_step(
                x_tr, Y_tr, M_tr, batch_size=batch_size
            )

            r_va = model_sym.e_step(
                x_va, Y_va, M_va, batch_size=batch_size
            )

            for _ in range(sym_m_steps):

                history.append(
                    m_epoch(model_sym, opt_sym, r_tr)
                )

                val_history.append(
                    validation_q_loss(model_sym, r_va)
                )

                _, auc, _ = calculate_metrics(model_sym)

                if auc > best_auc_sym:
                    best_auc_sym = auc
                    best_weights_sym = model_sym.get_weights()

        if best_weights_sym is not None:
            model_sym.set_weights(best_weights_sym)

        # ----------------------------------------------------
        # PHASE B: INITIALIZE FULL MODEL
        # ----------------------------------------------------

        init_full_from_symmetric(
            model, model_sym, x_tr, K
        )

        best_auc = -np.inf
        best_acc = 0.0
        best_f1 = 0.0
        best_weights = None
        patience_counter = 0

        # ----------------------------------------------------
        # PHASE C: FULL EM
        # ----------------------------------------------------

        for em in range(1, em_iters + 1):

            r_tr = model.e_step(
                x_tr, Y_tr, M_tr, batch_size=batch_size
            )

            r_va = model.e_step(
                x_va, Y_va, M_va, batch_size=batch_size
            )

            for _ in range(m_steps):

                history.append(
                    m_epoch(model, optimizer, r_tr)
                )

                val_history.append(
                    validation_q_loss(model, r_va)
                )

            acc, auc, f1 = calculate_metrics(model)

            if auc > best_auc + min_delta:

                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_weights = model.get_weights()
                patience_counter = 0

            elif em >= min_em_iters:
                patience_counter += 1

            if em >= min_em_iters and patience_counter >= patience:

                print(
                    f"[Full] Early stopping en EM {em}. "
                    f"Mejor AUC={best_auc:.4f}"
                )

                break

    # ========================================================
    # PANDA-S AND PANDA-D
    # ========================================================

    else:

        for em in range(1, em_iters + 1):

            r_tr = model.e_step(
                x_tr, Y_tr, M_tr, batch_size=batch_size
            )

            r_va = model.e_step(
                x_va, Y_va, M_va, batch_size=batch_size
            )

            for _ in range(m_steps):

                history.append(
                    m_epoch(model, optimizer, r_tr)
                )

                val_history.append(
                    validation_q_loss(model, r_va)
                )

            acc, auc, f1 = calculate_metrics(model)

            if auc > best_auc + min_delta:

                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_weights = model.get_weights()
                patience_counter = 0

            elif em >= min_em_iters:
                patience_counter += 1

            if em >= min_em_iters and patience_counter >= patience:

                print(
                    f"Early stopping en EM {em}. "
                    f"Mejor AUC={best_auc:.4f}"
                )

                break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_weights is not None:
        model.set_weights(best_weights)

    return (
        model,
        best_acc,
        best_auc,
        best_f1,
        np.array(history),
        np.array(val_history)
    )


# ============================================================
# BACKBONE
# ============================================================

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
