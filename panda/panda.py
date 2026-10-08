import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

# =========================
# Multi-annotator model (symmetric head) — EM style
# =========================
@register_keras_serializable()
class MultiAnnotatorEM(tf.keras.Model):
    def __init__(self, d, K, T, backbone,eta_type='symmetric', **kwargs):
        super().__init__(**kwargs)
        self.d, self.K, self.T = d, K, T
        self.backbone = backbone
        self.eta_type = eta_type

        # p(z|x) head
        self.V = self.add_weight(shape=(K, d), initializer="glorot_uniform", name="V")
        self.b = self.add_weight(shape=(K,), initializer="zeros", name="b")

        if eta_type == 'symmetric':
            # # symmetric annotator head: η_t(x) = σ(w_t^T x - γ_t)
            # self.W = self.add_weight(shape=(T, d), initializer="glorot_uniform", name="W")
            # self.gamma = self.add_weight(shape=(T,), initializer="zeros", name="gamma")
            # [T, d] weights and [T] biases
            self.W = self.add_weight(shape=(T, d),
                                     initializer="glorot_uniform",
                                     trainable=True, name="W")
            self.gamma = self.add_weight(shape=(T,),
                                         initializer="zeros",
                                         trainable=True, name="gamma")
        elif eta_type == 'full':
            # Full confusion: [T, K, K, d] weights, [T, K, K] biases
            self.U = self.add_weight(shape=(T, K, K, d),
                                     initializer="glorot_uniform",
                                     trainable=True, name="U")
            self.Gamma = self.add_weight(shape=(T, K, K),
                                         initializer="zeros",
                                         trainable=True, name="Gamma")
        elif eta_type == 'diagonal':
            # Diagonal confusion: [T, K, d] weights, [T, K] biases
            self.U = self.add_weight(shape=(T, K, d),
                                     initializer="glorot_uniform",
                                     trainable=True, name="U")
            self.Gamma = self.add_weight(shape=(T, K),
                                         initializer="zeros",
                                         trainable=True, name="Gamma")

    # ----- helper: components -----
    def _pi_and_eta(self, x_in):
        X = self.backbone(x_in)                               # [N,d]
        logits_pi = tf.matmul(X, self.V, transpose_b=True) - self.b   # [N,K]
        Pi = tf.nn.softmax(logits_pi, axis=-1)                        # [N,K]
        if self.eta_type == 'symmetric':
          logits_eta = tf.matmul(X, self.W, transpose_b=True) - self.gamma  # [N,T]
          Eta = tf.nn.sigmoid(logits_eta)
          P = self._P_conditional(Eta)                             # [N,T,K,K]
        elif self.eta_type == 'full':
          #print(self.U.shape)
          S = tf.einsum('nd,tkld->ntkl', X, self.U) - self.Gamma[None,:,:,:]
          P = tf.nn.softmax(S, axis=-1)  # [N,T,K,K]
        elif self.eta_type == 'diagonal':
          #print(self.U.shape)
          S = tf.einsum('nd,tkd->ntk', X, self.U) - self.Gamma[None,:,:]
          Eta = tf.nn.sigmoid(S)  # [N,T,K]
          eye = tf.eye(self.K, batch_shape=[1,1])
          ones = tf.ones_like(eye)
          diag = Eta[:, :, :, None] * eye                  # [N,T,K,K]
          off = ((1.0 - Eta)[:, :, :, None] / tf.cast(self.K-1, tf.float32)) * (ones - eye)
          P = diag + off                                   # [N,T,K,K]
        else:
          raise ValueError(f"Unknown eta_type: {self.eta_type}")
        return Pi, P

    # ----- helper: build P(y|z,x) with symmetric errors -----
    def _P_conditional(self, Eta):
        # P[i,t,k,l] = η if l==k else (1-η)/(K-1)
        eye = tf.eye(self.K, batch_shape=[1,1])             # [1,1,K,K]
        ones = tf.ones_like(eye)
        diag = Eta[:, :, None, None] * eye                  # [N,T,K,K]
        off = ((1.0 - Eta)[:, :, None, None] / tf.cast(self.K-1, tf.float32)) * (ones - eye)
        return diag + off                                   # [N,T,K,K]

    # ==============
    # E-step: r = posterior over z (stop-gradient)
    # ==============
    def e_step(self, x_in, Y, M, batch_size=512):
        N = tf.shape(x_in)[0]
        r_list = []
        for s in range(0, N, batch_size):
            e = tf.minimum(s + batch_size, N)
            xb, Yb, Mb = x_in[s:e], Y[s:e], M[s:e]
            Pi, P = self._pi_and_eta(xb)                   # [B,K], [B,T,K,K]
            logP = tf.math.log(tf.clip_by_value(P, 1e-12, 1.0))
            # log p(y|z=k,x) = sum_t M * sum_l Y * log P
            log_py_given_k = tf.einsum('btl,btkl->btk', Yb, logP)         # [B,T,K]
            log_py_given_k = tf.reduce_sum(log_py_given_k * Mb[:, :, None], axis=1)  # [B,K]
            log_pi = tf.math.log(tf.clip_by_value(Pi, 1e-12, 1.0))        # [B,K]
            log_num = log_pi + log_py_given_k                              # [B,K]
            log_den = tf.reduce_logsumexp(log_num, axis=-1, keepdims=True) # [B,1]
            r = tf.exp(log_num - log_den)                                  # [B,K]
            r_list.append(tf.stop_gradient(r))
        return tf.concat(r_list, axis=0)                                    # [N,K]


    # ==============
    # M-step: Q-loss with fixed r (no gradient through r)
    # Q(θ) = E_r[ log p(y|z,x;θ) + log p(z|x;θ) ]
    # ==============
    def q_loss(self, x_in, Y, M, r):
        Pi, P = self._pi_and_eta(x_in)                   # [N,K], [N,T,K,K]
        logP = tf.math.log(tf.clip_by_value(P, 1e-12, 1.0))
        log_py_given_k = tf.einsum('ntl,ntkl->ntk', Y, logP)           # [N,T,K]
        log_py_given_k = tf.reduce_sum(log_py_given_k * M[:, :, None], axis=1)  # [N,K]
        log_pi = tf.math.log(tf.clip_by_value(Pi, 1e-12, 1.0))         # [N,K]
        # stop grad to be safe (should already be stopped outside)
        r_fixed = tf.stop_gradient(r)
        q = tf.reduce_sum(r_fixed * (log_py_given_k + log_pi), axis=-1)  # [N]
        return -tf.reduce_mean(q)  # minimize -Q


    # Prediction head: p(z|x)
    def predict_proba(self, x_in):
        Pi, P = self._pi_and_eta(x_in)
        return Pi, P

    def call(self, inputs, training=False):
        Pi, _ = self._pi_and_eta(inputs)
        return Pi


    def get_config(self):
        return {
            "d": self.d,
            "K": self.K,
            "T": self.T,
            "eta_type": self.eta_type,
            "backbone": tf.keras.utils.serialize_keras_object(self.backbone),
        }

    @classmethod
    def from_config(cls, config):
        backbone_config = config.pop("backbone")
        backbone = tf.keras.utils.deserialize_keras_object(backbone_config)
        return cls(backbone=backbone, **config)