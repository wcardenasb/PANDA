import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from panda.utils import simular_etiquetadores


DATASETS = {
    "Voice": {"type": "protocol", "protocol": ["YG", "YR", "YB"], "X": "X1", "Z": "y", "scale": True},
    "Polarity": {"type": "train_test"},
    "Music": {"type": "train_test"},
    "spirals": {"type": "annotations", "X": "X", "Z": "y", "Y": "Y", "scale": False},
    "blobs": {"type": "annotations", "X": "X", "Z": "y", "Y": "Y", "scale": False},
    "circles": {"type": "annotations", "X": "X", "Z": "y", "Y": "Y", "scale": False},

    "Breast": {"type": "simulated", "scale": True, "label_offset": -1},
    "Bupa": {"type": "simulated", "scale": True, "label_offset": -1},
    "Ionosphere": {"type": "simulated", "scale": True, "label_offset": -1},
    "Pima": {"type": "simulated", "scale": True, "label_offset": -1},
    "TicTacToe": {"type": "simulated", "scale": True, "label_offset": -1},
    "Occupancy": {"type": "simulated", "scale": True, "label_offset": -1, "test_size": 0.90},
    "Skin_NonSkin": {"type": "simulated", "scale": True, "label_offset": -1, "test_size": 0.95},

    "Iris": {"type": "simulated", "scale": True, "label_offset": 0},
    "Wine": {"type": "simulated", "scale": True, "label_offset": 0},
    "Segmentation": {"type": "simulated", "scale": True, "label_offset": 0},
    "Western": {"type": "simulated", "scale": True, "label_offset": 0},
}


def load_dataset(name, base_path, protocol=None, seed=42, p_errors=0.3):
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' no está configurado. Disponibles: {list(DATASETS.keys())}")

    config = DATASETS[name]
    data = loadmat(os.path.join(base_path, f"{name}.mat"))

    if config["type"] == "protocol": return _load_protocol(data, config, protocol, seed)
    if config["type"] == "annotations": return _load_annotations(data, config, seed)
    if config["type"] == "train_test": return _load_train_test(data, seed)
    if config["type"] == "simulated": return _load_simulated(data, config, name, seed, p_errors)

    raise ValueError(f"Tipo de dataset desconocido: {config['type']}")


def _load_protocol(data, config, protocol, seed):
    protocol = config["protocol"] if protocol is None else protocol
    results = []

    for grb in protocol:
        X, Z, Y = data[config["X"]], data[config["Z"]], data[grb]
        Y = (Y > 0).astype(int)
        M = np.ones((X.shape[0], Y.shape[1]))

        if config["scale"]:
            X = StandardScaler().fit_transform(X)

        K, T = len(np.unique(Y)), Y.shape[1]
        Y, Z = tf.one_hot(Y, depth=K).numpy(), tf.one_hot(Z.ravel(), depth=K).numpy()
        train_data, val_data = _split_data(X, Y, M, Z, test_size=0.3, seed=seed)

        results.append({"name": grb, "train_data": train_data, "val_data": val_data, "input_dim": X.shape[1], "K": K, "T": T})

    return results


def _load_annotations(data, config, seed):
    X, Z, Y = data[config["X"]], data[config["Z"]].ravel(), data[config["Y"]]
    M = np.ones_like(Y)

    if config["scale"]:
        X = StandardScaler().fit_transform(X)

    K, T = len(np.unique(Z)), Y.shape[1]
    Z, Y = tf.one_hot(Z, depth=K).numpy(), tf.one_hot(Y, depth=K).numpy()
    train_data, val_data = _split_data(X, Y, M, Z, test_size=0.3, seed=seed)

    return [{"name": None, "train_data": train_data, "val_data": val_data, "input_dim": X.shape[1], "K": K, "T": T}]


def _load_train_test(data, seed):
    x_train, Z_train, Y_train = data["Xtrain"], data["ytrain"].ravel(), data["Ytrain"]

    Y_train[Y_train == -1.0] = 0.0
    M_train = (Y_train != -1e20).astype(int)

    x_val, Z_val = data["Xtest"], data["ytest"].ravel()
    Y_val = np.ones((x_val.shape[0], Y_train.shape[1]))
    M_val = np.ones((x_val.shape[0], Y_train.shape[1]))

    Z_val[Z_val == -1.0] = 0.0

    T, K = Y_train.shape[1], len(np.unique(Y_train)) - 1
    Y_train = tf.one_hot(Y_train, depth=K).numpy()
    Z_train = tf.one_hot(Z_train, depth=K).numpy()
    Z_val = tf.one_hot(Z_val, depth=K).numpy()

    return [{
        "name": None,
        "train_data": _to_tensors(x_train, Y_train, M_train, Z_train),
        "val_data": _to_tensors(x_val, Y_val, M_val, Z_val),
        "input_dim": x_train.shape[1],
        "K": K,
        "T": T
    }]


def _load_simulated(data, config, name, seed, p_errors):
    X, y = data["X"], data["y"].ravel() + config["label_offset"]

    if config["scale"]:
        X = StandardScaler().fit_transform(X)

    K, T = len(np.unique(y)), 5
    Y, M, Z = simular_etiquetadores(X, y, T=T, cluster="gmm", seed=seed, p_errors=p_errors)

    train_data, val_data = _split_data(X, Y, M, Z, test_size=config.get("test_size", 0.3), seed=seed)

    return [{"name": name, "train_data": train_data, "val_data": val_data, "input_dim": X.shape[1], "K": K, "T": T}]


def _split_data(X, Y, M, Z, test_size, seed):
    x_train, x_val, Y_train, Y_val, M_train, M_val, Z_train, Z_val = train_test_split(
        X, Y, M, Z, test_size=test_size, random_state=seed
    )
    return _to_tensors(x_train, Y_train, M_train, Z_train), _to_tensors(x_val, Y_val, M_val, Z_val)


def _to_tensors(X, Y, M, Z):
    return tuple(tf.constant(v, dtype=tf.float32) for v in (X, Y, M, Z))