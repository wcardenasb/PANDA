import tensorflow as tf
from tensorflow.keras import layers, Model


def build_crowd_nn(input_dim, K, T, units_list=(100, 90), lr=1e-3):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(units_list[0], activation="relu")(inputs)
    x = layers.Dense(units_list[1], activation="relu")(x)
    feature_layer = layers.Dense(K, activation="softmax", name="feature_layer")(x)
    outputs = [layers.Dense(K, activation="softmax", name=f"annotator_{t}")(feature_layer) for t in range(T)]
    model = Model(inputs=inputs, outputs=outputs)
    losses = {f"annotator_{t}": "categorical_crossentropy" for t in range(T)}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=losses)
    return model