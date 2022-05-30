import typing

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811


class NSNetwork(keras.Sequential):
    def __init__(self):
        super().__init__(
            [
                keras.layers.InputLayer(3),
                *(
                    keras.layers.Dense(20, activation=keras.activations.tanh)
                    for _ in range(8)
                ),
                keras.layers.Dense(2),
            ],
            "NSNetwork",
        )


def loss(
    network: NSNetwork,
    x: tf.Variable,  # Nx3
    lambda1: tf.Variable,
    lambda2: tf.Variable,
    u_measured: tf.Tensor,
    v_measured: tf.Tensor,
) -> float:

    x = x[:, 0]
    y = x[:, 1]
    t = x[:, 2]

    with tf.GradientTape() as third:
        with tf.GradientTape() as second:
            with tf.GradientTape() as first:
                psi, p = network([x, y, t])

            u = first.gradient(psi, y)
            v = -first.gradient(psi, x)

        u_t = second.gradient(u, t)
        u_x = second.gradient(u, x)
        u_y = second.gradient(u, y)

        v_t = second.gradient(v, t)
        v_x = second.gradient(v, x)
        v_y = second.gradient(v, y)

    u_xx = third.gradient(u_x, x)
    u_yy = third.gradient(u_y, y)

    v_xx = third.gradient(v_x, x)
    v_yy = third.gradient(v_y, y)

    p_x = third.gradient(p, x)
    p_y = third.gradient(p, y)

    f_u = u_t + lambda1 * (u * u_x + v * u_y) + p_x - lambda2 * (u_xx + u_yy)
    f_v = v_t + lambda1 * (u * v_x + v * v_y) + p_y - lambda2 * (v_xx + v_yy)

    return (
        tf.reduce_sum(tf.square(u_measured - u))
        + tf.reduce_sum(tf.square(v_measured - v))
        + tf.reduce_sum(tf.square(f_u))
        + tf.reduce_sum(tf.square(f_v))
    )
