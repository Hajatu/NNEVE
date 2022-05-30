import typing
from typing import cast

import tensorflow as tf
from rich.progress import Progress, TaskID
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811


__all__ = ["NSNetwork"]


class NSNetwork(keras.Sequential):

    lambda1: tf.Variable
    lambda2: tf.Variable
    optimizer: keras.optimizers.Optimizer

    def __init__(self):
        super().__init__(
            [
                keras.layers.InputLayer(3),
                *[
                    keras.layers.Dense(20, activation=keras.activations.tanh)
                    for _ in range(8)
                ],
                keras.layers.Dense(2),
            ],
            "NSNetwork",
        )
        self.lambda1 = tf.Variable(0.0)
        self.lambda2 = tf.Variable(0.0)
        self.optimizer = keras.optimizers.Adam()

    def train(
        self,
        x: tf.Tensor,
        u_measured: tf.Tensor,
        v_measured: tf.Tensor,
        epochs: int = 1000,
    ):
        x_min = tf.math.reduce_min(x)
        x_max = tf.math.reduce_max(x)
        x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

        with Progress() as progress:
            task = progress.add_task(
                description="Learning...", total=epochs + 1
            )
            for index in range(epochs):
                deriv_x = tf.Variable(initial_value=x)
                with tf.GradientTape() as tape:
                    loss_value = loss(
                        self,
                        deriv_x,
                        u_measured,
                        v_measured,
                    )

                trainable_vars = self.trainable_variables
                # ; print([f"{v.shape} -> {float(tf.size(v))}" for v in trainable_vars])
                gradients = tape.gradient(loss_value, trainable_vars)
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                self.callback(index, cast(float, loss_value), progress, task)

    def callback(
        self, index: int, loss: float, progress: Progress, task: TaskID
    ) -> None:
        message = (
            f"Epoch: {index:<7} Loss: {loss:<10.3f} "
            f"Lambda 1: {float(self.lambda1):<10.3f} "
            f"Lambda 2: {float(self.lambda2):<10.3f}"
        )
        progress.update(task, advance=1, description=message)


def loss(
    network: NSNetwork,
    xyt: tf.Variable,  # Nx3
    u_measured: tf.Tensor,
    v_measured: tf.Tensor,
) -> float:

    x = xyt[0]
    y = xyt[1]
    t = xyt[2]

    with tf.GradientTape(True) as third:
        with tf.GradientTape(True) as second:
            with tf.GradientTape(True) as first:
                retval = network(xyt)
                psi = retval[0]
                p = retval[1]

            u = first.gradient(psi, xyt[0])
            v = -first.gradient(psi, xyt[1])

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

    f_u = (
        u_t
        + network.lambda1 * (u * u_x + v * u_y)
        + p_x
        - network.lambda2 * (u_xx + u_yy)
    )
    f_v = (
        v_t
        + network.lambda1 * (u * v_x + v * v_y)
        + p_y
        - network.lambda2 * (v_xx + v_yy)
    )

    return (
        tf.reduce_sum(tf.square(u_measured - u))
        + tf.reduce_sum(tf.square(v_measured - v))
        + tf.reduce_sum(tf.square(f_u))
        + tf.reduce_sum(tf.square(f_v))
    )
