import tensorflow as tf

from nneve.benchmark.testing import disable_gpu_or_skip
from nneve.quantum_oscilator.network import QOConstants, QONetwork
from nneve.quantum_oscilator.tracker import QOTracker


def test_validate_output():
    tf.random.set_seed(0)
    disable_gpu_or_skip()

    constants = QOConstants(
        k=4.0,
        mass=1.0,
        x_left=-6.0,
        x_right=6.0,
        fb=0.0,
        sample_size=16,
        tracker=QOTracker(),
    )
    nn = QONetwork(constants=constants, is_debug=True)
    x = constants.sample()
    deriv_x = tf.Variable(initial_value=x)

    eigenvalue = 1.0

    assert x.shape == (16, 1)
    assert x.dtype == tf.float32
    assert deriv_x.shape == (16, 1)
    assert deriv_x.dtype == tf.float32

    residuum, y_values = nn._get_residuum_function(deriv_x, x, eigenvalue)  # type: ignore

    assert residuum.shape == ()
    assert residuum.dtype == tf.float32
    assert abs(float(residuum) - 257.0) < 1.0
    assert y_values.shape == (16, 1)
    assert y_values.dtype == tf.float32
