from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io
from numpy.typing import NDArray

from nneve.navier_stokes.network import NSNetwork

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"


def load_data_from(file: Path) -> Tuple[NDArray[np.float32], ...]:
    data = scipy.io.loadmat(file)

    u_star = data["U_star"]  # N x 2 x T
    p_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    x_star = data["X_star"]  # N x 2

    n_vals = x_star.shape[0]
    t_vals = t_star.shape[0]

    # Rearrange Data
    xx_vals = np.tile(x_star[:, 0:1], (1, t_vals))  # N x T
    yy_vals = np.tile(x_star[:, 1:2], (1, t_vals))  # N x T
    tt_vals = np.tile(t_star, (1, n_vals)).T  # N x T

    uu_vals = u_star[:, 0, :]  # N x T
    vv_vals = u_star[:, 1, :]  # N x T
    pp_vals = p_star  # N x T

    x = xx_vals.flatten()[:, None]  # NT x 1
    y = yy_vals.flatten()[:, None]  # NT x 1
    t = tt_vals.flatten()[:, None]  # NT x 1

    u = uu_vals.flatten()[:, None]  # NT x 1
    v = vv_vals.flatten()[:, None]  # NT x 1
    p = pp_vals.flatten()[:, None]  # NT x 1

    return x, y, t, u, v, p


def test_network_learning():
    # ---------------------------------- prepare --------------------------------- #
    x, y, t, u, v, p = load_data_from(DATA_DIR / "cylinder_nektar_wake.mat")
    idx = np.random.choice(x.shape[0], 5000, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]
    net = NSNetwork()
    # ---------------------------------- execute --------------------------------- #
    net.train(
        np.concatenate((x_train, y_train, t_train), axis=1),  # type: ignore
        u_train,
        v_train,
    )
