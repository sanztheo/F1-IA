from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def forward_with_cache(params: Dict[str, np.ndarray], x: np.ndarray):
    hs = [x.astype(np.float32)]
    L = len(params) // 2
    for i in range(L):
        z = hs[-1] @ params[f"W{i}"] + params[f"b{i}"]
        if i < L - 1:
            h = np.tanh(z)
        else:
            h = z
        hs.append(h)
    # outputs
    o = hs[-1]
    steer = np.tanh(o[0])
    throttle = 1.0 / (1.0 + np.exp(-o[1]))
    brake = 1.0 / (1.0 + np.exp(-o[2]))
    y = np.array([steer, throttle, brake], dtype=np.float32)
    return y, hs


def backward(params: Dict[str, np.ndarray], hs, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
    grads: Dict[str, np.ndarray] = {}
    L = len(params) // 2
    # dLoss/dy (MSE)
    dy = 2.0 * (y_pred - y_true) / y_true.size
    # back through output squashing
    o = hs[-1]
    dsteer_do0 = (1.0 - np.tanh(o[0]) ** 2)
    dthr_do1 = (y_pred[1] * (1.0 - y_pred[1]))
    dbrk_do2 = (y_pred[2] * (1.0 - y_pred[2]))
    do = np.zeros_like(o)
    do[0] = dy[0] * dsteer_do0
    do[1] = dy[1] * dthr_do1
    do[2] = dy[2] * dbrk_do2

    delta = do
    for i in reversed(range(L)):
        h_prev = x.astype(np.float32) if i == 0 else hs[i]
        grads[f"W{i}"] = np.outer(h_prev, delta)
        grads[f"b{i}"] = delta.copy()
        if i > 0:
            dh_prev = params[f"W{i}"] @ delta
            dz_prev = (1.0 - hs[i] ** 2) * dh_prev
            delta = dz_prev
    return grads


def sgd_train(params: Dict[str, np.ndarray], sizes: Tuple[int, ...], X: np.ndarray, Y: np.ndarray,
              epochs: int = 5, batch: int = 256, lr: float = 1e-3) -> Dict[str, np.ndarray]:
    n = len(X)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        for off in range(0, n, batch):
            sl = idx[off:off + batch]
            if sl.size == 0:
                continue
            gsum = {k: np.zeros_like(v) for k, v in params.items()}
            for i in sl:
                y_pred, hs = forward_with_cache(params, X[i])
                grads = backward(params, hs, X[i], y_pred, Y[i])
                for k in params.keys():
                    gsum[k] += grads[k]
            for k in params.keys():
                params[k] -= lr * (gsum[k] / sl.size)
    return params

