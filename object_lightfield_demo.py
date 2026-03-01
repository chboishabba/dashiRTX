#!/usr/bin/env python3
"""
object_lightfield_demo.py

Prototype: object/lightfield split.
- Object boundary represented by a ring field (implicit contour).
- Lightfield/transport represented by a simple kernel/operator K(x,y).

This is a minimal demo to make the conceptual split explicit.
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt


def synth_ring(N, center, r0=0.55, band=0.05, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N]
    cx, cy = center
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.45 * N)
    band_mask = np.abs(r - r0) < band

    S = np.zeros((N, N), dtype=np.int8)
    S[(r >= r0) & band_mask] = +1
    S[(r < r0) & band_mask] = -1

    texture = 0.4 * band_mask.astype(np.float32) + 0.6 * rng.random((N, N), dtype=np.float32)
    texture = texture / (texture.max() + 1e-9)

    I = 0.4 * band_mask.astype(np.float32) + 0.6 * rng.random((N, N), dtype=np.float32)
    I = I / (I.max() + 1e-9)

    return S, I, texture, r


def ring_field(N, center, r0=0.55):
    yy, xx = np.mgrid[0:N, 0:N]
    cx, cy = center
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.45 * N)
    phi = r - r0
    return phi


def estimate_ring_params(S):
    yy, xx = np.mgrid[0:S.shape[0], 0:S.shape[1]]
    mask = (np.abs(S) > 0)
    if np.sum(mask) == 0:
        return (S.shape[1] / 2.0, S.shape[0] / 2.0, 0.55)
    cx = float(np.mean(xx[mask]))
    cy = float(np.mean(yy[mask]))
    r = np.sqrt((xx[mask] - cx) ** 2 + (yy[mask] - cy) ** 2)
    r0 = float(np.median(r) / (0.45 * S.shape[0]))
    return (cx, cy, r0)


def fit_operator_K(S, I, phi, texture, target, lr=0.5, steps=200):
    # Simple logistic kernel: K = sigmoid(w0 + w1*I + w2*frontier + w3*|phi|)
    frontier = (np.abs(S) > 0).astype(np.float32)
    x0 = np.ones_like(I)
    x1 = I
    x2 = frontier
    x3 = np.abs(phi)

    X = np.stack([x0, x1, x2, x3], axis=-1).reshape(-1, 4)
    y = target.reshape(-1)
    tex = texture.reshape(-1)

    w = np.array([0.0, 0.5, 0.5, -0.5], dtype=np.float32)

    for _ in range(steps):
        logits = X @ w
        K = 1.0 / (1.0 + np.exp(-logits))
        pred = K * tex
        err = pred - y
        dK = err * tex
        dlog = dK * (K * (1.0 - K))
        grad = (X.T @ dlog) / X.shape[0]
        w -= lr * grad.astype(np.float32)

    logits = X @ w
    K = 1.0 / (1.0 + np.exp(-logits))
    K = K.reshape(S.shape)
    return w, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    center = (0.52 * args.N, 0.48 * args.N)
    S, I, texture, r = synth_ring(args.N, center, seed=args.seed)

    # "Object": estimate ring params from S and build ring field phi
    cx, cy, r0 = estimate_ring_params(S)
    phi = ring_field(args.N, (cx, cy), r0=r0)

    # "Lightfield" target: simple target render
    frontier = (np.abs(S) > 0).astype(np.float32)
    target = (0.25 + 1.5 * (0.6 * I + 0.4 * frontier)) * texture

    w, K = fit_operator_K(S, I, phi, texture, target)
    pred = K * texture
    mae = float(np.mean(np.abs(pred - target)))

    print(f"[ring] cx={cx:.2f} cy={cy:.2f} r0={r0:.3f}")
    print(f"[operator] w={w.tolist()}  mae={mae:.4f}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.figure(); plt.imshow(S, vmin=-1, vmax=1); plt.title("S (frontier)"); plt.colorbar()
        plt.figure(); plt.imshow(phi); plt.title("phi (ring field)"); plt.colorbar()
        plt.figure(); plt.imshow(K); plt.title("K (operator)"); plt.colorbar()
        plt.figure(); plt.imshow(target); plt.title("target render"); plt.colorbar()
        plt.figure(); plt.imshow(pred); plt.title("pred render"); plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
