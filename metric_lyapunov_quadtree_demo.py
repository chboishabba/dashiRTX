#!/usr/bin/env python3
"""metric_lyapunov_quadtree_demo.py

A minimal, runnable reference implementation of:

1) Metric-aware Lyapunov descent training loop for a refresh field u ∈ [-1,1]
   (relaxed ternary control; probability m = (1+u)/2).

2) Minimal quadtree prototype for O(N_active) "render" accumulation:
   - build + refine based on a frontier/error criterion
   - accumulate node contributions instead of per-pixel loops

This is intentionally self-contained (numpy only; matplotlib optional for plots).

Run:
  python metric_lyapunov_quadtree_demo.py --show
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Dict

import numpy as np


# -----------------------------
# Utilities
# -----------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def safe_bern_entropy(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """H(m) = - m log2 m - (1-m) log2(1-m)"""
    m = np.clip(m, eps, 1.0 - eps)
    return -(m * np.log2(m) + (1.0 - m) * np.log2(1.0 - m))

def d_safe_bern_entropy_dm(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """d/dm H(m) = log2((1-m)/m)"""
    m = np.clip(m, eps, 1.0 - eps)
    return np.log2((1.0 - m) / m)

def tv_l2_and_grad(U: np.ndarray, eta_tv: float) -> Tuple[float, np.ndarray]:
    """Smooth L2-TV: eta * sum((dx)^2 + (dy)^2)."""
    if eta_tv <= 0:
        return 0.0, np.zeros_like(U)

    dx = np.zeros_like(U)
    dy = np.zeros_like(U)
    dx[:, :-1] = U[:, 1:] - U[:, :-1]
    dy[:-1, :] = U[1:, :] - U[:-1, :]

    tv = eta_tv * float(np.sum(dx * dx + dy * dy))

    div = np.zeros_like(U)
    div[:, :-1] -= dx[:, :-1]
    div[:, 1:]  += dx[:, :-1]
    div[:-1, :] -= dy[:-1, :]
    div[1:,  :] += dy[:-1, :]

    grad = 2.0 * eta_tv * div
    return tv, grad

def area_superlinear_penalty(m: np.ndarray, gamma_area: float, p: float = 1.15) -> Tuple[float, np.ndarray]:
    """gamma * (sum m)^p and its gradient."""
    if gamma_area <= 0:
        return 0.0, np.zeros_like(m)
    a = float(np.sum(m))
    val = gamma_area * (a ** p)
    da = gamma_area * p * (a ** (p - 1.0)) if a > 0 else 0.0
    grad = np.full_like(m, da)
    return val, grad


# -----------------------------
# Synth scene + "transport" signals (toy)
# -----------------------------

def synth_signals(H: int, W: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = 0.55 * W, 0.45 * H
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.45 * min(H, W))

    obj = r < 0.55
    band = np.abs(r - 0.55) < 0.05

    S = np.zeros((H, W), dtype=np.int8)
    S[(r >= 0.55) & band] = +1
    S[(r < 0.55) & band] = -1

    e = 0.15 * band.astype(np.float32) + 0.03 * rng.random((H, W), dtype=np.float32)
    e += 0.08 * (~obj).astype(np.float32) * (0.2 + 0.8 * rng.random((H, W), dtype=np.float32))

    hx, hy = 0.25 * W, 0.65 * H
    hot = np.exp(-(((xx - hx) / (0.08 * W)) ** 2 + ((yy - hy) / (0.08 * H)) ** 2)).astype(np.float32)

    I = (0.6 * band.astype(np.float32) + 1.2 * hot).astype(np.float32)
    I = I / (I.max() + 1e-9)

    h = (0.5 * I + 0.25 * rng.random((H, W), dtype=np.float32)).astype(np.float32)
    h = h / (h.max() + 1e-9)

    return {"e": e, "S": S, "I": I, "h": h, "obj": obj.astype(np.float32), "band": band.astype(np.float32), "hot": hot}


# -----------------------------
# Metric-aware Lyapunov loss + gradient (relaxed ternary control)
# -----------------------------

@dataclass
class LossTerms:
    rate: float
    comp: float
    dist: float
    tv: float
    metric_reg: float
    contract_reg: float
    total: float

@dataclass
class TrainConfig:
    lam: float = 6.0
    beta: float = 1.0
    c_pos: float = 0.06
    c_neg: float = 0.20
    eta_tv: float = 0.03
    gamma_area: float = 5e-4
    p_area: float = 1.15
    alpha: float = 12.0
    kappa: float = 6.0
    gamma_metric: float = 0.03
    mu_contract: float = 0.0
    kappa_target: float = 0.97
    eps: float = 1e-6

def compute_metric(u: np.ndarray, I: np.ndarray, S: np.ndarray) -> np.ndarray:
    H, W = u.shape
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        -4.0 * u[1:-1, 1:-1]
        + u[1:-1, :-2] + u[1:-1, 2:]
        + u[:-2, 1:-1] + u[2:, 1:-1]
    )
    curv = np.abs(lap)
    frontier = (np.abs(S) > 0).astype(np.float32)
    beta_I, beta_S, beta_curv = 3.0, 1.5, 0.75
    G = 1.0 + beta_I * I + beta_S * frontier + beta_curv * curv
    return G.astype(np.float32)

def metric_smoothness(G: np.ndarray, gamma_metric: float) -> float:
    if gamma_metric <= 0:
        return 0.0
    dx = np.zeros_like(G)
    dy = np.zeros_like(G)
    dx[:, :-1] = G[:, 1:] - G[:, :-1]
    dy[:-1, :] = G[1:, :] - G[:-1, :]
    return gamma_metric * float(np.sum(dx*dx + dy*dy))

def loss_and_grad_u(u: np.ndarray, e: np.ndarray, S: np.ndarray, I: np.ndarray, h: np.ndarray, cfg: TrainConfig) -> Tuple[LossTerms, np.ndarray, np.ndarray]:
    m = clamp01((1.0 + u) * 0.5)

    w = (1.0 + cfg.alpha * I) * (1.0 + cfg.kappa * h)
    e_eff = e * w

    Hm = safe_bern_entropy(m, cfg.eps)
    rate = cfg.beta * float(np.sum(Hm))
    d_rate_dm = cfg.beta * d_safe_bern_entropy_dm(m, cfg.eps)

    supp = (np.abs(S) > 0)
    c_map = np.where(supp, cfg.c_pos, cfg.c_neg).astype(np.float32)

    comp = float(np.sum(c_map * m))
    d_comp_dm = c_map

    area_val, d_area_dm = area_superlinear_penalty(m, cfg.gamma_area, cfg.p_area)
    comp += area_val
    d_comp_dm = d_comp_dm + d_area_dm

    dist = cfg.lam * float(np.sum((1.0 - m) * e_eff))
    d_dist_dm = -cfg.lam * e_eff

    tv, d_tv_du = tv_l2_and_grad(u, cfg.eta_tv)

    G = compute_metric(u, I, S)
    mreg = metric_smoothness(G, cfg.gamma_metric)

    contract = 0.0
    d_contract_du = np.zeros_like(u)

    dm_du = 0.5
    dL_dm = d_rate_dm + d_comp_dm + d_dist_dm
    dL_du = dL_dm * dm_du + d_tv_du + d_contract_du

    total = rate + comp + dist + tv + mreg + contract
    terms = LossTerms(rate=rate, comp=comp, dist=dist, tv=tv, metric_reg=mreg, contract_reg=contract, total=total)
    return terms, dL_du.astype(np.float32), G

def lyapunov_metric_train(
    u0: np.ndarray,
    e: np.ndarray, S: np.ndarray, I: np.ndarray, h: np.ndarray,
    cfg: TrainConfig,
    steps: int = 200,
    lr: float = 0.4,
    armijo: float = 1e-4,
    backtrack: int = 20,
    seed: int = 0,
    verbose_every: int = 20,
) -> Tuple[np.ndarray, List[LossTerms]]:
    u = u0.astype(np.float32).copy()
    log: List[LossTerms] = []

    for t in range(steps):
        terms, grad, G = loss_and_grad_u(u, e, S, I, h, cfg)
        log.append(terms)

        G_inv = 1.0 / (G + 1e-6)
        step_dir = G_inv * grad
        descent_measure = float(np.sum(grad * step_dir))

        lr_t = lr
        accepted = False
        for _ in range(backtrack):
            u_new = np.clip(u - lr_t * step_dir, -1.0, 1.0)
            new_terms, _, _ = loss_and_grad_u(u_new, e, S, I, h, cfg)
            if new_terms.total <= terms.total - armijo * lr_t * descent_measure:
                u = u_new
                accepted = True
                break
            lr_t *= 0.5

        if not accepted:
            break

        if verbose_every and (t % verbose_every == 0 or t == steps - 1):
            print(f"[train] step {t:4d}  L={terms.total:10.4f}  rate={terms.rate:9.2f} comp={terms.comp:9.2f} dist={terms.dist:9.2f}")

    # record final
    terms, _, _ = loss_and_grad_u(u, e, S, I, h, cfg)
    log.append(terms)
    return u, log


# -----------------------------
# Minimal Quadtree (ultrametric) prototype
# -----------------------------

@dataclass
class QNode:
    x0: int
    y0: int
    size: int
    depth: int
    children: Optional[List["QNode"]] = None
    mean_e: float = 0.0
    mean_frontier: float = 0.0
    mean_I: float = 0.0
    mean_h: float = 0.0

def node_stats(node: QNode, e: np.ndarray, S: np.ndarray, I: np.ndarray, h: np.ndarray) -> None:
    x0, y0, sz = node.x0, node.y0, node.size
    patch_e = e[y0:y0+sz, x0:x0+sz]
    patch_S = S[y0:y0+sz, x0:x0+sz]
    patch_I = I[y0:y0+sz, x0:x0+sz]
    patch_h = h[y0:y0+sz, x0:x0+sz]
    node.mean_e = float(np.mean(patch_e))
    node.mean_frontier = float(np.mean((np.abs(patch_S) > 0).astype(np.float32)))
    node.mean_I = float(np.mean(patch_I))
    node.mean_h = float(np.mean(patch_h))

def subdivide(node: QNode) -> List[QNode]:
    sz = node.size
    assert sz % 2 == 0
    hsz = sz // 2
    x0, y0, d = node.x0, node.y0, node.depth + 1
    return [
        QNode(x0,      y0,      hsz, d),
        QNode(x0+hsz,  y0,      hsz, d),
        QNode(x0,      y0+hsz,  hsz, d),
        QNode(x0+hsz,  y0+hsz,  hsz, d),
    ]

def gather_leaves(node: QNode) -> List[QNode]:
    if node.children is None:
        return [node]
    out: List[QNode] = []
    for c in node.children:
        out.extend(gather_leaves(c))
    return out

def build_quadtree(
    e: np.ndarray, S: np.ndarray, I: np.ndarray, h: np.ndarray,
    m: np.ndarray,
    min_size: int = 4
) -> QNode:
    H, W = e.shape
    assert H == W, "prototype assumes square"
    root = QNode(0, 0, H, 0)
    node_stats(root, e, S, I, h)

    def refine(n: QNode) -> bool:
        if n.size <= min_size:
            return False
        x0, y0, sz = n.x0, n.y0, n.size
        mean_m = float(np.mean(m[y0:y0+sz, x0:x0+sz]))
        return (mean_m > 0.25) or (n.mean_frontier > 0.03) or (n.mean_I > 0.35) or (n.mean_e > 0.12)

    def rec(n: QNode):
        if refine(n):
            n.children = subdivide(n)
            for c in n.children:
                node_stats(c, e, S, I, h)
                rec(c)

    rec(root)
    return root

def quadtree_accumulate(
    root: QNode,
    texture: np.ndarray,
    view_param: float
) -> np.ndarray:
    H, W = texture.shape
    out = np.zeros_like(texture, dtype=np.float32)

    def kernel(n: QNode, v: float) -> float:
        return float(0.2 + 1.8 * (0.6 * n.mean_I + 0.4 * n.mean_frontier) * (0.5 + 0.5 * math.sin(v)))

    for n in gather_leaves(root):
        x0, y0, sz = n.x0, n.y0, n.size
        avg = float(np.mean(texture[y0:y0+sz, x0:x0+sz]))
        out[y0:y0+sz, x0:x0+sz] += kernel(n, view_param) * avg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--lr", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    H = W = args.N
    sig = synth_signals(H, W, seed=args.seed)
    e, S, I, h = sig["e"], sig["S"], sig["I"], sig["h"]

    rng = np.random.default_rng(args.seed)
    u0 = np.clip(0.05 * rng.standard_normal((H, W)).astype(np.float32), -1.0, 1.0)

    cfg = TrainConfig()
    u, log = lyapunov_metric_train(u0, e, S, I, h, cfg, steps=args.steps, lr=args.lr, seed=args.seed)

    m = clamp01((1.0 + u) * 0.5)
    uT = np.zeros_like(S)
    uT[u > 0.33] = +1
    uT[u < -0.33] = -1

    root = build_quadtree(e, S, I, h, m, min_size=4)
    leaves = gather_leaves(root)
    print(f"[quadtree] leaves={len(leaves)} vs pixels={H*W}")

    texture = (0.35 * sig["obj"] + 1.2 * sig["hot"]).astype(np.float32)
    out_qt = quadtree_accumulate(root, texture, view_param=1.3)

    print(f"[final] m mean={m.mean():.3f} frac(m>0.5)={(m>0.5).mean():.3f}")
    print(f"[final] uT frac +1={(uT==1).mean():.3f} 0={(uT==0).mean():.3f} -1={(uT==-1).mean():.3f}")
    print(f"[loss] start={log[0].total:.4f} end={log[-1].total:.4f} logged={len(log)}")

    if args.show:
        try:
            import matplotlib.pyplot as plt
            plt.figure(); plt.plot([t.total for t in log]); plt.title("Lyapunov loss"); plt.xlabel("step"); plt.ylabel("L")
            plt.figure(); plt.imshow(e); plt.title("e (post-warp error)"); plt.colorbar()
            plt.figure(); plt.imshow(S, vmin=-1, vmax=1); plt.title("S (signed frontier)"); plt.colorbar()
            plt.figure(); plt.imshow(I, vmin=0, vmax=1); plt.title("I (importance)"); plt.colorbar()
            plt.figure(); plt.imshow(m, vmin=0, vmax=1); plt.title("m=(1+u)/2"); plt.colorbar()
            plt.figure(); plt.imshow(uT, vmin=-1, vmax=1); plt.title("uT quantized"); plt.colorbar()
            plt.figure(); plt.imshow(out_qt); plt.title("quadtree accumulate"); plt.colorbar()
            plt.show()
        except Exception as ex:
            print("[warn] matplotlib failed:", ex)

if __name__ == "__main__":
    main()
