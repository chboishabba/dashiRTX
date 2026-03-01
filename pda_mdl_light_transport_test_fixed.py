#!/usr/bin/env python3
"""
pda_mdl_light_transport_test.py

Self-contained PDA–MDL toy for learning an invariant "refresh mask" for reprojection errors.

What it does
------------
1) Build a tiny synthetic 3D scene (two spheres: big gray + small bright red).
2) Render depth + radiance from camera t0 and camera t1 (pinhole rays).
3) Reproject radiance t0 -> t1 using depth0 and full pinhole pose transform (with z-buffer occlusion test).
4) Compute:
   - error map e = |rad1_true - rad1_reproj|
   - signed frontier S0 in {-1,0,+1} (a ternary carrier) using depth edges + motion direction
   - importance I0 (precomputed leverage proxy)
   - persistent state h (Step B memory)
5) Train two learners to predict a refresh probability mask m(p):
   - Logistic (linear -> sigmoid)
   - Tiny 2-layer MLP (per-pixel, conv-like spirit)
   using the MDL-style loss with Step A+B:
     e_eff = e * (1 + alpha*I0) * (1 + kappa*h)
6) Plot:
   - Depth0, Signed frontier S0, Importance I0, Occlusion/transport error
   - Reprojected vs true radiance
   - Learned refresh prob and chosen binary mask
   - MDL' vs threshold on heldout

Run
---
python pda_mdl_light_transport_test.py --outdir outputs_pda --show

Dependencies: numpy, matplotlib (no torch).
"""

import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------
# Camera + rendering (pinhole)
# -------------------------------

@dataclass
class Camera:
    """Pinhole camera in world coordinates."""
    pos: np.ndarray        # (3,)
    R: np.ndarray          # (3,3) world_from_cam rotation
    f: float               # focal length in pixels
    cx: float
    cy: float

    def cam_from_world(self, Xw: np.ndarray) -> np.ndarray:
        # Xc = R^T (Xw - pos) because R is world_from_cam
        return (self.R.T @ (Xw - self.pos).T).T

    def world_from_cam(self, Xc: np.ndarray) -> np.ndarray:
        return (self.R @ Xc.T).T + self.pos

    def project(self, Xw: np.ndarray):
        Xc = self.cam_from_world(Xw)
        z = Xc[:, 2]
        z_safe = np.where(z == 0, 1e-9, z)
        u = self.f * (Xc[:, 0] / z_safe) + self.cx
        v = self.f * (Xc[:, 1] / z_safe) + self.cy
        return u, v, z

    def unproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = (u - self.cx) * z / self.f
        y = (v - self.cy) * z / self.f
        Xc = np.stack([x, y, z], axis=-1)
        return self.world_from_cam(Xc)


@dataclass
class Sphere:
    center: np.ndarray  # (3,)
    radius: float
    color: np.ndarray   # (3,) RGB in [0,1]
    emissive: float = 0.0  # add emissive term


def ray_sphere_intersect(ro, rd, sphere: Sphere):
    """
    Ray origin ro (3,), direction rd (...,3) normalized
    Returns t_hit (...,) with np.inf if no hit.
    """
    oc = ro - sphere.center
    b = 2.0 * np.einsum("...i,i->...", rd, oc)
    c = np.dot(oc, oc) - sphere.radius * sphere.radius
    disc = b * b - 4.0 * c
    hit = disc >= 0.0
    t = np.full(rd.shape[:-1], np.inf, dtype=np.float32)
    if np.any(hit):
        sdisc = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - sdisc) / 2.0
        t1 = (-b + sdisc) / 2.0
        t_candidate = np.where(t0 > 1e-4, t0, np.where(t1 > 1e-4, t1, np.inf))
        t = np.where(hit, t_candidate, np.inf).astype(np.float32)
    return t


def render_scene(cam: Camera, H: int, W: int, spheres):
    """
    Render depth (camera z) and radiance (RGB) with simple lambert + emissive.
    Background is black.
    """
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))  # u=j, v=i
    u = jj.astype(np.float32)
    v = ii.astype(np.float32)

    # rays in camera coords
    x = (u - cam.cx) / cam.f
    y = (v - cam.cy) / cam.f
    rd_c = np.stack([x, y, np.ones_like(x)], axis=-1)
    rd_c = rd_c / (np.linalg.norm(rd_c, axis=-1, keepdims=True) + 1e-9)

    # transform to world
    rd_w = (cam.R @ rd_c.reshape(-1, 3).T).T.reshape(H, W, 3)
    ro_w = cam.pos.astype(np.float32)

    # intersect all spheres, take nearest
    t_min = np.full((H, W), np.inf, dtype=np.float32)
    hit_id = np.full((H, W), -1, dtype=np.int32)

    for si, s in enumerate(spheres):
        t = ray_sphere_intersect(ro_w, rd_w, s)
        better = t < t_min
        t_min = np.where(better, t, t_min)
        hit_id = np.where(better, si, hit_id)

    depth = t_min.copy()
    depth[np.isinf(depth)] = 0.0  # 0 means no hit

    # shade
    rad = np.zeros((H, W, 3), dtype=np.float32)
    valid = hit_id >= 0
    if np.any(valid):
        Pw = ro_w + rd_w * depth[..., None]
        N = np.zeros_like(Pw)
        for si, s in enumerate(spheres):
            mask = hit_id == si
            if np.any(mask):
                n = Pw[mask] - s.center
                n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
                N[mask] = n

        L = np.array([0.2, 0.6, 1.0], dtype=np.float32)
        L = L / (np.linalg.norm(L) + 1e-9)
        ndotl = np.clip(np.einsum("...i,i->...", N, L), 0.0, 1.0)

        for si, s in enumerate(spheres):
            mask = hit_id == si
            if np.any(mask):
                base = s.color[None, None, :]
                rad[mask] = (0.15 + 0.85 * ndotl[mask])[..., None] * base + s.emissive * base

    luma = (0.2126 * rad[..., 0] + 0.7152 * rad[..., 1] + 0.0722 * rad[..., 2]).astype(np.float32)
    return depth.astype(np.float32), luma.astype(np.float32), rad.astype(np.float32)


def reproject_luma(depth0, luma0, cam0: Camera, cam1: Camera, depth1):
    """
    Reproject luma0 from cam0 to cam1 using depth0 and full pinhole model.
    Includes a z-buffer occlusion test against depth1 (disocclusion shows as error).
    Returns:
      luma_warp: (H,W) warped luma0 into cam1 frame (0 where missing/occluded)
      valid_warp: (H,W) bool where warp wrote
      flow: (H,W,2) pixel flow (u1-u0, v1-v0) for valid depth0
    """
    H, W = depth0.shape
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    u0 = jj.astype(np.float32)
    v0 = ii.astype(np.float32)
    z0 = depth0

    mask0 = z0 > 0
    u0v = u0[mask0]
    v0v = v0[mask0]
    z0v = z0[mask0]
    if u0v.size == 0:
        return np.zeros_like(luma0), np.zeros_like(mask0), np.zeros((H, W, 2), dtype=np.float32)

    Pw = cam0.unproject(u0v, v0v, z0v)  # (N,3)
    u1, v1, z1 = cam1.project(Pw)       # (N,)

    luma_warp = np.zeros((H, W), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)

    u1i = np.round(u1).astype(np.int32)
    v1i = np.round(v1).astype(np.int32)
    inb = (u1i >= 0) & (u1i < W) & (v1i >= 0) & (v1i < H) & (z1 > 0)

    u1i = u1i[inb]; v1i = v1i[inb]; z1 = z1[inb]
    u0v = u0v[inb]; v0v = v0v[inb]
    src_l = luma0[v0v.astype(np.int32), u0v.astype(np.int32)]

    d1 = depth1[v1i, u1i]
    ok = (d1 > 0) & (z1 <= d1 + 1e-3)
    u1i = u1i[ok]; v1i = v1i[ok]; z1 = z1[ok]
    u0v = u0v[ok]; v0v = v0v[ok]; src_l = src_l[ok]

    for uu, vv, zz, ll in zip(u1i, v1i, z1, src_l):
        if zz < zbuf[vv, uu]:
            zbuf[vv, uu] = zz
            luma_warp[vv, uu] = ll
            valid[vv, uu] = True

    flow = np.zeros((H, W, 2), dtype=np.float32)
    Pw_full = cam0.unproject(u0[mask0], v0[mask0], z0[mask0])
    u1f, v1f, _ = cam1.project(Pw_full)
    flow[..., 0][mask0] = u1f - u0[mask0]
    flow[..., 1][mask0] = v1f - v0[mask0]
    return luma_warp, valid, flow


# -------------------------------
# Signed frontier S (ternary)
# -------------------------------

def signed_frontier(depth0, flow, edge_thr=0.03):
    """
    S0 in {-1,0,+1}:
      - edges via |grad depth|
      - sign via sign(dot(grad depth, flow_dir))
    """
    dzdx = np.zeros_like(depth0)
    dzdy = np.zeros_like(depth0)
    dzdx[:, 1:-1] = 0.5 * (depth0[:, 2:] - depth0[:, :-2])
    dzdy[1:-1, :] = 0.5 * (depth0[2:, :] - depth0[:-2, :])
    gmag = np.sqrt(dzdx * dzdx + dzdy * dzdy)

    fu = flow[..., 0]
    fv = flow[..., 1]
    fn = np.sqrt(fu * fu + fv * fv) + 1e-9
    fux = fu / fn
    fvy = fv / fn

    dot = dzdx * fux + dzdy * fvy

    S = np.zeros_like(depth0, dtype=np.int8)
    edge = (gmag > edge_thr) & (depth0 > 0)
    S[edge & (dot > 0)] = 1
    S[edge & (dot < 0)] = -1
    return S.astype(np.float32), gmag.astype(np.float32)


# -------------------------------
# Importance I0 (precomputed leverage proxy)
# -------------------------------

def importance_map(depth0, luma0, gmag, focus_center=None):
    lum = luma0.copy()
    lum = lum / (lum.max() + 1e-9)
    gm = gmag.copy()
    gm = gm / (gm.max() + 1e-9)

    I = 0.65 * lum + 0.35 * gm
    I = np.clip(I, 0.0, 1.0)

    if focus_center is not None:
        cy, cx = focus_center
        H, W = depth0.shape
        yy, xx = np.meshgrid(np.arange(W), np.arange(H))
        rr2 = (xx - cy) ** 2 + (yy - cx) ** 2
        sigma2 = (0.12 * max(H, W)) ** 2
        bump = np.exp(-rr2 / (2.0 * sigma2)).astype(np.float32)
        I = np.clip(I + 0.5 * bump, 0.0, 1.0)

    return I.astype(np.float32)


# -------------------------------
# Learners: logistic + tiny MLP
# -------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bern_entropy(m):
    eps = 1e-9
    m = np.clip(m, eps, 1 - eps)
    return -(m * np.log2(m) + (1 - m) * np.log2(1 - m))


def d_bern_entropy_dm(m):
    eps = 1e-9
    m = np.clip(m, eps, 1 - eps)
    return np.log2((1 - m) / m)


def tv_and_grad(m, H, W, eta):
    M = m.reshape(H, W)
    dx = np.zeros_like(M)
    dy = np.zeros_like(M)
    dx[:, :-1] = M[:, 1:] - M[:, :-1]
    dy[:-1, :] = M[1:, :] - M[:-1, :]

    tv = eta * (np.sum(dx * dx) + np.sum(dy * dy))

    g = np.zeros_like(M)
    g[:, :-1] -= 2 * eta * dx[:, :-1]
    g[:, 1:]  += 2 * eta * dx[:, :-1]
    g[:-1, :] -= 2 * eta * dy[:-1, :]
    g[1:, :]  += 2 * eta * dy[:-1, :]
    return tv, g.reshape(-1)


def mdl_loss_and_grad_logits(logits, e_eff, S0, H, W,
                            lam=2.0, eta_tv=0.02,
                            c_pos=0.10, c_neg=0.14):
    m = sigmoid(logits)
    rate = np.sum(bern_entropy(m))
    dH = d_bern_entropy_dm(m)

    c_map = np.where(S0 > 0, c_pos, c_neg).astype(np.float32)
    dist = lam * np.sum((1.0 - m) * e_eff)
    comp = np.sum(c_map * m)

    # TV only valid for single-frame shapes
    if m.size == H * W and eta_tv > 0:
        tv, d_tv_dm = tv_and_grad(m, H, W, eta_tv)
    else:
        tv = 0.0
        d_tv_dm = 0.0

    loss = rate + comp + dist + tv
    dL_dm = dH + c_map - lam * e_eff + d_tv_dm
    dlog = dL_dm * (m * (1.0 - m))
    return loss, dlog, m


class LogisticMask:
    def __init__(self, D, seed=0):
        rng = np.random.default_rng(seed)
        self.w = (0.01 * rng.standard_normal(D)).astype(np.float32)
        self.b = np.float32(0.0)

    def forward(self, X):
        return X @ self.w + self.b

    def step(self, dw, db, lr):
        self.w -= lr * dw
        self.b -= lr * db


class TinyMLP:
    def __init__(self, D, Hh=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (0.05 * rng.standard_normal((D, Hh))).astype(np.float32)
        self.b1 = np.zeros((Hh,), dtype=np.float32)
        self.W2 = (0.05 * rng.standard_normal((Hh, 1))).astype(np.float32)
        self.b2 = np.zeros((1,), dtype=np.float32)

    def forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        logits = (h @ self.W2).reshape(-1) + self.b2[0]
        return logits, h

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


def train_logistic(X, e_eff, S0, H, W, epochs=200, lr=0.2, seed=0,
                   lam=2.0, eta_tv=0.02, c_pos=0.10, c_neg=0.14):
    model = LogisticMask(X.shape[1], seed=seed)
    for _ in range(epochs):
        logits = model.forward(X)
        loss, dlog, _ = mdl_loss_and_grad_logits(
            logits, e_eff, S0, H, W, lam=lam, eta_tv=eta_tv, c_pos=c_pos, c_neg=c_neg
        )
        dw = (X.T @ dlog).astype(np.float32) / X.shape[0]
        db = np.mean(dlog).astype(np.float32)
        model.step(dw, db, lr)
    logits = model.forward(X)
    loss, _, m = mdl_loss_and_grad_logits(
        logits, e_eff, S0, H, W, lam=lam, eta_tv=eta_tv, c_pos=c_pos, c_neg=c_neg
    )
    return model, m, loss


def train_mlp(X, e_eff, S0, H, W, epochs=250, lr=0.08, seed=1,
              lam=2.0, eta_tv=0.02, c_pos=0.10, c_neg=0.14):
    model = TinyMLP(X.shape[1], Hh=16, seed=seed)
    for _ in range(epochs):
        logits, h = model.forward(X)
        loss, dlog, _ = mdl_loss_and_grad_logits(
            logits, e_eff, S0, H, W, lam=lam, eta_tv=eta_tv, c_pos=c_pos, c_neg=c_neg
        )
        dh = (dlog[:, None] @ model.W2.T) * (1.0 - h * h)
        dW2 = (h.T @ dlog[:, None]).astype(np.float32) / X.shape[0]
        db2 = np.mean(dlog).astype(np.float32)
        dW1 = (X.T @ dh).astype(np.float32) / X.shape[0]
        db1 = np.mean(dh, axis=0).astype(np.float32)
        model.step((dW1, db1, dW2, np.array([db2], dtype=np.float32)), lr)
    logits, _ = model.forward(X)
    loss, _, m = mdl_loss_and_grad_logits(
        logits, e_eff, S0, H, W, lam=lam, eta_tv=eta_tv, c_pos=c_pos, c_neg=c_neg
    )
    return model, m, loss


# -------------------------------
# Feature assembly
# -------------------------------

def build_features(depth0, S0, I0, h, flow, luma0):
    H, W = depth0.shape
    d = depth0 / (depth0.max() + 1e-9)
    S = S0
    I = I0
    hh = h
    fm = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
    fm = fm / (fm.max() + 1e-9)
    lum = luma0 / (luma0.max() + 1e-9)

    X = np.stack([d, S, I, hh, fm, lum], axis=-1).astype(np.float32)
    X = X.reshape(-1, X.shape[-1])
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    return X


# -------------------------------
# Dataset generation
# -------------------------------

def rot_y(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def make_cameras(H, W, f, base_pos, base_yaw, dx, dyaw):
    cam0 = Camera(
        pos=np.array(base_pos, dtype=np.float32),
        R=rot_y(base_yaw).astype(np.float32),
        f=f, cx=(W - 1) / 2.0, cy=(H - 1) / 2.0
    )
    cam1 = Camera(
        pos=np.array([base_pos[0] + dx, base_pos[1], base_pos[2]], dtype=np.float32),
        R=rot_y(base_yaw + dyaw).astype(np.float32),
        f=f, cx=(W - 1) / 2.0, cy=(H - 1) / 2.0
    )
    return cam0, cam1


def build_pair(H, W, f, spheres, base_pos, base_yaw, dx, dyaw, focus_center=None,
               h_prev=None, beta=0.90):
    cam0, cam1 = make_cameras(H, W, f, base_pos, base_yaw, dx, dyaw)
    d0, l0, _ = render_scene(cam0, H, W, spheres)
    d1, l1, _ = render_scene(cam1, H, W, spheres)
    l1_w, _, flow = reproject_luma(d0, l0, cam0, cam1, d1)

    err = np.abs(l1 - l1_w).astype(np.float32)
    S0, gmag = signed_frontier(d0, flow, edge_thr=0.03)
    I0 = importance_map(d0, l0, gmag, focus_center=focus_center)

    if h_prev is None:
        h_prev = np.zeros((H, W), dtype=np.float32)
    visible = (d0 > 0).astype(np.float32)
    h = beta * h_prev + I0 * visible

    X = build_features(d0, S0, I0, h, flow, l0)

    return {
        "d0": d0, "l0": l0, "d1": d1, "l1": l1, "l1_w": l1_w, "flow": flow,
        "err": err, "S0": S0, "I0": I0, "h": h, "X": X,
        "h_next": h
    }


def stack_dataset(pairs):
    X = np.concatenate([p["X"] for p in pairs], axis=0)
    err = np.concatenate([p["err"].reshape(-1) for p in pairs], axis=0).astype(np.float32)
    S0 = np.concatenate([p["S0"].reshape(-1) for p in pairs], axis=0).astype(np.float32)
    I0 = np.concatenate([p["I0"].reshape(-1) for p in pairs], axis=0).astype(np.float32)
    h = np.concatenate([p["h"].reshape(-1) for p in pairs], axis=0).astype(np.float32)
    return X, err, S0, I0, h


# -------------------------------
# Threshold sweep (MDL' curve)
# -------------------------------

def mdl_proxy_threshold_sweep(m_prob, err, bits_per_active=0.6, compute_per_active=0.08):
    ths = np.linspace(0.05, 0.95, 19)
    vals = []
    for th in ths:
        active = (m_prob >= th)
        rate = bits_per_active * np.sum(active)
        comp = compute_per_active * np.sum(active)
        dist = np.sum(err[~active])
        vals.append(rate + comp + dist)
    return ths, np.array(vals, dtype=np.float32)


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--W", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="outputs_pda")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--alpha", type=float, default=25.0, help="Step A leverage weight")
    ap.add_argument("--kappa", type=float, default=8.0, help="Step B persistence weight")
    ap.add_argument("--beta", type=float, default=0.90, help="persistence decay")
    ap.add_argument("--lam", type=float, default=2.0, help="distortion weight")
    ap.add_argument("--eta_tv", type=float, default=0.02, help="TV prior weight")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    H, W = args.H, args.W
    f = 45.0

    spheres = [
        Sphere(center=np.array([0.0, 0.0, 3.2], dtype=np.float32),
               radius=1.10,
               color=np.array([0.72, 0.72, 0.72], dtype=np.float32),
               emissive=0.0),
        Sphere(center=np.array([1.35, 0.05, 3.05], dtype=np.float32),
               radius=0.42,
               color=np.array([1.0, 0.1, 0.1], dtype=np.float32),
               emissive=1.2),
    ]

    base_pos = (0.0, 0.0, 0.0)
    base_yaw = 0.0

    pairs = []
    h_prev = np.zeros((H, W), dtype=np.float32)
    focus_center = (int(0.55 * H), int(0.78 * W))

    for _ in range(args.pairs):
        dx = rng.uniform(-0.10, 0.10)
        dyaw = rng.uniform(-0.10, 0.10)
        pair = build_pair(
            H, W, f, spheres,
            base_pos, base_yaw, dx, dyaw,
            focus_center=focus_center,
            h_prev=h_prev,
            beta=args.beta
        )
        h_prev = pair["h_next"]
        pairs.append(pair)

    n_train = max(2, int(0.75 * len(pairs)))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    Xtr, err_tr, S_tr, I_tr, h_tr = stack_dataset(train_pairs)
    Xva, err_va, S_va, I_va, h_va = stack_dataset(val_pairs)

    # Step A + Step B effective distortion
    e_tr = err_tr * (1.0 + args.alpha * I_tr) * (1.0 + args.kappa * h_tr)
    e_va = err_va * (1.0 + args.alpha * I_va) * (1.0 + args.kappa * h_va)

    print("[train] logistic...")
    _, _, loss_log = train_logistic(
        Xtr, e_tr, S_tr, H, W,
        epochs=220, lr=0.25, seed=args.seed,
        lam=args.lam, eta_tv=args.eta_tv
    )
    print(f"  logistic train loss: {loss_log:.3f}")

    print("[train] tiny MLP...")
    mlp_model, _, loss_mlp = train_mlp(
        Xtr, e_tr, S_tr, H, W,
        epochs=260, lr=0.08, seed=args.seed + 1,
        lam=args.lam, eta_tv=args.eta_tv
    )
    print(f"  mlp train loss: {loss_mlp:.3f}")

    # Heldout mask probabilities
    logits_va, _ = mlp_model.forward(Xva)
    m_va = sigmoid(logits_va)

    ths, mdl_curve = mdl_proxy_threshold_sweep(m_va, err_va, bits_per_active=0.6, compute_per_active=0.08)
    th_best = ths[np.argmin(mdl_curve)]

    # visualize one frame
    vis = train_pairs[-1]
    d0 = vis["d0"]; S0 = vis["S0"]; I0 = vis["I0"]; err = vis["err"]
    l1w = vis["l1_w"]; l1 = vis["l1"]

    Xvis = vis["X"]
    logits_vis, _ = mlp_model.forward(Xvis)
    m_vis = sigmoid(logits_vis).reshape(H, W)
    chosen = (m_vis >= th_best).astype(np.float32)

    l1_pred = l1w.copy()
    l1_pred[chosen > 0] = l1[chosen > 0]
    err_after = np.abs(l1 - l1_pred)

    # Plots
    plt.figure(figsize=(14, 3.6))
    ax = plt.subplot(1, 4, 1); im=ax.imshow(d0, vmin=0); ax.set_title("Depth t0"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 4, 2); im=ax.imshow(S0, vmin=-1, vmax=1, cmap="bwr"); ax.set_title("Signed frontier S0"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 4, 3); im=ax.imshow(I0, vmin=0); ax.set_title("Importance I0 (precomputed)"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 4, 4); im=ax.imshow(err, vmin=0); ax.set_title("Occlusion/transport error"); plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(outdir / "figure_1_fields.png", dpi=160)

    plt.figure(figsize=(14, 3.6))
    ax = plt.subplot(1, 4, 1); im=ax.imshow(l1w, vmin=0); ax.set_title("Reprojected luma t0→t1"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 4, 2); im=ax.imshow(l1, vmin=0); ax.set_title("True luma t1"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 4, 3); im=ax.imshow(m_vis, vmin=0, vmax=1); ax.set_title("Learned refresh prob m"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 4, 4); im=ax.imshow(chosen, vmin=0, vmax=1); ax.set_title(f"Chosen mask (th={th_best:.2f})"); plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(outdir / "figure_2_masks.png", dpi=160)

    plt.figure(figsize=(10, 3.4))
    plt.plot(ths, mdl_curve, marker="o")
    plt.axvline(th_best, linestyle="--")
    plt.xlabel("threshold")
    plt.ylabel("MDL' proxy")
    plt.title("MDL' vs threshold (heldout)")
    plt.tight_layout()
    plt.savefig(outdir / "figure_3_mdl_curve.png", dpi=160)

    plt.figure(figsize=(14, 3.6))
    ax = plt.subplot(1, 3, 1); im=ax.imshow(err, vmin=0); ax.set_title("Error (stale reprojection)"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 3, 2); im=ax.imshow(err_after, vmin=0); ax.set_title("Error after refreshing chosen"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(1, 3, 3); im=ax.imshow(np.abs(err - err_after), vmin=0); ax.set_title("Error reduction"); plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(outdir / "figure_4_error_reduction.png", dpi=160)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print(f"[done] wrote figures to: {outdir.resolve()}")
    print("Try tweaking --alpha (Step A) and --kappa (Step B) if the mask is too sparse/dense.")
    print("Expectation: mask concentrates on disocclusion/frontier + bright hotspot.")


if __name__ == "__main__":
    main()
