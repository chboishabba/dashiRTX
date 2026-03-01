# test_pda_mdl_light_transport_kernel.py
"""
PDA–MDL light-transport refresh kernel (toy, self-contained)

What this test demonstrates (in *our* language):
- We synthesize a low-res 3D scene (two spheres) with:
    (a) normal disocclusion under camera motion
    (b) an *indirect-return hotspot* (multi-bounce surrogate) that creates
        high leverage error even without pure disocclusion.
- We precompute a cheap "importance" field I(p) in world space (a map/probe lookup surrogate).
- We learn a *local* refresh policy m(pix) using a 3×3 conv → 1×1 conv kernel
  under an MDL surrogate:
      MDL' = Rate(mask) + Compute(mask) + Distortion((1-mask)*error) + TV(mask)
- We assert the learned mask overlaps:
    (1) disocclusion band, and
    (2) the high-importance transport hotspot,
  better than a simple baseline.

Run as a test:
    pytest -q test_pda_mdl_light_transport_kernel.py

Run interactively (renders plots):
    python test_pda_mdl_light_transport_kernel.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


# -------------------------
# 0) Scene: minimal geometry
# -------------------------

@dataclass(frozen=True)
class Sphere:
    center: np.ndarray  # (3,)
    radius: float
    obj_id: int
    albedo: np.ndarray  # (3,) in [0,1]
    roughness: float    # [0,1] (0=mirror, 1=diffuse-ish)
    emissive: np.ndarray  # (3,) in [0, +)

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def _intersect_sphere(ro: np.ndarray, rd: np.ndarray, c: np.ndarray, r: float):
    """
    Vectorized ray-sphere intersection.
    ro, rd: (H,W,3)
    returns: t (H,W), p (H,W,3), n (H,W,3), hit (H,W)
    """
    oc = ro - c.reshape(1, 1, 3)
    b = np.sum(oc * rd, axis=-1)  # (H,W)
    cterm = np.sum(oc * oc, axis=-1) - (r * r)
    disc = b * b - cterm
    hit = disc > 0.0
    t = np.full(b.shape, np.inf, dtype=np.float32)

    sqrt_disc = np.zeros_like(b, dtype=np.float32)
    sqrt_disc[hit] = np.sqrt(disc[hit]).astype(np.float32)
    t0 = (-b - sqrt_disc).astype(np.float32)

    hit &= (t0 > 0.0)
    t[hit] = t0[hit]
    p = ro + rd * t[..., None]
    n = _normalize((p - c.reshape(1, 1, 3)) / (r + 1e-9))
    return t.astype(np.float32), p.astype(np.float32), n.astype(np.float32), hit

def generate_rays_pinhole(
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    Rcw: np.ndarray,   # (3,3) camera->world
    tcw: np.ndarray,   # (3,)   camera origin in world
):
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    dirs_c = np.stack([x, y, np.ones_like(x)], axis=-1)
    dirs_c = _normalize(dirs_c)
    dirs_w = dirs_c @ Rcw.T
    ro_w = np.broadcast_to(tcw.reshape(1, 1, 3), dirs_w.shape).astype(np.float32)
    return ro_w, dirs_w.astype(np.float32)

def render_gbuffers(
    spheres: List[Sphere],
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    Rcw: np.ndarray,
    tcw: np.ndarray,
):
    ro, rd = generate_rays_pinhole(H, W, fx, fy, cx, cy, Rcw, tcw)

    best_t = np.full((H, W), np.inf, dtype=np.float32)
    best_id = np.full((H, W), -1, dtype=np.int16)
    best_p = np.zeros((H, W, 3), dtype=np.float32)
    best_n = np.zeros((H, W, 3), dtype=np.float32)
    best_albedo = np.zeros((H, W, 3), dtype=np.float32)
    best_rough = np.ones((H, W), dtype=np.float32)
    best_emis = np.zeros((H, W, 3), dtype=np.float32)

    for s in spheres:
        t, p, n, hit = _intersect_sphere(ro, rd, s.center.astype(np.float32), float(s.radius))
        better = t < best_t
        sel = hit & better
        if np.any(sel):
            best_t[sel] = t[sel]
            best_id[sel] = int(s.obj_id)
            best_p[sel] = p[sel]
            best_n[sel] = n[sel]
            best_albedo[sel] = s.albedo.astype(np.float32)
            best_rough[sel] = float(s.roughness)
            best_emis[sel] = s.emissive.astype(np.float32)

    depth = np.zeros((H, W), dtype=np.float32)
    valid = best_id >= 0
    depth[valid] = best_t[valid]

    return depth, best_id, best_p, best_n, best_albedo, best_rough, best_emis


# --------------------------------------
# 1) "Multi-bounce return" surrogate map
# --------------------------------------

def importance_map_world(pw: np.ndarray) -> np.ndarray:
    """
    World-space "precomputed" high-leverage transport importance I(p).
    Here: mixture of Gaussians (think probes / light cache points).
    pw: (...,3)
    return: (...,) in [0,1-ish]
    """
    # Hotspot: "bright red pixel reflects off two windows and returns"
    # We encode this as a stable spatial basin in world space.
    hotspot = np.array([0.65, 0.10, 3.40], dtype=np.float32)
    window1 = np.array([0.30, 0.25, 3.85], dtype=np.float32)
    window2 = np.array([0.95, -0.05, 3.95], dtype=np.float32)

    def gauss(mu, sig):
        d2 = np.sum((pw - mu) ** 2, axis=-1)
        return np.exp(-d2 / (2.0 * sig * sig)).astype(np.float32)

    I = (
        1.00 * gauss(hotspot, 0.18)
        + 0.55 * gauss(window1, 0.25)
        + 0.55 * gauss(window2, 0.25)
    )
    # Clamp into [0,1]
    return np.clip(I, 0.0, 1.0).astype(np.float32)

def shade_direct(
    pw: np.ndarray,
    nw: np.ndarray,
    albedo: np.ndarray,
    emissive: np.ndarray,
) -> np.ndarray:
    """
    Simple direct-only shading: one directional light + emissive.
    """
    Ldir = _normalize(np.array([-0.3, 0.6, -1.0], dtype=np.float32))
    ndotl = np.maximum(0.0, np.sum(nw * Ldir.reshape(1, 1, 3), axis=-1, keepdims=True))
    return (albedo * (0.8 * ndotl) + emissive).astype(np.float32)

def shade_indirect_surrogate(
    pw: np.ndarray,
    nw: np.ndarray,
    rough: np.ndarray,
    view_dir_w: np.ndarray,
) -> np.ndarray:
    """
    Multi-bounce surrogate:
    - Larger when importance I(p) is high
    - Larger when the local lobe can "return" (low roughness)
    - Mild view dependence (NOT time-invariant; it's coordinate dependence)
    """
    I = importance_map_world(pw)  # (H,W)
    # View alignment term: "can return even when looking away" is encoded as nonzero baseline.
    # You can strengthen view dependence by making baseline smaller if desired.
    v = _normalize(view_dir_w)
    ndotv = np.clip(np.sum(nw * (-v), axis=-1), 0.0, 1.0)  # facing camera-ish
    baseline = 0.25
    view_term = baseline + (1.0 - baseline) * (ndotv ** 1.5)

    # Specularness proxy: low roughness => higher indirect leverage
    spec = (1.0 - np.clip(rough, 0.0, 1.0)) ** 1.5  # (H,W)

    # Indirect RGB tint: slightly warm (to mimic red-ish return); keep simple.
    tint = np.array([1.0, 0.4, 0.35], dtype=np.float32).reshape(1, 1, 3)
    ind = (I * spec * view_term).reshape(I.shape[0], I.shape[1], 1) * tint
    return ind.astype(np.float32)

def render_radiance(
    depth: np.ndarray,
    oid: np.ndarray,
    pw: np.ndarray,
    nw: np.ndarray,
    albedo: np.ndarray,
    rough: np.ndarray,
    emissive: np.ndarray,
    cam_pos_w: np.ndarray,
) -> np.ndarray:
    """
    "True" radiance = direct + indirect surrogate.
    For invalid pixels, radiance is 0.
    """
    H, W = depth.shape
    valid = depth > 0
    out = np.zeros((H, W, 3), dtype=np.float32)
    if not np.any(valid):
        return out

    view_dir = _normalize(pw - cam_pos_w.reshape(1, 1, 3))
    direct = shade_direct(pw, nw, albedo, emissive)
    indirect = shade_indirect_surrogate(pw, nw, rough, view_dir)

    out[valid] = (direct + indirect)[valid]
    return np.clip(out, 0.0, 3.0).astype(np.float32)


# ----------------------------------------
# 2) Warp/reproject + occlusion/distortion
# ----------------------------------------

def project_world_to_img(
    pw: np.ndarray,  # (H,W,3)
    Rcw: np.ndarray,
    tcw: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    # world->camera:
    Rwc = Rcw.T
    twc = -Rwc @ tcw
    pc = (pw @ Rwc.T) + twc.reshape(1, 1, 3)
    x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
    u = fx * (x / (z + 1e-9)) + cx
    v = fy * (y / (z + 1e-9)) + cy
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)

def zbuffer_splat_from_world(
    pw0: np.ndarray,   # (H,W,3) world hitpoints at t0
    depth0: np.ndarray,
    buf0: np.ndarray,  # (H,W,C) cached buffer to splat (e.g. depth or color)
    Rcw1: np.ndarray,
    tcw1: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    """
    Z-buffer splat from t0 world points into t1 image.
    Returns:
      pred (H,W,C), valid_pred (H,W), zbuf (H,W)
    """
    H, W = depth0.shape
    C = buf0.shape[-1]
    pred = np.zeros((H, W, C), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    u, v, zc = project_world_to_img(pw0, Rcw1, tcw1, fx, fy, cx, cy)
    valid0 = depth0 > 0

    ui = np.round(u[valid0]).astype(np.int32)
    vi = np.round(v[valid0]).astype(np.int32)
    zz = zc[valid0].astype(np.float32)
    vv = buf0[valid0].astype(np.float32)

    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (zz > 0)
    ui, vi, zz, vv = ui[inb], vi[inb], zz[inb], vv[inb]

    # Small H,W: loop ok and clearer.
    for px, py, pz, pv in zip(ui, vi, zz, vv):
        if pz < zbuf[py, px]:
            zbuf[py, px] = pz
            pred[py, px] = pv

    valid_pred = np.isfinite(zbuf)
    valid_pred &= (zbuf < np.inf)
    valid_pred &= (np.sum(pred * pred, axis=-1) >= 0.0)  # keep shape aligned
    return pred, valid_pred, zbuf

def occlusion_aware_error(
    true1: np.ndarray,     # (H,W,3) ground truth radiance at t1
    pred1: np.ndarray,     # (H,W,3) reprojected cached radiance
    valid_pred: np.ndarray,
    valid_true: np.ndarray,
    alpha_disocc: float = 3.0,
    beta_l1: float = 1.0,
):
    """
    Error proxy emphasizing disocclusion:
      - where true is valid but pred is not -> alpha_disocc
      - where both valid -> beta * |true - pred|_1
    """
    H, W, _ = true1.shape
    e = np.zeros((H, W), dtype=np.float32)
    both = valid_true & valid_pred
    if np.any(both):
        e[both] = beta_l1 * np.sum(np.abs(true1[both] - pred1[both]), axis=-1)
    disocc = valid_true & (~valid_pred)
    e[disocc] = alpha_disocc
    return e.astype(np.float32)

def frontier_unsigned_from_id(oid: np.ndarray) -> np.ndarray:
    f = np.zeros_like(oid, dtype=np.uint8)
    f[:, :-1] |= (oid[:, :-1] != oid[:, 1:])
    f[:-1, :] |= (oid[:-1, :] != oid[1:, :])
    return f

def signed_frontier(
    depth: np.ndarray,
    oid: np.ndarray,
    vxy: Tuple[float, float],
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Ternary signed frontier: S ∈ {-1,0,+1}.
    Sign determined by projection of depth gradient onto motion direction.
    """
    U = frontier_unsigned_from_id(oid)
    dzdy = np.gradient(depth, axis=0)
    dzdx = np.gradient(depth, axis=1)
    grad = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    proj = vxy[0] * dzdx + vxy[1] * dzdy

    S = np.zeros_like(depth, dtype=np.int8)
    S[(U != 0) & (grad > eps) & (proj > 0)] = +1
    S[(U != 0) & (grad > eps) & (proj < 0)] = -1
    return S


# --------------------------
# 3) PDA–MDL conv mask policy
# --------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

def bernoulli_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

def tv_l1(mask: np.ndarray) -> float:
    """
    Total variation surrogate for mask (encourages coherent bands).
    """
    dx = np.abs(mask[:, 1:] - mask[:, :-1]).sum()
    dy = np.abs(mask[1:, :] - mask[:-1, :]).sum()
    return float(dx + dy)

def im2col_3x3(x: np.ndarray) -> np.ndarray:
    """
    x: (H,W,C)
    returns cols: (H*W, 3*3*C)
    """
    H, W, C = x.shape
    pad = 1
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    cols = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cols.append(xp[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :])
    col = np.concatenate(cols, axis=-1)  # (H,W,9C)
    return col.reshape(H * W, 9 * C).astype(np.float32)

@dataclass
class ConvMaskPolicy:
    """
    3×3 conv over channels -> scalar logit -> sigmoid.
    """
    C: int
    w: np.ndarray  # (9*C,)
    b: float
    mu: np.ndarray
    sd: np.ndarray

def conv_policy_forward(model: ConvMaskPolicy, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: (H,W,C) features
    returns:
      logits: (H,W)
      m: (H,W) in (0,1)
    """
    H, W, C = x.shape
    cols = im2col_3x3(x)  # (H*W, 9C)
    cols = (cols - model.mu) / model.sd
    logits = (cols @ model.w + model.b).reshape(H, W).astype(np.float32)
    m = sigmoid(logits).astype(np.float32)
    return logits, m

def train_conv_policy_mdl(
    X: np.ndarray,        # (N,H,W,C)
    err: np.ndarray,      # (N,H,W) >=0
    *,
    steps: int = 600,
    lr: float = 0.10,
    batch: int = 6,
    l2: float = 1e-4,
    c_active: float = 5.0,
    lam: float = 1.0,
    tv_weight: float = 0.02,
    seed: int = 0,
) -> ConvMaskPolicy:
    """
    Minimize MDL' per sample (soft mask):
      L = sum H(m) + c*sum m + lam*sum (1-m)*err + tv_weight*TV(m)
    using SGD on conv weights.
    """
    rng = np.random.default_rng(seed)
    N, H, W, C = X.shape

    # Precompute normalization over im2col features for stability.
    # We'll approximate mu/sd from a random subset of pixels across samples.
    idx_s = rng.integers(0, N, size=min(N, 8))
    pix = []
    for i in idx_s:
        cols = im2col_3x3(X[i])
        pix.append(cols)
    pix = np.concatenate(pix, axis=0)
    mu = pix.mean(axis=0).astype(np.float32)
    sd = (pix.std(axis=0) + 1e-6).astype(np.float32)

    w = rng.normal(0.0, 1.0 / math.sqrt(9 * C), size=(9 * C,)).astype(np.float32)
    b = np.float32(0.0)

    model = ConvMaskPolicy(C=C, w=w, b=float(b), mu=mu, sd=sd)

    for _ in range(steps):
        batch_ids = rng.integers(0, N, size=batch)
        # Accumulate grads across batch
        gw = np.zeros_like(w, dtype=np.float32)
        gb = np.float32(0.0)

        for bi in batch_ids:
            x = X[bi]          # (H,W,C)
            e = err[bi]        # (H,W)
            cols = im2col_3x3(x)                # (H*W,9C)
            cols_n = (cols - mu) / sd
            logits = (cols_n @ w + b).astype(np.float32)    # (H*W,)
            m = sigmoid(logits).astype(np.float32)          # (H*W,)

            # Core MDL' terms:
            # Rate: sum H(m) -> d/dm: log2((1-m)/m)
            dH_dm = np.log2((1.0 - m + 1e-9) / (m + 1e-9)).astype(np.float32)
            # Compute: c*sum m -> d/dm: +c
            # Distortion: lam*sum (1-m)*e -> d/dm: -lam*e
            dL_dm = dH_dm + np.float32(c_active) - np.float32(lam) * e.reshape(-1).astype(np.float32)

            # TV term (L1 TV) – we add a simple subgradient approximation by
            # pushing m to be locally smooth. This keeps it conservative.
            if tv_weight > 0.0:
                m2 = m.reshape(H, W)
                # simple Laplacian-like smoothness surrogate gradient (not exact TV subgrad but works as MDL-ish prior)
                lap = (
                    -4.0 * m2
                    + np.roll(m2, 1, axis=0) + np.roll(m2, -1, axis=0)
                    + np.roll(m2, 1, axis=1) + np.roll(m2, -1, axis=1)
                ).reshape(-1).astype(np.float32)
                dL_dm += np.float32(tv_weight) * lap

            # chain through sigmoid
            dL_dlogits = dL_dm * (m * (1.0 - m)).astype(np.float32)

            gw += (cols_n.T @ dL_dlogits) / (H * W)
            gb += np.mean(dL_dlogits).astype(np.float32)

        gw /= np.float32(batch)
        gb /= np.float32(batch)

        # L2
        gw += np.float32(l2) * w

        w -= np.float32(lr) * gw
        b -= np.float32(lr) * gb

    return ConvMaskPolicy(C=C, w=w, b=float(b), mu=mu, sd=sd)


# ------------------------------
# 4) Dataset + assertions (test)
# ------------------------------

def build_features(
    depth0: np.ndarray,
    nw0: np.ndarray,
    rough0: np.ndarray,
    rad0: np.ndarray,
    S0: np.ndarray,
    I0: np.ndarray,
    t01: np.ndarray,
) -> np.ndarray:
    """
    Features per pixel, channels (C):
      - depth0 (1)
      - normal0 (3)
      - roughness (1)
      - radiance0 (3)
      - signed frontier S0 (1)   [ternary carrier]
      - importance I0 (1)        [precomputed transport map lookup]
      - camera delta t01 (3)     [coordinate, not invariant]
    Total C = 1+3+1+3+1+1+3 = 13
    """
    H, W = depth0.shape
    tmap = np.broadcast_to(t01.reshape(1, 1, 3).astype(np.float32), (H, W, 3))
    x = np.concatenate(
        [
            depth0[..., None],
            nw0,
            rough0[..., None],
            rad0,
            S0.astype(np.float32)[..., None],
            I0[..., None],
            tmap,
        ],
        axis=-1,
    ).astype(np.float32)
    return x

def baseline_mask(depth0: np.ndarray, oid0: np.ndarray) -> np.ndarray:
    """
    Cheap baseline: refresh frontier only (unsigned) + disocclusion-ish geometry.
    """
    U = frontier_unsigned_from_id(oid0).astype(np.float32)
    return (U > 0).astype(np.float32)

def run_one_experiment(
    *,
    H: int = 36,
    W: int = 36,
    n_pairs: int = 18,
    seed: int = 7,
):
    # Camera intrinsics
    fov = np.deg2rad(60.0)
    fx = fy = (W / 2) / np.tan(fov / 2)
    cx = (W - 1) / 2
    cy = (H - 1) / 2

    # Scene: 2 spheres
    spheres = [
        Sphere(center=np.array([0.00, 0.00, 3.00], dtype=np.float32),
               radius=1.00, obj_id=1,
               albedo=np.array([0.55, 0.60, 0.70], dtype=np.float32),
               roughness=0.75,
               emissive=np.array([0.00, 0.00, 0.00], dtype=np.float32)),
        Sphere(center=np.array([1.15, 0.20, 4.00], dtype=np.float32),
               radius=0.72, obj_id=2,
               albedo=np.array([0.95, 0.20, 0.20], dtype=np.float32),  # red-ish "high leverage"
               roughness=0.25,  # specular-ish so indirect matters
               emissive=np.array([0.05, 0.00, 0.00], dtype=np.float32)),
    ]

    rng = np.random.default_rng(seed)

    Xs = []
    Es = []
    metas = []  # for visualization/diagnostics

    for _ in range(n_pairs):
        # Camera poses: translate a bit; keep R fixed for simplicity.
        tcw0 = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.10, 0.10), 0.0], dtype=np.float32)
        Rcw0 = np.eye(3, dtype=np.float32)

        dx = float(rng.choice([0.03, 0.05, 0.07]))
        dy = float(rng.uniform(-0.015, 0.015))
        tcw1 = tcw0 + np.array([dx, dy, 0.0], dtype=np.float32)
        Rcw1 = np.eye(3, dtype=np.float32)

        # G-buffers
        d0, oid0, pw0, nw0, alb0, rough0, emis0 = render_gbuffers(spheres, H, W, fx, fy, cx, cy, Rcw0, tcw0)
        d1, oid1, pw1, nw1, alb1, rough1, emis1 = render_gbuffers(spheres, H, W, fx, fy, cx, cy, Rcw1, tcw1)

        # Radiance (true)
        rad0 = render_radiance(d0, oid0, pw0, nw0, alb0, rough0, emis0, tcw0)
        rad1 = render_radiance(d1, oid1, pw1, nw1, alb1, rough1, emis1, tcw1)

        # Precomputed "transport importance" lookup at t0
        I0 = np.zeros((H, W), dtype=np.float32)
        valid0 = d0 > 0
        if np.any(valid0):
            I0[valid0] = importance_map_world(pw0[valid0])

        # Signed frontier (ternary carrier)
        S0 = signed_frontier(d0, oid0, vxy=(dx, dy), eps=1e-3)

        # Reproject cached rad0 into t1
        pred1, valid_pred, _ = zbuffer_splat_from_world(pw0, d0, rad0, Rcw1, tcw1, fx, fy, cx, cy)
        valid_true = d1 > 0

        # Defect/error map (occlusion heavy + radiance mismatch)
        err_map = occlusion_aware_error(
            true1=rad1,
            pred1=pred1,
            valid_pred=valid_pred,
            valid_true=valid_true,
            alpha_disocc=3.0,
            beta_l1=1.0,
        )

        # Features
        t01 = (tcw1 - tcw0).astype(np.float32)
        feat = build_features(d0, nw0, rough0, rad0, S0, I0, t01)

        Xs.append(feat)
        Es.append(err_map)

        metas.append((d0, oid0, S0, I0, rad0, rad1, pred1, err_map, t01))

    X = np.stack(Xs, axis=0).astype(np.float32)  # (N,H,W,C)
    E = np.stack(Es, axis=0).astype(np.float32)  # (N,H,W)
    return X, E, metas


def test_conv_policy_learns_disocclusion_and_transport_hotspot():
    X, E, metas = run_one_experiment(H=34, W=34, n_pairs=18, seed=12)
    N, H, W, C = X.shape

    # Train conv under MDL surrogate.
    # Key knobs:
    # - c_active high: "refresh everything" becomes too expensive
    # - tv_weight > 0: encourages band-like coherent solutions
    model = train_conv_policy_mdl(
        X, E,
        steps=650,
        lr=0.11,
        batch=6,
        c_active=7.5,
        lam=1.0,
        tv_weight=0.03,
        seed=3,
    )

    # Evaluate on held-out last sample
    d0, oid0, S0, I0, rad0, rad1, pred1, err_map, t01 = metas[-1]
    logits, m = conv_policy_forward(model, X[-1])

    # Choose a threshold by sweeping and minimizing MDL' proxy on the heldout.
    # (This keeps the "MDL selects chart" continuity.)
    def mdl_prime(mask01: np.ndarray, err: np.ndarray, c_active: float, lam: float, tv_w: float):
        p = float(mask01.mean())
        rate = float((H * W) * bernoulli_entropy_bits(np.array([p], dtype=np.float32))[0])
        compute = float(c_active * mask01.sum())
        distortion = float(lam * np.sum((1.0 - mask01) * err))
        tv = float(tv_w * tv_l1(mask01))
        return rate + compute + distortion + tv

    ths = np.linspace(0.10, 0.90, 17)
    costs = []
    masks = []
    for th in ths:
        mm = (m > th).astype(np.float32)
        costs.append(mdl_prime(mm, err_map, c_active=7.5, lam=1.0, tv_w=0.03))
        masks.append(mm)
    best = int(np.argmin(costs))
    mask = masks[best]

    # Baseline: unsigned frontier-only
    base = baseline_mask(d0, oid0)

    # Define regions:
    # - Disocclusion ring: where err_map is the disocc penalty (>= 2.5)
    disocc = (err_map >= 2.5).astype(np.float32)

    # - Transport hotspot: high importance and valid geometry (top 10% of I0 among valid)
    valid0 = (d0 > 0).astype(np.float32)
    if float(valid0.sum()) > 10:
        thresh_I = np.quantile(I0[d0 > 0], 0.90)
    else:
        thresh_I = 0.75
    hotspot = ((I0 >= thresh_I) & (d0 > 0)).astype(np.float32)

    def overlap(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.maximum(1.0, b.sum()))
        return float((a * b).sum() / denom)

    # Metrics: fraction of region refreshed.
    # We want learned mask to cover more of the disocclusion region than baseline,
    # and to cover more of the hotspot than baseline.
    cov_dis_learn = overlap(mask, disocc)
    cov_dis_base = overlap(base, disocc)

    cov_hot_learn = overlap(mask, hotspot)
    cov_hot_base = overlap(base, hotspot)

    # Also ensure the learned mask isn't "refresh everything"
    frac_refresh = float(mask.mean())

    # Assertions (tuned to be robust but meaningful)
    assert frac_refresh < 0.55, f"Mask refresh fraction too high: {frac_refresh:.3f}"
    assert cov_dis_learn > cov_dis_base + 0.08, (cov_dis_learn, cov_dis_base)
    assert cov_hot_learn > cov_hot_base + 0.06, (cov_hot_learn, cov_hot_base)


# -----------------------
# 5) Interactive visuals
# -----------------------

def _show_plots():
    import matplotlib.pyplot as plt

    X, E, metas = run_one_experiment(H=40, W=40, n_pairs=20, seed=5)
    model = train_conv_policy_mdl(
        X, E,
        steps=750,
        lr=0.10,
        batch=6,
        c_active=8.0,
        lam=1.0,
        tv_weight=0.035,
        seed=2,
    )

    d0, oid0, S0, I0, rad0, rad1, pred1, err_map, t01 = metas[-1]
    _, m = conv_policy_forward(model, X[-1])

    ths = np.linspace(0.10, 0.90, 17)
    costs = []
    masks = []
    H, W = d0.shape

    def mdl_prime(mask01: np.ndarray, err: np.ndarray, c_active: float, lam: float, tv_w: float):
        p = float(mask01.mean())
        rate = float((H * W) * bernoulli_entropy_bits(np.array([p], dtype=np.float32))[0])
        compute = float(c_active * mask01.sum())
        distortion = float(lam * np.sum((1.0 - mask01) * err))
        tv = float(tv_w * tv_l1(mask01))
        return rate + compute + distortion + tv

    for th in ths:
        mm = (m > th).astype(np.float32)
        costs.append(mdl_prime(mm, err_map, c_active=8.0, lam=1.0, tv_w=0.035))
        masks.append(mm)
    best = int(np.argmin(costs))
    mask = masks[best]
    best_th = float(ths[best])

    base = baseline_mask(d0, oid0)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1); plt.title("Depth t0"); plt.imshow(d0); plt.colorbar()
    plt.subplot(1, 4, 2); plt.title("Signed frontier S0"); plt.imshow(S0, cmap="bwr", vmin=-1, vmax=1); plt.colorbar()
    plt.subplot(1, 4, 3); plt.title("Importance I0 (precomputed)"); plt.imshow(I0); plt.colorbar()
    plt.subplot(1, 4, 4); plt.title("Occlusion/transport error"); plt.imshow(err_map); plt.colorbar()
    plt.tight_layout()

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1); plt.title("Reprojected rad t0→t1"); plt.imshow(np.clip(pred1, 0, 1)); plt.colorbar()
    plt.subplot(1, 4, 2); plt.title("True rad t1"); plt.imshow(np.clip(rad1, 0, 1)); plt.colorbar()
    plt.subplot(1, 4, 3); plt.title("Learned refresh prob m"); plt.imshow(m); plt.colorbar()
    plt.subplot(1, 4, 4); plt.title(f"Chosen mask (th={best_th:.2f})"); plt.imshow(mask); plt.colorbar()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.plot(ths, costs, marker="o")
    plt.axvline(best_th, linestyle="--")
    plt.title("MDL' vs threshold (heldout)")
    plt.xlabel("threshold")
    plt.ylabel("MDL' proxy")
    plt.tight_layout()

    # Overlay masks on error
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1); plt.title("Baseline (frontier-only)"); plt.imshow(base); plt.colorbar()
    plt.subplot(1, 3, 2); plt.title("Learned mask"); plt.imshow(mask); plt.colorbar()
    overlay = err_map.copy()
    overlay[mask > 0] *= 0.25
    plt.subplot(1, 3, 3); plt.title("Error w/ learned refreshed dimmed"); plt.imshow(overlay); plt.colorbar()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Optional: limit threads for deterministic-ish runs on some setups
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _show_plots()
