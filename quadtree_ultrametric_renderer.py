#!/usr/bin/env python3
"""
quadtree_ultrametric_renderer.py

Minimal, self-contained quadtree (ultrametric) renderer prototype.
Builds a tree from synthetic signals and accumulates node contributions
instead of per-pixel loops to demonstrate O(N_active) behavior.

Run:
  python quadtree_ultrametric_renderer.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import math
import numpy as np

try:
    import numba as nb
    _NUMBA = True
except Exception:
    nb = None
    _NUMBA = False


def synth_fields(H: int, W: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    cx, cy = 0.52 * W, 0.48 * H
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.45 * min(H, W))
    band = np.abs(r - 0.55) < 0.05

    S = np.zeros((H, W), dtype=np.int8)
    S[(r >= 0.55) & band] = +1
    S[(r < 0.55) & band] = -1

    e = 0.12 * band.astype(np.float32) + 0.03 * rng.random((H, W), dtype=np.float32)
    I = (0.4 * band.astype(np.float32) + 0.6 * rng.random((H, W), dtype=np.float32)).astype(np.float32)

    texture = (0.4 * band.astype(np.float32) + 0.6 * rng.random((H, W), dtype=np.float32)).astype(np.float32)
    texture = texture / (texture.max() + 1e-9)

    m = np.clip(0.15 + 0.9 * band.astype(np.float32) + 0.1 * rng.random((H, W), dtype=np.float32), 0, 1)
    return e, S, I, texture, m


@dataclass
class QNode:
    x0: int
    y0: int
    size: int
    depth: int
    children: Optional[List["QNode"]] = None
    mean_frontier: float = 0.0
    mean_I: float = 0.0
    mean_e: float = 0.0


def node_stats(node: QNode, e: np.ndarray, S: np.ndarray, I: np.ndarray) -> None:
    x0, y0, sz = node.x0, node.y0, node.size
    patch_e = e[y0:y0+sz, x0:x0+sz]
    patch_S = S[y0:y0+sz, x0:x0+sz]
    patch_I = I[y0:y0+sz, x0:x0+sz]

    node.mean_e = float(np.mean(patch_e))
    node.mean_frontier = float(np.mean((np.abs(patch_S) > 0).astype(np.float32)))
    node.mean_I = float(np.mean(patch_I))


def subdivide(node: QNode) -> List[QNode]:
    sz = node.size
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


def build_quadtree(e: np.ndarray, S: np.ndarray, I: np.ndarray, m: np.ndarray, min_size: int = 4, refine_scale: float = 1.0) -> QNode:
    H, W = e.shape
    assert H == W, "prototype assumes square input"

    root = QNode(0, 0, H, 0)
    node_stats(root, e, S, I)

    def refine(n: QNode) -> bool:
        if n.size <= min_size:
            return False
        x0, y0, sz = n.x0, n.y0, n.size
        mean_m = float(np.mean(m[y0:y0+sz, x0:x0+sz]))
        m_thr = min(0.95, 0.25 * refine_scale)
        f_thr = min(0.95, 0.03 * refine_scale)
        i_thr = min(0.95, 0.35 * refine_scale)
        e_thr = min(0.95, 0.10 * refine_scale)
        return (mean_m > m_thr) or (n.mean_frontier > f_thr) or (n.mean_I > i_thr) or (n.mean_e > e_thr)

    def rec(n: QNode):
        if refine(n):
            n.children = subdivide(n)
            for c in n.children:
                node_stats(c, e, S, I)
                rec(c)

    rec(root)
    return root


def quadtree_render(root: QNode, texture: np.ndarray, view_param: float = 1.0, bounce_cost: int = 0) -> np.ndarray:
    H, W = texture.shape
    out = np.zeros_like(texture, dtype=np.float32)

    def kernel(n: QNode, v: float, bcost: int) -> float:
        k = 0.25 + 1.5 * (0.6 * n.mean_I + 0.4 * n.mean_frontier) * (0.5 + 0.5 * math.sin(v))
        for _ in range(bcost):
            k = math.sin(k * 1.13 + 0.1)
        return float(k)

    for n in gather_leaves(root):
        x0, y0, sz = n.x0, n.y0, n.size
        avg = float(np.mean(texture[y0:y0+sz, x0:x0+sz]))
        out[y0:y0+sz, x0:x0+sz] += kernel(n, view_param, bounce_cost) * avg

    return out


def quadtree_render_vectorized(root: QNode, texture: np.ndarray, view_param: float = 1.0, bounce_cost: int = 0) -> np.ndarray:
    H, W = texture.shape
    out = np.zeros_like(texture, dtype=np.float32)

    leaves = gather_leaves(root)
    xs = np.array([n.x0 for n in leaves], dtype=np.int32)
    ys = np.array([n.y0 for n in leaves], dtype=np.int32)
    szs = np.array([n.size for n in leaves], dtype=np.int32)
    mean_I = np.array([n.mean_I for n in leaves], dtype=np.float32)
    mean_frontier = np.array([n.mean_frontier for n in leaves], dtype=np.float32)

    # integral image for fast block averages
    integ = texture.cumsum(axis=0).cumsum(axis=1)
    x1 = xs + szs - 1
    y1 = ys + szs - 1

    def rect_sum(x0, y0, x1, y1):
        total = integ[y1, x1]
        if x0 > 0:
            total = total - integ[y1, x0 - 1]
        if y0 > 0:
            total = total - integ[y0 - 1, x1]
        if x0 > 0 and y0 > 0:
            total = total + integ[y0 - 1, x0 - 1]
        return total

    sums = np.empty(len(leaves), dtype=np.float32)
    for i in range(len(leaves)):
        sums[i] = rect_sum(xs[i], ys[i], x1[i], y1[i])
    avgs = sums / (szs * szs)

    kernels = 0.25 + 1.5 * (0.6 * mean_I + 0.4 * mean_frontier) * (0.5 + 0.5 * math.sin(view_param))
    for _ in range(bounce_cost):
        kernels = np.sin(kernels * 1.13 + 0.1)

    for i in range(len(leaves)):
        x0, y0, sz = xs[i], ys[i], szs[i]
        out[y0:y0+sz, x0:x0+sz] += kernels[i] * avgs[i]

    return out


if _NUMBA:
    @nb.njit
    def _fill_blocks(out, xs, ys, szs, vals):
        for i in range(xs.shape[0]):
            x0 = xs[i]
            y0 = ys[i]
            sz = szs[i]
            v = vals[i]
            for yy in range(y0, y0 + sz):
                for xx in range(x0, x0 + sz):
                    out[yy, xx] += v

    @nb.njit
    def _render_blocks_numba(out, integ, xs, ys, szs, mean_I, mean_frontier, view_param, bounce_cost):
        for i in range(xs.shape[0]):
            x0 = xs[i]
            y0 = ys[i]
            sz = szs[i]
            x1 = x0 + sz - 1
            y1 = y0 + sz - 1

            total = integ[y1, x1]
            if x0 > 0:
                total -= integ[y1, x0 - 1]
            if y0 > 0:
                total -= integ[y0 - 1, x1]
            if x0 > 0 and y0 > 0:
                total += integ[y0 - 1, x0 - 1]

            avg = total / (sz * sz)
            kernel = 0.25 + 1.5 * (0.6 * mean_I[i] + 0.4 * mean_frontier[i]) * (0.5 + 0.5 * math.sin(view_param))
            for _ in range(bounce_cost):
                kernel = math.sin(kernel * 1.13 + 0.1)
            v = kernel * avg
            for yy in range(y0, y0 + sz):
                for xx in range(x0, x0 + sz):
                    out[yy, xx] += v


def quadtree_render_numba(root: QNode, texture: np.ndarray, view_param: float = 1.0, bounce_cost: int = 0) -> np.ndarray:
    if not _NUMBA:
        raise RuntimeError("numba not available")

    H, W = texture.shape
    out = np.zeros_like(texture, dtype=np.float32)

    leaves = gather_leaves(root)
    xs = np.array([n.x0 for n in leaves], dtype=np.int32)
    ys = np.array([n.y0 for n in leaves], dtype=np.int32)
    szs = np.array([n.size for n in leaves], dtype=np.int32)
    mean_I = np.array([n.mean_I for n in leaves], dtype=np.float32)
    mean_frontier = np.array([n.mean_frontier for n in leaves], dtype=np.float32)

    integ = texture.cumsum(axis=0).cumsum(axis=1)
    x1 = xs + szs - 1
    y1 = ys + szs - 1

    sums = np.empty(len(leaves), dtype=np.float32)
    for i in range(len(leaves)):
        total = integ[y1[i], x1[i]]
        if xs[i] > 0:
            total -= integ[y1[i], xs[i] - 1]
        if ys[i] > 0:
            total -= integ[ys[i] - 1, x1[i]]
        if xs[i] > 0 and ys[i] > 0:
            total += integ[ys[i] - 1, xs[i] - 1]
        sums[i] = total

    avgs = sums / (szs * szs)
    kernels = 0.25 + 1.5 * (0.6 * mean_I + 0.4 * mean_frontier) * (0.5 + 0.5 * math.sin(view_param))
    for _ in range(bounce_cost):
        kernels = np.sin(kernels * 1.13 + 0.1)
    vals = kernels * avgs

    _fill_blocks(out, xs, ys, szs, vals.astype(np.float32))
    return out


def quadtree_render_numba_full(root: QNode, texture: np.ndarray, view_param: float = 1.0, bounce_cost: int = 0) -> np.ndarray:
    if not _NUMBA:
        raise RuntimeError("numba not available")

    H, W = texture.shape
    out = np.zeros_like(texture, dtype=np.float32)

    leaves = gather_leaves(root)
    xs = np.array([n.x0 for n in leaves], dtype=np.int32)
    ys = np.array([n.y0 for n in leaves], dtype=np.int32)
    szs = np.array([n.size for n in leaves], dtype=np.int32)
    mean_I = np.array([n.mean_I for n in leaves], dtype=np.float32)
    mean_frontier = np.array([n.mean_frontier for n in leaves], dtype=np.float32)

    integ = texture.cumsum(axis=0).cumsum(axis=1)
    _render_blocks_numba(out, integ, xs, ys, szs, mean_I, mean_frontier, view_param, bounce_cost)
    return out


def main():
    H = W = 96
    e, S, I, texture, m = synth_fields(H, W, seed=0)
    root = build_quadtree(e, S, I, m, min_size=4)
    leaves = gather_leaves(root)
    out = quadtree_render(root, texture, view_param=1.2)

    print(f"[quadtree] leaves={len(leaves)} vs pixels={H*W}")
    print(f"[render] out mean={out.mean():.4f} max={out.max():.4f}")


if __name__ == "__main__":
    main()
