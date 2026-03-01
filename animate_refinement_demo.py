#!/usr/bin/env python3
"""
animate_refinement_demo.py

Animated quadtree refinement demo with metrics.
Generates a GIF showing where refinement focuses and how error quantiles evolve.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

import quadtree_ultrametric_renderer as qtr


def block_mean(integ, x0, y0, sz):
    x1 = x0 + sz - 1
    y1 = y0 + sz - 1
    total = integ[y1, x1]
    if x0 > 0:
        total -= integ[y1, x0 - 1]
    if y0 > 0:
        total -= integ[y0 - 1, x1]
    if x0 > 0 and y0 > 0:
        total += integ[y0 - 1, x0 - 1]
    return total / (sz * sz)

def synth_moving_fields(N, center, base_noise):
    yy, xx = np.mgrid[0:N, 0:N]
    cx, cy = center
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.45 * N)
    band = np.abs(r - 0.55) < 0.05

    S = np.zeros((N, N), dtype=np.int8)
    S[(r >= 0.55) & band] = +1
    S[(r < 0.55) & band] = -1

    noise_e, noise_I, noise_tex = base_noise
    e = 0.12 * band.astype(np.float32) + 0.03 * noise_e
    I = (0.4 * band.astype(np.float32) + 0.6 * noise_I).astype(np.float32)
    texture = (0.4 * band.astype(np.float32) + 0.6 * noise_tex).astype(np.float32)
    texture = texture / (texture.max() + 1e-9)
    m = np.clip(0.15 + 0.9 * band.astype(np.float32) + 0.1 * noise_tex, 0, 1)
    return e, S, I, texture, m

def dilate_mask(mask, steps=1):
    m = mask.copy()
    for _ in range(steps):
        p = np.pad(m, 1, mode="edge")
        m = np.maximum.reduce([
            p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:],
            p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:],
            p[2:, 0:-2], p[2:, 1:-1], p[2:, 2:],
        ])
    return m


def refine_topk(root, err, S, I, topk=4, min_size=4, frontier_weight=2.0, frontier_min=0.02, frontier_dilate=2, var_weight=2.0, refine_threshold=None):
    frontier = (np.abs(S) > 0).astype(np.float32)
    frontier_d = dilate_mask(frontier, steps=frontier_dilate)
    err_focus = err * (0.1 + frontier_weight * frontier_d)
    integ = err_focus.cumsum(axis=0).cumsum(axis=1)
    integ_f = frontier_d.cumsum(axis=0).cumsum(axis=1)
    integ_e = err.cumsum(axis=0).cumsum(axis=1)
    integ_e2 = (err * err).cumsum(axis=0).cumsum(axis=1)
    leaves = qtr.gather_leaves(root)
    scored = []
    for n in leaves:
        if n.size <= min_size:
            continue
        mean_err = block_mean(integ, n.x0, n.y0, n.size)
        mean_frontier = block_mean(integ_f, n.x0, n.y0, n.size)
        mean_e = block_mean(integ_e, n.x0, n.y0, n.size)
        mean_e2 = block_mean(integ_e2, n.x0, n.y0, n.size)
        var_e = max(0.0, mean_e2 - mean_e * mean_e)
        if mean_frontier < frontier_min:
            continue
        score = mean_err * (1.0 + var_weight * var_e)
        scored.append((score, mean_err, n))
    scored.sort(key=lambda x: x[0], reverse=True)

    refined = []
    picked = []
    picked_ids = set()
    for _, _, n in scored[:topk]:
        nid = id(n)
        if nid not in picked_ids:
            picked.append(n)
            picked_ids.add(nid)

    if refine_threshold is not None:
        for _, mean_err, n in scored:
            if mean_err >= refine_threshold:
                nid = id(n)
                if nid not in picked_ids:
                    picked.append(n)
                    picked_ids.add(nid)

    for n in picked:
        if n.children is None and n.size > min_size:
            n.children = qtr.subdivide(n)
            for c in n.children:
                qtr.node_stats(c, err, S, I)
            refined.append(n)
    return refined


def update_leaf_stats(root, e, S, I):
    for n in qtr.gather_leaves(root):
        qtr.node_stats(n, e, S, I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--bounce_cost", type=int, default=10)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="outputs_pda")
    ap.add_argument("--frontier_weight", type=float, default=2.0)
    ap.add_argument("--frontier_min", type=float, default=0.02)
    ap.add_argument("--frontier_dilate", type=int, default=2)
    ap.add_argument("--var_weight", type=float, default=2.0)
    ap.add_argument("--refine_threshold", type=float, default=None, help="refine all leaves with mean error >= threshold")
    ap.add_argument("--show_frontier", action="store_true")
    ap.add_argument("--view", type=str, default="error", choices=["error", "result", "pixel", "diff"])
    ap.add_argument("--moving_center", action="store_true", help="move ring center over time")
    ap.add_argument("--move_amp", type=float, default=0.12, help="center motion amplitude as fraction of N")
    ap.add_argument("--move_period", type=float, default=20.0, help="motion period in steps")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    base_noise = (
        rng.random((args.N, args.N), dtype=np.float32),
        rng.random((args.N, args.N), dtype=np.float32),
        rng.random((args.N, args.N), dtype=np.float32),
    )

    center0 = (0.52 * args.N, 0.48 * args.N)
    e, S, I, texture, m = synth_moving_fields(args.N, center0, base_noise)
    e = e * args.sparsity
    I = I * args.sparsity
    m = m * args.sparsity
    refine_scale = min(8.0, 1.0 / max(args.sparsity, 1e-3))

    def compute_pix(Iv, Sv, tex):
        f = (np.abs(Sv) > 0).astype(np.float32)
        k = 0.25 + 1.5 * (0.6 * Iv + 0.4 * f) * (0.5 + 0.5 * np.sin(1.2))
        for _ in range(args.bounce_cost):
            k = np.sin(k * 1.13 + 0.1)
        return k * tex, f

    pix, frontier = compute_pix(I, S, texture)

    root = qtr.build_quadtree(e, S, I, m, min_size=4, refine_scale=refine_scale)

    q25s, q50s, q75s, q99s, maes = [], [], [], [], []
    frames = []

    for step in range(args.steps + 1):
        if args.moving_center and step > 0:
            dx = args.move_amp * args.N * np.sin(2.0 * np.pi * step / args.move_period)
            dy = args.move_amp * args.N * np.cos(2.0 * np.pi * step / args.move_period)
            center = (center0[0] + dx, center0[1] + dy)
            e, S, I, texture, m = synth_moving_fields(args.N, center, base_noise)
            update_leaf_stats(root, e, S, I)
            pix, frontier = compute_pix(I, S, texture)

        qt = qtr.quadtree_render_numba_full(root, texture, view_param=1.2, bounce_cost=args.bounce_cost)
        err = np.abs(pix - qt)
        q25, q50, q75, q99 = np.quantile(err.reshape(-1), [0.25, 0.50, 0.75, 0.99])
        mae = float(np.mean(err))
        q25s.append(q25); q50s.append(q50); q75s.append(q75); q99s.append(q99); maes.append(mae)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        ax0, ax1 = axes

        if args.view in ("error", "diff"):
            im = ax0.imshow(err, cmap="magma")
            ax0.set_title(f"Error map (step {step})")
        elif args.view == "result":
            im = ax0.imshow(qt, cmap="viridis")
            ax0.set_title(f"Quadtree render (step {step})")
        else:
            im = ax0.imshow(pix, cmap="viridis")
            ax0.set_title(f"Pixel render (step {step})")
        ax0.axis("off")
        fig.colorbar(im, ax=ax0, fraction=0.046)

        if args.show_frontier:
            ax0.contour(frontier, levels=[0.5], colors="lime", linewidths=0.6, alpha=0.7)

        # overlay refined blocks (current leaves)
        for n in qtr.gather_leaves(root):
            if n.size <= 4:
                continue
            rect = patches.Rectangle((n.x0, n.y0), n.size, n.size, linewidth=0.6,
                                     edgecolor="cyan", facecolor="none", alpha=0.2)
            ax0.add_patch(rect)

        ax1.plot(q99s, label="q99")
        ax1.plot(q75s, label="q75")
        ax1.plot(q50s, label="q50")
        ax1.plot(q25s, label="q25")
        ax1.plot(maes, label="mae")
        ax1.set_xlabel("refine step")
        ax1.set_title("Error metrics")
        ax1.legend(loc="upper right", fontsize=8)

        frame_path = outdir / f"refine_frame_{step:02d}.png"
        fig.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)
        frames.append(imageio.v2.imread(frame_path))

        if step < args.steps:
            refine_topk(
                root, err, S, I,
                topk=args.topk, min_size=4,
                frontier_weight=args.frontier_weight,
                frontier_min=args.frontier_min,
                frontier_dilate=args.frontier_dilate,
                var_weight=args.var_weight,
                refine_threshold=args.refine_threshold
            )

    gif_path = outdir / "refinement_demo.gif"
    imageio.mimsave(gif_path, frames, fps=args.fps)
    print(f"[done] wrote {gif_path}")


if __name__ == "__main__":
    main()
