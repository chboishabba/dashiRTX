#!/usr/bin/env python3
"""
benchmark_report.py

Benchmark accuracy + compute for:
1) PDA/MDL learner old vs new (metric regularization + optional ternary penalty)
2) Pixel-space rendering vs quadtree ultrametric rendering
"""

from __future__ import annotations
import time
import numpy as np

import pda_mdl_light_transport_test as pda
import quadtree_ultrametric_renderer as qtr


def time_block(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def bench_pda(H=40, W=40, pairs=14, seed=0):
    rng = np.random.default_rng(seed)
    f = 45.0
    spheres = [
        pda.Sphere(center=np.array([0.0, 0.0, 3.2], dtype=np.float32),
                   radius=1.10,
                   color=np.array([0.72, 0.72, 0.72], dtype=np.float32),
                   emissive=0.0),
        pda.Sphere(center=np.array([1.35, 0.05, 3.05], dtype=np.float32),
                   radius=0.42,
                   color=np.array([1.0, 0.1, 0.1], dtype=np.float32),
                   emissive=1.2),
    ]

    base_pos = (0.0, 0.0, 0.0)
    base_yaw = 0.0
    h_prev = np.zeros((H, W), dtype=np.float32)
    focus_center = (int(0.55 * H), int(0.78 * W))

    pairs_list = []
    for _ in range(pairs):
        dx = rng.uniform(-0.10, 0.10)
        dyaw = rng.uniform(-0.10, 0.10)
        pair = pda.build_pair(
            H, W, f, spheres,
            base_pos, base_yaw, dx, dyaw,
            focus_center=focus_center,
            h_prev=h_prev,
            beta=0.90
        )
        h_prev = pair["h_next"]
        pairs_list.append(pair)

    train_pairs = pairs_list[: max(2, int(0.7 * len(pairs_list)))]
    val_pair = pairs_list[-1]

    def stack_frames(pairs):
        Xs, errs, Ss, Is, hs = [], [], [], [], []
        for p in pairs:
            Xs.append(p["X"])
            errs.append(p["err"].reshape(-1))
            Ss.append(p["S0"].reshape(-1))
            Is.append(p["I0"].reshape(-1))
            hs.append(p["h"].reshape(-1))
        return Xs, errs, Ss, Is, hs

    Xtr_list, err_tr_list, S_tr_list, I_tr_list, h_tr_list = stack_frames(train_pairs)
    Xva = val_pair["X"]
    err_va = val_pair["err"].reshape(-1)
    S_va = val_pair["S0"].reshape(-1)
    I_va = val_pair["I0"].reshape(-1)
    h_va = val_pair["h"].reshape(-1)

    alpha = 25.0
    kappa = 8.0
    e_tr_list = [
        err_tr_list[i] * (1.0 + alpha * I_tr_list[i]) * (1.0 + kappa * h_tr_list[i])
        for i in range(len(err_tr_list))
    ]
    e_va = err_va * (1.0 + alpha * I_va) * (1.0 + kappa * h_va)

    def train_mlp_multiframe(X_list, e_list, S_list, I_list, epochs, lr, seed,
                             lam, eta_tv, gamma_metric, gamma_tern):
        model = pda.TinyMLP(X_list[0].shape[1], Hh=16, seed=seed)
        for _ in range(epochs):
            dW1 = np.zeros_like(model.W1)
            db1 = np.zeros_like(model.b1)
            dW2 = np.zeros_like(model.W2)
            db2 = np.zeros_like(model.b2)
            for X, e_eff, S0, I0 in zip(X_list, e_list, S_list, I_list):
                logits, h = model.forward(X)
                loss, dlog, _ = pda.mdl_loss_and_grad_logits(
                    logits, e_eff, S0, I0, H, W,
                    lam=lam, eta_tv=eta_tv,
                    gamma_metric=gamma_metric, gamma_tern=gamma_tern
                )
                dh = (dlog[:, None] @ model.W2.T) * (1.0 - h * h)
                dW2 += (h.T @ dlog[:, None]).astype(np.float32) / X.shape[0]
                db2 += np.mean(dlog).astype(np.float32)
                dW1 += (X.T @ dh).astype(np.float32) / X.shape[0]
                db1 += np.mean(dh, axis=0).astype(np.float32)
            model.step((dW1, db1, dW2, db2), lr)
        logits, _ = model.forward(X_list[-1])
        loss, _, m = pda.mdl_loss_and_grad_logits(
            logits, e_list[-1], S_list[-1], I_list[-1], H, W,
            lam=lam, eta_tv=eta_tv,
            gamma_metric=gamma_metric, gamma_tern=gamma_tern
        )
        return model, m, loss

    # OLD: no metric reg, no ternary
    def train_old():
        return train_mlp_multiframe(
            Xtr_list, e_tr_list, S_tr_list, I_tr_list,
            epochs=160, lr=0.08, seed=seed + 1,
            lam=2.0, eta_tv=0.02,
            gamma_metric=0.0, gamma_tern=0.0
        )

    # NEW: metric reg + ternary penalty
    def train_new():
        return train_mlp_multiframe(
            Xtr_list, e_tr_list, S_tr_list, I_tr_list,
            epochs=160, lr=0.08, seed=seed + 1,
            lam=2.0, eta_tv=0.02,
            gamma_metric=0.03, gamma_tern=0.02
        )

    (old_model, old_m, old_loss), t_old = time_block(train_old)
    (new_model, new_m, new_loss), t_new = time_block(train_new)

    # Validation reconstruction error using heldout mask
    logits_va_old, _ = old_model.forward(Xva)
    logits_va_new, _ = new_model.forward(Xva)
    m_va_old = pda.logits_to_mask(logits_va_old)
    m_va_new = pda.logits_to_mask(logits_va_new)

    ths_old, curve_old = pda.mdl_proxy_threshold_sweep(m_va_old, err_va)
    ths_new, curve_new = pda.mdl_proxy_threshold_sweep(m_va_new, err_va)
    th_old = ths_old[np.argmin(curve_old)]
    th_new = ths_new[np.argmin(curve_new)]

    # apply to a single val frame to compute recon error
    vis = val_pair
    l1w = vis["l1_w"].reshape(-1)
    l1 = vis["l1"].reshape(-1)

    logits_vis_old, _ = old_model.forward(vis["X"])
    logits_vis_new, _ = new_model.forward(vis["X"])
    m_vis_old = pda.logits_to_mask(logits_vis_old)
    m_vis_new = pda.logits_to_mask(logits_vis_new)

    chosen_old = (m_vis_old >= th_old).astype(np.float32)
    chosen_new = (m_vis_new >= th_new).astype(np.float32)

    pred_old = l1w.copy()
    pred_new = l1w.copy()
    pred_old[chosen_old > 0] = l1[chosen_old > 0]
    pred_new[chosen_new > 0] = l1[chosen_new > 0]

    mae_old = float(np.mean(np.abs(pred_old - l1)))
    mae_new = float(np.mean(np.abs(pred_new - l1)))

    return {
        "old": {
            "train_loss": float(old_loss),
            "train_sec": t_old,
            "val_mdl_min": float(np.min(curve_old)),
            "val_mae": mae_old,
            "active_frac": float(np.mean(chosen_old))
        },
        "new": {
            "train_loss": float(new_loss),
            "train_sec": t_new,
            "val_mdl_min": float(np.min(curve_new)),
            "val_mae": mae_new,
            "active_frac": float(np.mean(chosen_new))
        }
    }


def bench_quadtree(N=96, seed=0, sparsity=1.0, use_numba=False, bounce_cost=0):
    e, S, I, texture, m = qtr.synth_fields(N, N, seed=seed)
    e = e * sparsity
    I = I * sparsity
    m = m * sparsity
    refine_scale = min(8.0, 1.0 / max(sparsity, 1e-3))

    def pixel_render():
        frontier = (np.abs(S) > 0).astype(np.float32)
        kernel = 0.25 + 1.5 * (0.6 * I + 0.4 * frontier) * (0.5 + 0.5 * np.sin(1.2))
        for _ in range(bounce_cost):
            kernel = np.sin(kernel * 1.13 + 0.1)
        return kernel * texture

    def quadtree_render():
        root = qtr.build_quadtree(e, S, I, m, min_size=4, refine_scale=refine_scale)
        out = qtr.quadtree_render(root, texture, view_param=1.2, bounce_cost=bounce_cost)
        return out, root

    def quadtree_render_vec():
        root = qtr.build_quadtree(e, S, I, m, min_size=4, refine_scale=refine_scale)
        out = qtr.quadtree_render_vectorized(root, texture, view_param=1.2, bounce_cost=bounce_cost)
        return out, root

    def quadtree_render_numba():
        root = qtr.build_quadtree(e, S, I, m, min_size=4, refine_scale=refine_scale)
        out = qtr.quadtree_render_numba(root, texture, view_param=1.2, bounce_cost=bounce_cost)
        return out, root

    def quadtree_render_numba_full():
        root = qtr.build_quadtree(e, S, I, m, min_size=4, refine_scale=refine_scale)
        out = qtr.quadtree_render_numba_full(root, texture, view_param=1.2, bounce_cost=bounce_cost)
        return out, root

    pix_out, t_pix = time_block(pixel_render)
    (qt_out, root), t_qt = time_block(quadtree_render)
    (qt_out_vec, root_vec), t_qt_vec = time_block(quadtree_render_vec)
    qt_num = None
    if use_numba:
        try:
            # warm up JIT (compile) without timing
            _ = qtr.quadtree_render_numba_full(qtr.build_quadtree(e, S, I, m, min_size=4, refine_scale=refine_scale), texture, view_param=1.2, bounce_cost=bounce_cost)
            (qt_out_num, root_num), t_qt_num = time_block(quadtree_render_numba_full)
            qt_num = (qt_out_num, root_num, t_qt_num)
        except Exception:
            qt_num = None

    leaves = len(qtr.gather_leaves(root))
    mae = float(np.mean(np.abs(pix_out - qt_out)))
    mae_vec = float(np.mean(np.abs(pix_out - qt_out_vec)))

    out = {
        "pixel": {"sec": t_pix},
        "quadtree": {"sec": t_qt, "leaves": leaves, "mae": mae},
        "quadtree_vec": {
            "sec": t_qt_vec,
            "leaves": len(qtr.gather_leaves(root_vec)),
            "mae": mae_vec
        }
    }
    if qt_num is not None:
        qt_out_num, root_num, t_qt_num = qt_num
        out["quadtree_numba"] = {
            "sec": t_qt_num,
            "leaves": len(qtr.gather_leaves(root_num)),
            "mae": float(np.mean(np.abs(pix_out - qt_out_num)))
        }
    return out


def bench_quadtree_sizes(sizes=(512, 1024, 2048), seed=0, bounce_cost=0):
    rows = []
    for N in sizes:
        res = bench_quadtree(N=N, seed=seed, sparsity=1.0, use_numba=True, bounce_cost=bounce_cost)
        rows.append((N, res))
    return rows


def bench_quadtree_sweep(N=96, seed=0, sparsities=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02), bounce_cost=0):
    rows = []
    for s in sparsities:
        res = bench_quadtree(N=N, seed=seed, sparsity=s, use_numba=True, bounce_cost=bounce_cost)
        rows.append((s, res))
    return rows


def bench_quality_quantiles(N=1024, seed=0, sparsities=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02), bounce_cost=0):
    rows = []
    for s in sparsities:
        e, S, I, texture, m = qtr.synth_fields(N, N, seed=seed)
        e = e * s
        I = I * s
        m = m * s
        refine_scale = min(8.0, 1.0 / max(s, 1e-3))

        frontier = (np.abs(S) > 0).astype(np.float32)
        kernel = 0.25 + 1.5 * (0.6 * I + 0.4 * frontier) * (0.5 + 0.5 * np.sin(1.2))
        for _ in range(bounce_cost):
            kernel = np.sin(kernel * 1.13 + 0.1)
        pix = kernel * texture

        root = qtr.build_quadtree(e, S, I, m, min_size=4, refine_scale=refine_scale)
        qt = qtr.quadtree_render_numba_full(root, texture, view_param=1.2, bounce_cost=bounce_cost)

        err = np.abs(pix - qt)
        q25, q50, q75, q99 = np.quantile(err.reshape(-1), [0.25, 0.50, 0.75, 0.99])

        # Optional refinement: split the worst-error leaves and re-render
        refine_steps = 2
        topk = 4
        min_size = 4

        integ = err.cumsum(axis=0).cumsum(axis=1)

        def block_mean(x0, y0, sz):
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

        for _ in range(refine_steps):
            leaves = qtr.gather_leaves(root)
            scored = []
            for n in leaves:
                if n.size <= min_size:
                    continue
                scored.append((block_mean(n.x0, n.y0, n.size), n))
            scored.sort(key=lambda x: x[0], reverse=True)
            changed = 0
            for _, n in scored[:topk]:
                if n.children is None and n.size > min_size:
                    n.children = qtr.subdivide(n)
                    for c in n.children:
                        qtr.node_stats(c, e, S, I)
                    changed += 1
            if changed == 0:
                break
            qt = qtr.quadtree_render_numba_full(root, texture, view_param=1.2, bounce_cost=bounce_cost)
            err = np.abs(pix - qt)
            integ = err.cumsum(axis=0).cumsum(axis=1)

        q25r, q50r, q75r, q99r = np.quantile(err.reshape(-1), [0.25, 0.50, 0.75, 0.99])
        rows.append((
            s,
            len(qtr.gather_leaves(root)),
            float(q25), float(q50), float(q75), float(q99),
            float(q25r), float(q50r), float(q75r), float(q99r),
        ))
    return rows


def main():
    pda_res = bench_pda()
    bounce_cost = 30
    qt_res = bench_quadtree(bounce_cost=bounce_cost)
    qt_sweep = bench_quadtree_sweep(bounce_cost=bounce_cost)
    qt_sizes = bench_quadtree_sizes(bounce_cost=bounce_cost)
    qt_quant = bench_quality_quantiles(bounce_cost=bounce_cost)

    print("== PDA/MDL Benchmark (old vs new) ==")
    for k, v in pda_res.items():
        print(f"[{k}] train_loss={v['train_loss']:.3f}  train_sec={v['train_sec']:.3f}  val_mdl_min={v['val_mdl_min']:.3f}  val_mae={v['val_mae']:.4f}  active_frac={v['active_frac']:.3f}")

    print(f"\n== Rendering Benchmark (pixel vs quadtree, bounce_cost={bounce_cost}) ==")
    print(f"[pixel] sec={qt_res['pixel']['sec']:.4f}")
    print(f"[quadtree] sec={qt_res['quadtree']['sec']:.4f}  leaves={qt_res['quadtree']['leaves']}  mae={qt_res['quadtree']['mae']:.5f}")
    print(f"[quadtree_vec] sec={qt_res['quadtree_vec']['sec']:.4f}  leaves={qt_res['quadtree_vec']['leaves']}  mae={qt_res['quadtree_vec']['mae']:.5f}")
    if "quadtree_numba" in qt_res:
        print(f"[quadtree_numba] sec={qt_res['quadtree_numba']['sec']:.4f}  leaves={qt_res['quadtree_numba']['leaves']}  mae={qt_res['quadtree_numba']['mae']:.5f}")

    print("\n== Quadtree Sparsity Sweep (pixel vs quadtree) ==")
    for s, res in qt_sweep:
        line = f"[s={s:.2f}] pixel={res['pixel']['sec']:.4f}  qt={res['quadtree']['sec']:.4f}  qt_vec={res['quadtree_vec']['sec']:.4f}"
        if "quadtree_numba" in res:
            line += f"  qt_numba={res['quadtree_numba']['sec']:.4f}"
        line += f"  leaves={res['quadtree']['leaves']}  mae={res['quadtree']['mae']:.4f}"
        print(line)

    print("\n== Quadtree Size Sweep (pixel vs quadtree) ==")
    for N, res in qt_sizes:
        line = f"[N={N}] pixel={res['pixel']['sec']:.4f}  qt={res['quadtree']['sec']:.4f}  qt_vec={res['quadtree_vec']['sec']:.4f}"
        if "quadtree_numba" in res:
            line += f"  qt_numba={res['quadtree_numba']['sec']:.4f}"
        line += f"  leaves={res['quadtree']['leaves']}  mae={res['quadtree']['mae']:.4f}"
        print(line)

    print("\n== Quality Quantiles (abs error vs pixel, N=1024) ==")
    print("format: base -> refined (2 steps, top4 leaves)")
    for s, leaves, q25, q50, q75, q99, q25r, q50r, q75r, q99r in qt_quant:
        print(
            f"[s={s:.2f}] leaves={leaves}  "
            f"q25={q25:.5f}->{q25r:.5f}  q50={q50:.5f}->{q50r:.5f}  "
            f"q75={q75:.5f}->{q75r:.5f}  q99={q99:.5f}->{q99r:.5f}"
        )


if __name__ == "__main__":
    main()
