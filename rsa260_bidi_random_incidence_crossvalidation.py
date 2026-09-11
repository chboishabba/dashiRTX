#!/usr/bin/env python3
"""
RSA-260 bidi experiment: randomized fine-incidence carrier cross-validation.

Attribution / source coordinates:
- Eric Lu, "Factoring RSA-260", Cognition, 2026-09-09.
  Canonical link: https://cognition.com/blog/factoring-rsa-260
  Role: primary first-party execution source for the RSA-260 LA envelope.
- Don Coppersmith, "Solving homogeneous linear equations over GF(2) via block
  Wiedemann algorithm", Mathematics of Computation 62(205), 1994.
  DOI: 10.1090/S0025-5718-1994-1192970-7
  Role: primary algorithm-lineage source for block Wiedemann.
- Douglas H. Wiedemann, "Solving sparse linear equations over finite fields",
  IEEE Transactions on Information Theory 32(1), 1986.
  DOI: 10.1109/TIT.1986.1057137
  Role: upstream sparse finite-field linear-algebra source.

Semantic coordinates only (not proof/authority):
- GNFS Q140770
- sparse matrix Q1050404
- finite field Q603880
- RSA cryptosystem Q181551

This experiment does NOT claim historical RSA-260 matrix identity.  It tests
whether the same coarse shadow contract admits very different generator
geometry under changed fine incidence.
"""
from pathlib import Path
import json
import time
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_candidate_robustness.py")
source = BASE.read_text()
prefix = source.split("adapters = {", 1)[0]
env = {}
exec(compile(prefix, str(BASE) + ":prefix", "exec"), env)

ROWS, COLS = env["ROWS"], env["COLS"]
BASEX, BASEY = env["BASEX"], env["BASEY"]

def gf2_rank(M):
    M = M.copy()
    nr, nc = M.shape
    rank = 0
    for c in range(nc):
        nz = np.flatnonzero(M[rank:, c])
        if nz.size == 0:
            continue
        p = rank + nz[0]
        if p != rank:
            M[[rank, p]] = M[[p, rank]]
        nz = np.flatnonzero(M[:, c])
        nz = nz[nz != rank]
        if nz.size:
            M[nz] ^= M[rank]
        rank += 1
        if rank == nr:
            break
    return rank

def build_random_exact_degree_A(seed):
    rng = np.random.default_rng(seed)
    A = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        degree = 151 if r < 6 else 150
        A[r, rng.choice(COLS, size=degree, replace=False)] = 1
    return A

def evaluate_at_bound(A, shift, projection_index, max_degree):
    env["A"] = A
    env["MAXD"] = max_degree
    perm = (np.arange(COLS) + shift) % COLS
    apply = env["make_apply_B"](perm)
    return env["evaluate"](
        apply,
        BASEX ^ ((projection_index + 1) << 48),
        BASEY ^ ((projection_index + 1) << 40),
    )

def main():
    carrier_seeds = [260001, 260003, 260007, 260011]
    shifts = [29, 31]
    original_A = env["A"]
    rows = []
    started = time.perf_counter()
    try:
        for carrier_seed in carrier_seeds:
            A = build_random_exact_degree_A(carrier_seed)
            rank = gf2_rank(A)
            item = {
                "carrier_seed": carrier_seed,
                "rank": rank,
                "left_nullity": ROWS - rank,
                "degree151_rows": int((A.sum(axis=1) == 151).sum()),
                "degree150_rows": int((A.sum(axis=1) == 150).sum()),
                "adapters": [],
            }
            for shift in shifts:
                low = [evaluate_at_bound(A, shift, 0, 40)]
                escalated = evaluate_at_bound(A, shift, 0, 80)
                item["adapters"].append({
                    "adapter": f"rotate{shift}",
                    "degree_le_40_passes": sum(bool(r.get("passed")) for r in low),
                    "degree_le_40_results": low,
                    "escalated_max_degree": 80,
                    "escalated_result": escalated,
                })
            rows.append(item)
    finally:
        env["A"] = original_A

    report = {
        "carrier_shape": [ROWS, COLS],
        "row_excess": ROWS - COLS,
        "coarse_contract": {
            "full_rank_required": True,
            "degree151_rows": 6,
            "degree150_rows": 918,
        },
        "randomized_carriers": rows,
        "summary": {
            "carriers_tested": len(rows),
            "low_degree_runs_tested": len(rows) * len(shifts),
            "low_degree_passes": sum(
                a["degree_le_40_passes"]
                for r in rows for a in r["adapters"]
            ),
            "escalated_runs_tested": len(rows) * len(shifts),
            "escalated_passes": sum(
                bool(a["escalated_result"].get("passed"))
                for r in rows for a in r["adapters"]
            ),
            "escalated_degrees": [
                a["escalated_result"].get("degree")
                for r in rows for a in r["adapters"]
            ],
        },
        "firewalls": {
            "same_coarse_contract_implies_low_degree_generator": False,
            "no_degree_le_40_implies_no_generator": False,
            "consumer_success_implies_historical_matrix_identity": False,
            "qid_or_doi_imports_proof": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
