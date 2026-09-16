#!/usr/bin/env python3
"""Regression for the synthetic RSA-260 mksol/Krylov action binding receipt.

The production script must expose a deterministic baseline receipt retaining
shape-bound digests for A, the identity permutation, V, recovered F, the Krylov
family K_i, and the resulting sum_i K_i F_i action.  It must also verify the
Krylov recurrence and equality of stored-vs-streaming action evaluation.

This is synthetic/runtime evidence only; it is not exact CADO mksol semantics.
"""

from rsa260_bidi_mksol_action_binding import compute_binding_receipt


def test_baseline_binding_receipt() -> None:
    r = compute_binding_receipt()
    assert r["degree"] == 17
    assert r["coefficient_family_shape"] == [17, 8, 8]
    assert r["seed_block_shape"] == [924, 8]
    assert r["krylov_block_shape"] == [924, 8]
    assert r["action_shape"] == [924, 8]
    assert r["krylov_recurrence_all_equal"] is True
    assert r["action_stored_equals_streaming"] is True
    assert r["operator_linearity_spotcheck"] is True
    assert r["exact_cado_mksol_semantics"] is False
    assert r["boundary"]["runtime_apply_B_is_formal_Lean_M"] is False
    assert r["boundary"]["production_RSA260"] is False


if __name__ == "__main__":
    test_baseline_binding_receipt()
    print("ok")
