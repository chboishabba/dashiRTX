#!/usr/bin/env python3
"""Regression for the synthetic RSA-260 mksol/Krylov action binding receipt.

The production script must expose a deterministic baseline receipt retaining
shape-bound digests for A, the identity permutation, V, recovered F, the Krylov
family K_i, and the resulting sum_i K_i F_i action.  It must also verify the
Krylov recurrence, equality of stored-vs-streaming action evaluation, exact
entrywise equality between the original runtime `build_A()` constructor and the
modular-distance extensional formula intended for the Lean mirror, and the exact
136 row-packed bytes of the recovered 17x8x8 coefficient family.

This is synthetic/runtime evidence only; it is not exact CADO mksol semantics.
"""

from rsa260_bidi_mksol_action_binding import compute_binding_receipt

EXPECTED_F_ROW_BYTES_HEX = (
    "00000000000000004848004848484848"
    "8d5e681991b956ba9ecf127bd9c3402f"
    "149d07f109827ffc74c77c54eb2f0ff7"
    "3e86c7e4655b2423764903ed69d18482"
    "a77a93909d8940e8aa606979f949c023"
    "3bf6207175d6e49ec94f905edb01a69a"
    "2b6a0ca90a1f6ee740e77e71033f36b1"
    "a922f45c7d6582ad213da4146ada0000"
    "5300000000000000"
)
EXPECTED_F_ROW_BYTES_SHA256 = (
    "2545d4185bffa89562aebf7405a1db81443a7d49f64aeb83dc564a940ce13e79"
)


def test_baseline_binding_receipt() -> None:
    r = compute_binding_receipt()
    assert r["degree"] == 17
    assert r["coefficient_family_shape"] == [17, 8, 8]
    assert r["coefficient_row_bytes_count"] == 136
    assert r["coefficient_row_bytes_hex"] == EXPECTED_F_ROW_BYTES_HEX
    assert r["coefficient_row_bytes_sha256"] == EXPECTED_F_ROW_BYTES_SHA256
    assert r["seed_block_shape"] == [924, 8]
    assert r["krylov_block_shape"] == [924, 8]
    assert r["action_shape"] == [924, 8]
    assert r["krylov_recurrence_all_equal"] is True
    assert r["action_stored_equals_streaming"] is True
    assert r["operator_linearity_spotcheck"] is True
    assert r["lean_style_A_matches_runtime_A"] is True
    assert r["lean_style_A_mismatch_count"] == 0
    assert r["lean_style_A_sha256"] == r["A_sha256"]
    assert r["exact_cado_mksol_semantics"] is False
    assert r["boundary"]["runtime_apply_B_is_formal_Lean_M"] is False
    assert r["boundary"]["production_RSA260"] is False


if __name__ == "__main__":
    test_baseline_binding_receipt()
    print("ok")
