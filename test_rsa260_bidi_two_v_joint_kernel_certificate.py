#!/usr/bin/env python3
"""Regression for the synthetic two-V joint-kernel certificate.

The producer must replace the old scalar `rank = 136` observation with a finite
proof certificate: 136 selected rows from the stacked Krylov-column matrix and
an explicit 136x136 GF(2) inverse of that square minor.

This remains synthetic evidence only.  It is not CADO production semantics and
it does not by itself constitute a Lean kernel receipt.
"""

from rsa260_bidi_two_v_joint_kernel_certificate import compute_certificate


def test_two_v_joint_kernel_certificate() -> None:
    r = compute_certificate()
    assert r["schema"] == "rsa260-bidi-two-v-joint-kernel-certificate-v1"
    assert r["degree"] == 17
    assert r["coefficient_column_bits"] == 136
    assert r["baseline_krylov_column_rank"] == 128
    assert r["stacked_krylov_column_rank"] == 136
    assert len(r["selected_row_indices"]) == 136
    assert r["selected_square_shape"] == [136, 136]
    assert r["selected_square_rank"] == 136
    assert r["inverse_row_bytes_count"] == 2312
    assert r["left_inverse_identity"] is True
    assert r["right_inverse_identity"] is True
    assert r["two_v_intersection_kernel_zero"] is True
    assert r["baseline_seed"] == "BASEY"
    assert r["second_seed_formula"] == "BASEY xor (1 << 40)"
    assert r["exact_cado_mksol_context_family"] is False
    assert r["production_rsa260"] is False


if __name__ == "__main__":
    test_two_v_joint_kernel_certificate()
    print("ok")
