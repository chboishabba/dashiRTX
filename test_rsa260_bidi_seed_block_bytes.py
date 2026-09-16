#!/usr/bin/env python3
"""Regression for the canonical 924-byte row serialization of baseline V."""

from rsa260_bidi_mksol_action_binding import compute_binding_receipt

EXPECTED_SEED_ROW_BYTES_SHA256 = (
    "3cd881999d687ed98b37811f85a776fd3d11b03a381c898aab6f78b97e1701fe"
)


def test_seed_row_bytes() -> None:
    r = compute_binding_receipt()
    assert r["seed_row_bytes_count"] == 924
    assert r["seed_row_bytes_sha256"] == EXPECTED_SEED_ROW_BYTES_SHA256
    assert r["seed_row_bytes_first16_hex"] == "722cc23d374bf8d63ddf79b1b751abd8"
    assert r["seed_row_bytes_last16_hex"] == "828ca4751131b2ff6c16b6b56c89b425"
    assert r["boundary"]["runtime_V_is_formal_Lean_V"] is False


if __name__ == "__main__":
    test_seed_row_bytes()
    print("ok")
