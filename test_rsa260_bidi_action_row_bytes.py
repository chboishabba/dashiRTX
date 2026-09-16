#!/usr/bin/env python3
"""Regression for the canonical 924-byte row serialization of baseline action."""

from rsa260_bidi_mksol_action_binding import compute_binding_receipt

EXPECTED_ACTION_ROW_BYTES_SHA256 = (
    "51c708216fd1f0b73d8d18d856fb60197488c107df20d8ab6481c27179ad69e1"
)


def test_action_row_bytes() -> None:
    r = compute_binding_receipt()
    assert r["action_row_bytes_count"] == 924
    assert r["action_row_bytes_sha256"] == EXPECTED_ACTION_ROW_BYTES_SHA256
    assert r["action_row_bytes_first16_hex"] == "9f7077d364b3cb29243668b433c8fd44"
    assert r["action_row_bytes_last16_hex"] == "48605d7e71aceb12698e69509cce128e"
    assert r["boundary"]["runtime_action_is_formal_Lean_krylovCoefficientFamilyAction"] is False


if __name__ == "__main__":
    test_action_row_bytes()
    print("ok")
