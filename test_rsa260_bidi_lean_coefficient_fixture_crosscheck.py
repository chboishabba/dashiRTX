#!/usr/bin/env python3
"""Regression for the pinned Lean/runtime coefficient-byte crosscheck."""

from rsa260_bidi_lean_coefficient_fixture_crosscheck import crosscheck


def test_crosscheck() -> None:
    r = crosscheck()
    assert r["lean_source_commit"] == "332ad62036712a34a472234b5f0170f73c1c27de"
    assert r["runtime_source_commit"] == "7c869e2a44750292653b25833bcc2708df32edd6"
    assert r["byte_count"] == 136
    assert r["runtime_bytes_equal_pinned_lean_fixture_bytes"] is True
    assert r["bytes_sha256"] == "2545d4185bffa89562aebf7405a1db81443a7d49f64aeb83dc564a940ce13e79"
    assert r["boundary"]["lean_kernel_receipt"] is False
    assert r["boundary"]["production_cado_identity"] is False


if __name__ == "__main__":
    test_crosscheck()
    print("ok")
