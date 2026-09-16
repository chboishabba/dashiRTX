#!/usr/bin/env python3
"""Finite pinned Lean/runtime coefficient-byte crosscheck.

This script copies the 17 row-major 64-bit words from the exact Lean source
commit `332ad62036712a34a472234b5f0170f73c1c27de` and compares their 136
little-endian row bytes with the canonical dashiRTX runtime producer receipt at
`7c869e2a44750292653b25833bcc2708df32edd6`.

It is a source/runtime equality receipt only.  It is not a Lean kernel receipt,
does not prove cross-prover definitional equality, and does not identify the
synthetic fixture with production CADO/RSA-260 artifacts.
"""

from __future__ import annotations

import hashlib
import json

from rsa260_bidi_mksol_action_binding import compute_binding_receipt

LEAN_SOURCE_COMMIT = "332ad62036712a34a472234b5f0170f73c1c27de"
RUNTIME_SOURCE_COMMIT = "7c869e2a44750292653b25833bcc2708df32edd6"

LEAN_LAYER_WORDS = (
    0x0000000000000000,
    0x4848484848004848,
    0xBA56B99119685E8D,
    0x2F40C3D97B12CF9E,
    0xFC7F8209F1079D14,
    0xF70F2FEB547CC774,
    0x23245B65E4C7863E,
    0x8284D169ED034976,
    0xE840899D90937AA7,
    0x23C049F9796960AA,
    0x9EE4D6757120F63B,
    0x9AA601DB5E904FC9,
    0xE76E1F0AA90C6A2B,
    0xB1363F03717EE740,
    0xAD82657D5CF422A9,
    0x0000DA6A14A43D21,
    0x0000000000000053,
)


def lean_fixture_bytes() -> bytes:
    return b"".join(word.to_bytes(8, "little") for word in LEAN_LAYER_WORDS)


def crosscheck() -> dict:
    runtime = compute_binding_receipt()
    lean_bytes = lean_fixture_bytes()
    runtime_bytes = bytes.fromhex(runtime["coefficient_row_bytes_hex"])
    equal = lean_bytes == runtime_bytes
    digest = hashlib.sha256(lean_bytes).hexdigest()

    return {
        "schema": "rsa260-bidi-lean-coefficient-fixture-crosscheck-v1",
        "lean_source_commit": LEAN_SOURCE_COMMIT,
        "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
        "byte_count": len(lean_bytes),
        "bytes_sha256": digest,
        "runtime_bytes_equal_pinned_lean_fixture_bytes": equal,
        "runtime_bytes_sha256": runtime["coefficient_row_bytes_sha256"],
        "boundary": {
            "lean_kernel_receipt": False,
            "cross_prover_definitional_equality": False,
            "production_cado_identity": False,
        },
    }


def main() -> None:
    print(json.dumps(crosscheck(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
