#!/usr/bin/env python3
"""Executable uint8 oracle for the DASHI three-trit compact codec.

This is the runtime counterpart of:
  DASHI.ComputerScience.TriadicScalarCompactCodecReferenceExact
  DASHI.ComputerScience.TriadicFin27Byte256ABIExact

The ABI is deliberately simple:
  * codes 0..26 are the 27 valid compact states, in the exact Agda stateCode order;
  * codes 27..255 are reserved/invalid;
  * the carrier is numpy.uint8, so this is a real host byte binding rather than
    an unbounded Python integer model.

No performance, SIMD, SWAR, CUDA, ROCm, or device-equivalence claim is made here.
"""

from __future__ import annotations

from itertools import product
from typing import Final

import numpy as np

Trit = int
Triple = tuple[Trit, Trit, Trit]

NEG: Final[Trit] = -1
ZERO: Final[Trit] = 0
POS: Final[Trit] = 1
TRITS: Final[tuple[Trit, Trit, Trit]] = (NEG, ZERO, POS)
VALID_STATE_COUNT: Final[int] = 27

# Exact order from TriadicFin27Byte256ABIExact.stateCode.
STATE_TO_TRIPLE: Final[tuple[Triple, ...]] = (
    (ZERO, ZERO, ZERO),
    (ZERO, ZERO, NEG),
    (ZERO, ZERO, POS),
    (ZERO, NEG, ZERO),
    (ZERO, POS, ZERO),
    (ZERO, NEG, NEG),
    (ZERO, NEG, POS),
    (ZERO, POS, NEG),
    (ZERO, POS, POS),
    (NEG, ZERO, ZERO),
    (POS, ZERO, ZERO),
    (NEG, ZERO, NEG),
    (NEG, ZERO, POS),
    (POS, ZERO, NEG),
    (POS, ZERO, POS),
    (NEG, NEG, ZERO),
    (NEG, POS, ZERO),
    (POS, NEG, ZERO),
    (POS, POS, ZERO),
    (NEG, NEG, NEG),
    (NEG, NEG, POS),
    (NEG, POS, NEG),
    (NEG, POS, POS),
    (POS, NEG, NEG),
    (POS, NEG, POS),
    (POS, POS, NEG),
    (POS, POS, POS),
)

TRIPLE_TO_STATE: Final[dict[Triple, int]] = {
    triple: code for code, triple in enumerate(STATE_TO_TRIPLE)
}


def _check_triple(triple: Triple) -> None:
    if len(triple) != 3 or any(t not in TRITS for t in triple):
        raise ValueError(f"not a balanced-trit triple: {triple!r}")


def encode_u8(triple: Triple) -> np.uint8:
    """Encode one balanced-trit triple to the ABI byte range 0..26."""
    _check_triple(triple)
    return np.uint8(TRIPLE_TO_STATE[triple])


def decode_u8(code: np.uint8 | int) -> Triple:
    """Decode one ABI byte; reject the reserved region 27..255."""
    value = int(np.uint8(code))
    if value >= VALID_STATE_COUNT:
        raise ValueError(f"reserved triadic ABI byte: {value}")
    return STATE_TO_TRIPLE[value]


def support_mask(triple: Triple) -> np.uint8:
    """Three-bit support mask: bit 2=first, bit 1=second, bit 0=third."""
    _check_triple(triple)
    a, b, c = triple
    return np.uint8(((a != 0) << 2) | ((b != 0) << 1) | (c != 0))


def active_count(triple: Triple) -> int:
    """Count active/nonzero trits."""
    return int(int(support_mask(triple)).bit_count())


def prefix_sign_offsets(triple: Triple) -> tuple[int, int, int]:
    """Sign-stream address at each logical position = active trits before it."""
    _check_triple(triple)
    a, b, _ = triple
    return (0, int(a != 0), int(a != 0) + int(b != 0))


def self_test() -> None:
    triples = tuple(product(TRITS, repeat=3))
    assert len(triples) == VALID_STATE_COUNT
    assert len(STATE_TO_TRIPLE) == VALID_STATE_COUNT
    assert len(TRIPLE_TO_STATE) == VALID_STATE_COUNT
    assert len(set(STATE_TO_TRIPLE)) == VALID_STATE_COUNT

    # Exact total round trip over all 27 source states.
    for triple in triples:
        code = encode_u8(triple)
        assert isinstance(code, np.uint8)
        assert int(code) < VALID_STATE_COUNT
        assert decode_u8(code) == triple

        mask = int(support_mask(triple))
        assert 0 <= mask < 8
        assert active_count(triple) == mask.bit_count()

        offsets = prefix_sign_offsets(triple)
        a, b, c = triple
        expected = (0, int(a != 0), int(a != 0) + int(b != 0))
        assert offsets == expected
        assert offsets[2] <= active_count(triple)

    # Exact inverse on every valid encoded state.
    for code in range(VALID_STATE_COUNT):
        byte = np.uint8(code)
        assert int(encode_u8(decode_u8(byte))) == code

    # Reserved byte region is fail-closed.
    rejected = 0
    for code in range(VALID_STATE_COUNT, 256):
        try:
            decode_u8(np.uint8(code))
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f"reserved byte {code} unexpectedly decoded")
    assert rejected == 256 - VALID_STATE_COUNT

    print(
        "triadic_u8_codec_reference: ok; "
        f"valid={VALID_STATE_COUNT}; reserved={rejected}; carrier=numpy.uint8"
    )


if __name__ == "__main__":
    self_test()
