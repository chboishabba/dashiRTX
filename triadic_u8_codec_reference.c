#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Compiled scalar host-byte oracle for the DASHI three-trit compact ABI.
 *
 * Exact ABI order is shared with:
 *   DASHI.ComputerScience.TriadicFin27Byte256ABIExact.stateCode
 *
 * Valid byte values: 0..26.
 * Reserved/invalid byte values: 27..255.
 *
 * This is a scalar C11 reference. It does not claim SWAR/SIMD/CUDA/ROCm
 * equivalence or performance.
 */

typedef struct {
    int8_t a;
    int8_t b;
    int8_t c;
} TritTriple;

static const TritTriple STATE_TO_TRIPLE[27] = {
    { 0,  0,  0},
    { 0,  0, -1},
    { 0,  0,  1},
    { 0, -1,  0},
    { 0,  1,  0},
    { 0, -1, -1},
    { 0, -1,  1},
    { 0,  1, -1},
    { 0,  1,  1},
    {-1,  0,  0},
    { 1,  0,  0},
    {-1,  0, -1},
    {-1,  0,  1},
    { 1,  0, -1},
    { 1,  0,  1},
    {-1, -1,  0},
    {-1,  1,  0},
    { 1, -1,  0},
    { 1,  1,  0},
    {-1, -1, -1},
    {-1, -1,  1},
    {-1,  1, -1},
    {-1,  1,  1},
    { 1, -1, -1},
    { 1, -1,  1},
    { 1,  1, -1},
    { 1,  1,  1},
};

static int valid_trit(int8_t t) {
    return t == -1 || t == 0 || t == 1;
}

static int triple_equal(TritTriple x, TritTriple y) {
    return x.a == y.a && x.b == y.b && x.c == y.c;
}

static int encode_u8(TritTriple t, uint8_t *out) {
    if (!valid_trit(t.a) || !valid_trit(t.b) || !valid_trit(t.c) || out == NULL) {
        return 0;
    }
    for (uint8_t i = 0; i < 27; ++i) {
        if (triple_equal(t, STATE_TO_TRIPLE[i])) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

static int decode_u8(uint8_t code, TritTriple *out) {
    if (code >= 27 || out == NULL) {
        return 0;
    }
    *out = STATE_TO_TRIPLE[code];
    return 1;
}

static uint8_t support_mask(TritTriple t) {
    return (uint8_t)(((t.a != 0) << 2) | ((t.b != 0) << 1) | (t.c != 0));
}

static unsigned active_count(TritTriple t) {
    return (unsigned)__builtin_popcount((unsigned)support_mask(t));
}

static void prefix_sign_offsets(TritTriple t, unsigned out[3]) {
    out[0] = 0u;
    out[1] = (unsigned)(t.a != 0);
    out[2] = (unsigned)(t.a != 0) + (unsigned)(t.b != 0);
}

static void fail(const char *message) {
    fprintf(stderr, "triadic_u8_codec_reference.c: FAIL: %s\n", message);
    exit(EXIT_FAILURE);
}

int main(void) {
    unsigned valid = 0u;
    unsigned reserved = 0u;

    for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
            for (int c = -1; c <= 1; ++c) {
                TritTriple source = {(int8_t)a, (int8_t)b, (int8_t)c};
                TritTriple decoded;
                uint8_t code = 255u;
                unsigned offsets[3];

                if (!encode_u8(source, &code)) fail("encode rejected valid triple");
                if (code >= 27) fail("encode escaped valid ABI range");
                if (!decode_u8(code, &decoded)) fail("decode rejected valid code");
                if (!triple_equal(source, decoded)) fail("decode(encode(t)) != t");

                prefix_sign_offsets(source, offsets);
                if (offsets[0] != 0u) fail("first prefix offset != 0");
                if (offsets[1] != (unsigned)(source.a != 0)) fail("second prefix mismatch");
                if (offsets[2] != (unsigned)(source.a != 0) + (unsigned)(source.b != 0)) {
                    fail("third prefix mismatch");
                }
                if (active_count(source) != (unsigned)(source.a != 0) +
                                            (unsigned)(source.b != 0) +
                                            (unsigned)(source.c != 0)) {
                    fail("active count mismatch");
                }
                ++valid;
            }
        }
    }

    if (valid != 27u) fail("did not enumerate exactly 27 valid triples");

    for (unsigned code = 0u; code < 27u; ++code) {
        TritTriple decoded;
        uint8_t encoded = 255u;
        if (!decode_u8((uint8_t)code, &decoded)) fail("valid byte rejected");
        if (!encode_u8(decoded, &encoded)) fail("re-encode failed");
        if (encoded != (uint8_t)code) fail("encode(decode(code)) != code");
    }

    for (unsigned code = 27u; code < 256u; ++code) {
        TritTriple ignored;
        if (decode_u8((uint8_t)code, &ignored)) fail("reserved byte decoded");
        ++reserved;
    }

    if (reserved != 229u) fail("reserved-byte count mismatch");

    printf("triadic_u8_codec_reference.c: ok; valid=%u; reserved=%u; carrier=uint8_t\n",
           valid, reserved);
    return EXIT_SUCCESS;
}
