#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Packed GF(2) 256-bit row-dot oracle for the RSA-260 Block Wiedemann lane.
 *
 * One 256-bit row and one 256-bit vector are represented by four uint64_t
 * words. The GF(2) dot product is parity(popcount(row & vector)) across the
 * four words. This is a stage-local executable primitive consistent with the
 * published RSA-260 width-256 linear-algebra coordinate.
 *
 * It is NOT the RSA-260 matrix, NOT Lu/Devin's Block Wiedemann kernel, and NOT
 * an independent reproduction of the RSA-260 run.
 */

static unsigned parity64(uint64_t x) {
    return (unsigned)(__builtin_popcountll((unsigned long long)x) & 1u);
}

static unsigned gf2_dot256_packed(const uint64_t row[4], const uint64_t vec[4]) {
    return parity64(row[0] & vec[0]) ^
           parity64(row[1] & vec[1]) ^
           parity64(row[2] & vec[2]) ^
           parity64(row[3] & vec[3]);
}

static unsigned gf2_dot256_scalar(const uint64_t row[4], const uint64_t vec[4]) {
    unsigned parity = 0u;
    for (unsigned w = 0; w < 4; ++w) {
        for (unsigned b = 0; b < 64; ++b) {
            const unsigned rb = (unsigned)((row[w] >> b) & UINT64_C(1));
            const unsigned vb = (unsigned)((vec[w] >> b) & UINT64_C(1));
            parity ^= rb & vb;
        }
    }
    return parity;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static void fail(uint64_t case_id, unsigned got, unsigned expected) {
    fprintf(stderr,
            "rsa260_block_wiedemann_gf2_packed_oracle: FAIL case=%llu got=%u expected=%u\n",
            (unsigned long long)case_id, got, expected);
    exit(EXIT_FAILURE);
}

int main(void) {
    /* Exact one-hot basis checks: 256 matching + 256 disjoint cases. */
    uint64_t row[4] = {0, 0, 0, 0};
    uint64_t vec[4] = {0, 0, 0, 0};
    unsigned basis_cases = 0u;

    for (unsigned i = 0; i < 256; ++i) {
        for (unsigned w = 0; w < 4; ++w) {
            row[w] = 0;
            vec[w] = 0;
        }
        row[i / 64] = UINT64_C(1) << (i % 64);
        vec[i / 64] = UINT64_C(1) << (i % 64);
        if (gf2_dot256_packed(row, vec) != 1u) fail(i, gf2_dot256_packed(row, vec), 1u);
        ++basis_cases;

        const unsigned j = (i + 1u) & 255u;
        for (unsigned w = 0; w < 4; ++w) vec[w] = 0;
        vec[j / 64] = UINT64_C(1) << (j % 64);
        if (gf2_dot256_packed(row, vec) != 0u) fail(256u + i, gf2_dot256_packed(row, vec), 0u);
        ++basis_cases;
    }

    /* Deterministic broad comparison against a scalar bit-by-bit oracle. */
    const uint64_t random_cases = UINT64_C(1048576); /* 2^20 */
    for (uint64_t k = 0; k < random_cases; ++k) {
        for (unsigned w = 0; w < 4; ++w) {
            row[w] = mix64(k ^ (UINT64_C(0x9e3779b97f4a7c15) * (2u * w + 1u)));
            vec[w] = mix64((k + UINT64_C(0xd1b54a32d192ed03)) ^
                           (UINT64_C(0x94d049bb133111eb) * (2u * w + 2u)));
        }
        const unsigned got = gf2_dot256_packed(row, vec);
        const unsigned expected = gf2_dot256_scalar(row, vec);
        if (got != expected) fail(512u + k, got, expected);
    }

    printf("rsa260_block_wiedemann_gf2_packed_oracle: ok; basis_cases=%u; random_cases=%llu; width=256; carrier=4xuint64_t\n",
           basis_cases, (unsigned long long)random_cases);
    return EXIT_SUCCESS;
}
