#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * SWAR sign-compaction / factorized encode kernel for the DASHI 27-state ABI.
 *
 * Eight independent symbols occupy the eight byte lanes of two uint64_t words:
 *   support:  bit2=first active, bit1=second active, bit0=third active
 *   positive: bit2=first positive, bit1=second positive, bit0=third positive
 *             (bits on inactive positions are required to be zero)
 *
 * The output byte in every lane is the exact dense ABI code 0..26 used by
 * TriadicFin27Byte256ABIExact.stateCode.  The implementation is table-free and
 * computes the code from lane-parallel boolean/arithmetic identities.
 *
 * This pays SWAR sign compaction plus complete encoding from the already
 * factorized support/sign carrier. It does not yet pay raw-trit factorization,
 * SWAR decode, SIMD, CUDA, ROCm, Rust/CubeCL, or performance superiority.
 */

static const uint64_t L = UINT64_C(0x0101010101010101);

static uint64_t lane_bit(uint64_t x, unsigned bit) {
    return (x >> bit) & L;
}

static uint64_t swar_state_codes(uint64_t support, uint64_t positive) {
    const uint64_t a = lane_bit(support, 2);
    const uint64_t b = lane_bit(support, 1);
    const uint64_t c = lane_bit(support, 0);
    const uint64_t sa = lane_bit(positive, 2) & a;
    const uint64_t sb = lane_bit(positive, 1) & b;
    const uint64_t sc = lane_bit(positive, 0) & c;

    /* base(mask) for masks 0..7 is 0,1,3,5,9,11,15,19.
       Derived without lookup:
         9*a + 3*2^a*b + 2^(a+b)*c
       with 2^a = 1+a and 2^(a+b)=1+a+b+a*b for bits a,b. */
    const uint64_t ab = a & b;
    const uint64_t ac = a & c;
    const uint64_t bc = b & c;
    const uint64_t abc = ab & c;

    const uint64_t term_a = a + (a << 3);                 /* 9*a */
    const uint64_t three_b = b + (b << 1);               /* 3*b */
    const uint64_t three_ab = ab + (ab << 1);            /* 3*a*b */
    const uint64_t term_b = three_b + three_ab;           /* 3*(1+a)*b */
    const uint64_t term_c = c + ac + bc + abc;            /* (1+a+b+ab)*c */
    const uint64_t base = term_a + term_b + term_c;

    /* Within one support mask, signs are ordered positionally with negative=0,
       positive=1.  Hence the first active sign is the most significant sign
       digit.  Its weight is 2^(number of later active positions). */
    const uint64_t sa_a = sa & a;
    const uint64_t sb_b = sb & b;
    const uint64_t sc_c = sc & c;
    const uint64_t sign_a = sa_a + (sa_a & b) + (sa_a & c) + (sa_a & b & c);
    const uint64_t sign_b = sb_b + (sb_b & c);
    const uint64_t sign_index = sign_a + sign_b + sc_c;

    return base + sign_index;
}

typedef struct {
    uint8_t support;
    uint8_t positive;
} Factorized;

static const Factorized CODE_TO_FACTOR[27] = {
    {0u,0u},
    {1u,0u},{1u,1u},
    {2u,0u},{2u,2u},
    {3u,0u},{3u,1u},{3u,2u},{3u,3u},
    {4u,0u},{4u,4u},
    {5u,0u},{5u,1u},{5u,4u},{5u,5u},
    {6u,0u},{6u,2u},{6u,4u},{6u,6u},
    {7u,0u},{7u,1u},{7u,2u},{7u,3u},
    {7u,4u},{7u,5u},{7u,6u},{7u,7u}
};

static void fail(uint32_t key, unsigned lane, unsigned state, unsigned got) {
    fprintf(stderr,
            "triadic_sign_compact_encode_swar: FAIL key=%u lane=%u state=%u got=%u\n",
            key, lane, state, got);
    exit(EXIT_FAILURE);
}

int main(void) {
    const uint32_t cases = UINT32_C(14348907); /* 27^5 */

    for (uint32_t key = 0; key < cases; ++key) {
        uint32_t cursor = key;
        unsigned states[8];

        for (unsigned lane = 0; lane < 5; ++lane) {
            states[lane] = cursor % 27u;
            cursor /= 27u;
        }
        /* Deterministic mixed fillers exercise the remaining lanes throughout
           the full 27^5 core product without claiming exhaustive 27^8. */
        states[5] = (states[0] + 2u * states[1] + states[4]) % 27u;
        states[6] = (states[1] + 3u * states[2] + states[3]) % 27u;
        states[7] = (states[2] + 5u * states[3] + states[4]) % 27u;

        uint64_t supports = UINT64_C(0);
        uint64_t positives = UINT64_C(0);
        for (unsigned lane = 0; lane < 8; ++lane) {
            const Factorized f = CODE_TO_FACTOR[states[lane]];
            supports |= ((uint64_t)f.support) << (8u * lane);
            positives |= ((uint64_t)f.positive) << (8u * lane);
        }

        const uint64_t codes = swar_state_codes(supports, positives);
        for (unsigned lane = 0; lane < 8; ++lane) {
            const unsigned got = (unsigned)((codes >> (8u * lane)) & UINT64_C(0xff));
            if (got != states[lane]) {
                fail(key, lane, states[lane], got);
            }
        }
    }

    printf("triadic_sign_compact_encode_swar: ok; core_product=%u; lanes=8; carrier=uint64_t\n",
           cases);
    return EXIT_SUCCESS;
}
