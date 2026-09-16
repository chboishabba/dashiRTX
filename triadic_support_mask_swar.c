#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Genuine SWAR kernel for the DASHI triadic support-mask lane.
 *
 * Eight independent 3-bit support masks (values 0..7) are stored in the eight
 * byte lanes of one uint64_t. swar_popcount_bytes computes the population count
 * of every byte lane in parallel with the classic mask/add SWAR network.
 *
 * The self-test is exhaustive over the codec domain: 8^8 = 16,777,216 packed
 * words. This pays a concrete within-register parallel active-count operation;
 * it does not yet pay complete encode/decode SWAR, SIMD, CUDA, or ROCm.
 */

static uint64_t swar_popcount_bytes(uint64_t x) {
    x = x - ((x >> 1) & UINT64_C(0x5555555555555555));
    x = (x & UINT64_C(0x3333333333333333)) +
        ((x >> 2) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return x;
}

static unsigned scalar_popcount3(uint8_t mask) {
    return (unsigned)(mask & 1u) +
           (unsigned)((mask >> 1) & 1u) +
           (unsigned)((mask >> 2) & 1u);
}

static void fail(uint32_t key, unsigned lane, uint8_t mask,
                 unsigned got, unsigned expected) {
    fprintf(stderr,
            "triadic_support_mask_swar: FAIL key=%u lane=%u mask=%u got=%u expected=%u\n",
            key, lane, (unsigned)mask, got, expected);
    exit(EXIT_FAILURE);
}

int main(void) {
    const uint32_t cases = UINT32_C(16777216); /* 8^8 */

    for (uint32_t key = 0; key < cases; ++key) {
        uint32_t cursor = key;
        uint64_t packed = UINT64_C(0);
        uint8_t masks[8];

        for (unsigned lane = 0; lane < 8; ++lane) {
            const uint8_t mask = (uint8_t)(cursor & 7u);
            cursor >>= 3;
            masks[lane] = mask;
            packed |= ((uint64_t)mask) << (lane * 8u);
        }

        const uint64_t counts = swar_popcount_bytes(packed);
        for (unsigned lane = 0; lane < 8; ++lane) {
            const unsigned got = (unsigned)((counts >> (lane * 8u)) & UINT64_C(0xff));
            const unsigned expected = scalar_popcount3(masks[lane]);
            if (got != expected) {
                fail(key, lane, masks[lane], got, expected);
            }
        }
    }

    printf("triadic_support_mask_swar: ok; exhaustive_words=%u; lanes=8; carrier=uint64_t\n",
           cases);
    return EXIT_SUCCESS;
}
