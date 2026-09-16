#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * SWAR prefix-address kernel for the DASHI three-trit support mask.
 *
 * Eight independent support masks (one per byte lane, values 0..7) are packed
 * into one uint64_t. Bit 2 = first trit active, bit 1 = second, bit 0 = third.
 * For every byte lane simultaneously this computes:
 *
 *   offset(first)  = 0
 *   offset(second) = active(first)
 *   offset(third)  = active(first) + active(second)
 *
 * These are exactly the sign-stream prefix addresses used by the formal codec.
 * The self-test exhausts all 8^8 packed support-mask words.
 *
 * This pays SWAR prefix-address generation only. It does not yet pay sign-bit
 * compaction/scatter, full SWAR encode/decode, SIMD, CUDA, or ROCm.
 */

static uint64_t swar_second_offsets(uint64_t masks) {
    return (masks >> 2) & UINT64_C(0x0101010101010101);
}

static uint64_t swar_third_offsets(uint64_t masks) {
    const uint64_t first = (masks >> 2) & UINT64_C(0x0101010101010101);
    const uint64_t second = (masks >> 1) & UINT64_C(0x0101010101010101);
    return first + second;
}

static unsigned scalar_second_offset(uint8_t mask) {
    return (unsigned)((mask >> 2) & 1u);
}

static unsigned scalar_third_offset(uint8_t mask) {
    return (unsigned)((mask >> 2) & 1u) + (unsigned)((mask >> 1) & 1u);
}

static void fail(uint32_t key, unsigned lane, uint8_t mask,
                 unsigned got_second, unsigned expected_second,
                 unsigned got_third, unsigned expected_third) {
    fprintf(stderr,
            "triadic_prefix_swar: FAIL key=%u lane=%u mask=%u "
            "second=%u/%u third=%u/%u\n",
            key, lane, (unsigned)mask,
            got_second, expected_second, got_third, expected_third);
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

        const uint64_t second_offsets = swar_second_offsets(packed);
        const uint64_t third_offsets = swar_third_offsets(packed);

        for (unsigned lane = 0; lane < 8; ++lane) {
            const unsigned got_second =
                (unsigned)((second_offsets >> (lane * 8u)) & UINT64_C(0xff));
            const unsigned got_third =
                (unsigned)((third_offsets >> (lane * 8u)) & UINT64_C(0xff));
            const unsigned expected_second = scalar_second_offset(masks[lane]);
            const unsigned expected_third = scalar_third_offset(masks[lane]);

            if (got_second != expected_second || got_third != expected_third) {
                fail(key, lane, masks[lane],
                     got_second, expected_second, got_third, expected_third);
            }
        }
    }

    printf("triadic_prefix_swar: ok; exhaustive_words=%u; lanes=8; carrier=uint64_t\n",
           cases);
    return EXIT_SUCCESS;
}
