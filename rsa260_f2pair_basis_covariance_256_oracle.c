#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * F2^2 / four-state basis-covariance oracle for the RSA-260 width-256 lane.
 *
 * Pair 256 GF(2) coordinates into 128 two-bit symbols. For each of the six
 * invertible 2x2 matrices A over GF(2), transform the right vector by A and
 * the left vector by A^{-T}; the ordinary GF(2) dot product must be preserved.
 *
 * This is a structural cross-pollination with the four-colour carrier
 * {00,01,10,11} and its three nonzero differences. It is NOT a graph-colouring
 * algorithm, NOT a Kempe move, and NOT a Block Wiedemann solve.
 */

typedef struct { uint64_t w[4]; } Vec256;

typedef struct {
    unsigned a00, a01, a10, a11;
} Mat2;

static const Mat2 GL2[6] = {
    {1,0,0,1}, /* I */
    {0,1,1,0}, /* swap */
    {1,1,0,1},
    {1,0,1,1},
    {0,1,1,1},
    {1,1,1,0}
};

static unsigned get_bit(const Vec256 *v, unsigned i) {
    return (unsigned)((v->w[i >> 6] >> (i & 63u)) & UINT64_C(1));
}

static void set_bit(Vec256 *v, unsigned i, unsigned b) {
    const uint64_t mask = UINT64_C(1) << (i & 63u);
    if (b) v->w[i >> 6] |= mask;
    else   v->w[i >> 6] &= ~mask;
}

static unsigned parity64(uint64_t x) {
    return (unsigned)(__builtin_popcountll((unsigned long long)x) & 1u);
}

static unsigned dot256(const Vec256 *x, const Vec256 *y) {
    return parity64(x->w[0] & y->w[0]) ^
           parity64(x->w[1] & y->w[1]) ^
           parity64(x->w[2] & y->w[2]) ^
           parity64(x->w[3] & y->w[3]);
}

static Mat2 inverse2(Mat2 a) {
    /* Over GF(2), det = a00*a11 + a01*a10 = 1 for GL(2,2). */
    Mat2 r = {a.a11, a.a01, a.a10, a.a00};
    return r;
}

static Mat2 transpose2(Mat2 a) {
    Mat2 r = {a.a00, a.a10, a.a01, a.a11};
    return r;
}

static void transform_pairs(const Vec256 *in, Vec256 *out, Mat2 a) {
    out->w[0] = out->w[1] = out->w[2] = out->w[3] = 0;
    for (unsigned p = 0; p < 128; ++p) {
        const unsigned x0 = get_bit(in, 2u*p);
        const unsigned x1 = get_bit(in, 2u*p + 1u);
        const unsigned y0 = (a.a00 & x0) ^ (a.a01 & x1);
        const unsigned y1 = (a.a10 & x0) ^ (a.a11 & x1);
        set_bit(out, 2u*p, y0);
        set_bit(out, 2u*p + 1u, y1);
    }
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static void fail(const char *phase, uint64_t case_id, unsigned matrix_id) {
    fprintf(stderr,
            "rsa260_f2pair_basis_covariance_256_oracle: FAIL phase=%s case=%llu matrix=%u\n",
            phase, (unsigned long long)case_id, matrix_id);
    exit(EXIT_FAILURE);
}

int main(void) {
    uint64_t local_checks = 0;
    uint64_t broad_checks = 0;

    /* Exhaust all local F2^2 symbol pairs for all six invertible matrices. */
    for (unsigned mi = 0; mi < 6; ++mi) {
        const Mat2 a = GL2[mi];
        const Mat2 dual = transpose2(inverse2(a)); /* A^{-T} */
        for (unsigned lx = 0; lx < 4; ++lx) {
            for (unsigned ry = 0; ry < 4; ++ry) {
                Vec256 x = {{0,0,0,0}}, y = {{0,0,0,0}}, tx, ty;
                set_bit(&x, 0, lx & 1u);
                set_bit(&x, 1, (lx >> 1) & 1u);
                set_bit(&y, 0, ry & 1u);
                set_bit(&y, 1, (ry >> 1) & 1u);
                const unsigned before = dot256(&x, &y);
                transform_pairs(&x, &tx, dual);
                transform_pairs(&y, &ty, a);
                if (dot256(&tx, &ty) != before) fail("local", local_checks, mi);
                ++local_checks;
            }
        }
    }

    /* Broad deterministic width-256 checks. */
    const uint64_t cases = UINT64_C(262144); /* 2^18 */
    for (uint64_t k = 0; k < cases; ++k) {
        Vec256 x, y, tx, ty;
        for (unsigned w = 0; w < 4; ++w) {
            x.w[w] = mix64(k ^ (UINT64_C(0x9e3779b97f4a7c15) * (2u*w + 1u)));
            y.w[w] = mix64((k + UINT64_C(0xd1b54a32d192ed03)) ^
                           (UINT64_C(0x94d049bb133111eb) * (2u*w + 2u)));
        }
        const unsigned before = dot256(&x, &y);
        for (unsigned mi = 0; mi < 6; ++mi) {
            const Mat2 a = GL2[mi];
            const Mat2 dual = transpose2(inverse2(a));
            transform_pairs(&x, &tx, dual);
            transform_pairs(&y, &ty, a);
            if (dot256(&tx, &ty) != before) fail("broad", k, mi);
            ++broad_checks;
        }
    }

    printf("rsa260_f2pair_basis_covariance_256_oracle: ok; local_checks=%llu; broad_checks=%llu; matrices=6; pairs=128; width=256\n",
           (unsigned long long)local_checks,
           (unsigned long long)broad_checks);
    return EXIT_SUCCESS;
}
