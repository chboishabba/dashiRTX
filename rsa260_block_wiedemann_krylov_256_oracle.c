#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Synthetic Krylov-sequence oracle over GF(2) for the RSA-260 Block-Wiedemann lane.
 *
 * Reuses the same 256-coordinate carrier as the sparse SpMV oracle:
 *   Vec256 = 4 x uint64_t
 * and iterates x_{k+1} = M x_k for a deterministic synthetic sparse 256x256 matrix.
 *
 * The sequence is checked two ways at every step:
 *   packed SpMV over 4x64 blocks, and bit-by-bit scalar GF(2) SpMV.
 *
 * This is NOT the RSA-260 matrix, NOT Lu/Devin's Krylov sequence, NOT a block
 * Wiedemann implementation, and NOT an RSA-260 reproduction.
 */

typedef struct { uint64_t w[4]; } Vec256;
typedef struct { uint64_t row[256][4]; } Mat256;

static unsigned parity64(uint64_t x) {
    return (unsigned)(__builtin_popcountll((unsigned long long)x) & 1u);
}

static unsigned get_bit(const Vec256 *v, unsigned i) {
    return (unsigned)((v->w[i >> 6] >> (i & 63u)) & 1u);
}

static void set_bit(Vec256 *v, unsigned i) {
    v->w[i >> 6] |= UINT64_C(1) << (i & 63u);
}

static unsigned dot256(const uint64_t row[4], const Vec256 *v) {
    return parity64(row[0] & v->w[0]) ^
           parity64(row[1] & v->w[1]) ^
           parity64(row[2] & v->w[2]) ^
           parity64(row[3] & v->w[3]);
}

static void matvec_packed(const Mat256 *m, const Vec256 *x, Vec256 *y) {
    memset(y, 0, sizeof(*y));
    for (unsigned block = 0; block < 4; ++block) {
        for (unsigned r = 0; r < 64; ++r) {
            const unsigned row = block * 64u + r;
            if (dot256(m->row[row], x)) set_bit(y, row);
        }
    }
}

static void matvec_scalar(const Mat256 *m, const Vec256 *x, Vec256 *y) {
    memset(y, 0, sizeof(*y));
    for (unsigned r = 0; r < 256; ++r) {
        unsigned acc = 0u;
        for (unsigned c = 0; c < 256; ++c) {
            const unsigned mb = (unsigned)((m->row[r][c >> 6] >> (c & 63u)) & 1u);
            acc ^= mb & get_bit(x, c);
        }
        if (acc) set_bit(y, r);
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

static void build_sparse_matrix(Mat256 *m, uint64_t seed) {
    memset(m, 0, sizeof(*m));
    for (unsigned r = 0; r < 256; ++r) {
        for (unsigned k = 0; k < 5; ++k) {
            const uint64_t h = mix64(seed ^ ((uint64_t)r << 8) ^ k);
            const unsigned c = (unsigned)(h & 255u);
            m->row[r][c >> 6] ^= UINT64_C(1) << (c & 63u);
        }
    }
}

static int equal256(const Vec256 *a, const Vec256 *b) {
    return a->w[0] == b->w[0] && a->w[1] == b->w[1] &&
           a->w[2] == b->w[2] && a->w[3] == b->w[3];
}

static uint64_t fingerprint(const Vec256 *v) {
    uint64_t h = UINT64_C(0x6a09e667f3bcc909);
    for (unsigned i = 0; i < 4; ++i)
        h = mix64(h ^ v->w[i] ^ (UINT64_C(0x9e3779b97f4a7c15) * (i + 1u)));
    return h;
}

static void fail(unsigned seed_case, unsigned step) {
    fprintf(stderr, "rsa260_krylov_256_oracle: FAIL seed_case=%u step=%u\n", seed_case, step);
    exit(EXIT_FAILURE);
}

int main(void) {
    const unsigned matrix_cases = 64;
    const unsigned steps_per_case = 128;
    uint64_t checked_steps = 0;
    uint64_t sequence_digest = 0;

    for (unsigned mcase = 0; mcase < matrix_cases; ++mcase) {
        Mat256 m;
        Vec256 xp, xs, np, ns;
        build_sparse_matrix(&m, mix64((uint64_t)mcase + 1u));

        for (unsigned w = 0; w < 4; ++w) {
            const uint64_t seed = ((uint64_t)mcase << 32) ^
                                  (UINT64_C(0xd1b54a32d192ed03) * (w + 1u));
            xp.w[w] = mix64(seed);
            xs.w[w] = xp.w[w];
        }

        sequence_digest ^= fingerprint(&xp);

        for (unsigned step = 0; step < steps_per_case; ++step) {
            matvec_packed(&m, &xp, &np);
            matvec_scalar(&m, &xs, &ns);
            if (!equal256(&np, &ns)) fail(mcase, step);

            sequence_digest ^= mix64(fingerprint(&np) ^ ((uint64_t)mcase << 40) ^ step);
            xp = np;
            xs = ns;
            ++checked_steps;
        }
    }

    printf("rsa260_krylov_256_oracle: ok; matrices=%u; steps_per_matrix=%u; checked_steps=%llu; blocks=4x64; digest=%016llx\n",
           matrix_cases,
           steps_per_case,
           (unsigned long long)checked_steps,
           (unsigned long long)sequence_digest);
    return EXIT_SUCCESS;
}
