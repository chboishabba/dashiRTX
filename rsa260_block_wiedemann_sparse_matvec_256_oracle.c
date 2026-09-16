#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Synthetic sparse GF(2) 256x256 matrix-vector oracle for the RSA-260
 * Block-Wiedemann execution lane.
 *
 * The 256 coordinates are retained as four contiguous 64-coordinate blocks,
 * matching the already-paid packed row carrier (4 x uint64_t). This block shape
 * is also structurally analogous to DASHI's DNA256 = Vec DNA64 4 hierarchy,
 * but the algebra here is strictly GF(2); no biological semantics are imported.
 *
 * This is NOT the RSA-260 matrix and NOT Lu/Devin's implementation.
 */

typedef struct { uint64_t w[4]; } Vec256;
typedef struct { uint64_t row[256][4]; } Mat256;

static unsigned parity64(uint64_t x) {
    return (unsigned)(__builtin_popcountll((unsigned long long)x) & 1u);
}

static unsigned dot256(const uint64_t row[4], const Vec256 *v) {
    return parity64(row[0] & v->w[0]) ^
           parity64(row[1] & v->w[1]) ^
           parity64(row[2] & v->w[2]) ^
           parity64(row[3] & v->w[3]);
}

static void set_bit(Vec256 *v, unsigned i) {
    v->w[i >> 6] |= UINT64_C(1) << (i & 63u);
}

static unsigned get_bit(const Vec256 *v, unsigned i) {
    return (unsigned)((v->w[i >> 6] >> (i & 63u)) & 1u);
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
        /* Exactly five requested positions per row before duplicate cancellation. */
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

static void fail(uint64_t case_id) {
    fprintf(stderr, "rsa260_sparse_matvec_256_oracle: FAIL case=%llu\n",
            (unsigned long long)case_id);
    exit(EXIT_FAILURE);
}

int main(void) {
    Mat256 m;
    Vec256 x = {{0,0,0,0}}, packed, scalar;
    uint64_t checks = 0;

    /* Identity basis check across all four 64-coordinate blocks. */
    memset(&m, 0, sizeof(m));
    for (unsigned i = 0; i < 256; ++i)
        m.row[i][i >> 6] = UINT64_C(1) << (i & 63u);
    for (unsigned i = 0; i < 256; ++i) {
        memset(&x, 0, sizeof(x));
        set_bit(&x, i);
        matvec_packed(&m, &x, &packed);
        if (!equal256(&x, &packed)) fail(checks);
        ++checks;
    }

    /* Broad deterministic sparse matrices and vectors against scalar oracle. */
    for (uint64_t s = 0; s < 256; ++s) {
        build_sparse_matrix(&m, mix64(s + 1));
        for (uint64_t vcase = 0; vcase < 256; ++vcase) {
            for (unsigned w = 0; w < 4; ++w)
                x.w[w] = mix64((s << 32) ^ (vcase << 8) ^ w);
            matvec_packed(&m, &x, &packed);
            matvec_scalar(&m, &x, &scalar);
            if (!equal256(&packed, &scalar)) fail(checks);
            ++checks;
        }
    }

    printf("rsa260_sparse_matvec_256_oracle: ok; checks=%llu; rows=256; blocks=4x64; carrier=4xuint64_t\n",
           (unsigned long long)checks);
    return EXIT_SUCCESS;
}
