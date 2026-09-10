#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Synthetic Block-Wiedemann-shaped projection sequence oracle over GF(2).
 *
 * Carrier:
 *   Vec256 = 4 x uint64_t = 256 GF(2) coordinates.
 *
 * For a deterministic sparse 256x256 matrix M, deterministic projection block
 * X with 16 columns, and deterministic starting block Y with 16 columns, form
 *
 *     S_k = X^T M^k Y
 *
 * as a 16x16 GF(2) matrix for k = 0..31.
 *
 * Packed 4x64 execution is compared against an independent bit-by-bit scalar
 * implementation at every Krylov update and every one of the 256 projection
 * entries at every step.
 *
 * This is NOT the RSA-260 matrix, NOT the published m=n=512 run geometry,
 * NOT Lu/Devin's implementation, and NOT a full Block Wiedemann solve.
 */

enum { WIDTH = 256, WORDS = 4, BLOCK = 16, STEPS = 32 };

typedef struct { uint64_t w[WORDS]; } Vec256;
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;
typedef struct { Vec256 col[BLOCK]; } Block256x16;
typedef struct { uint16_t row[BLOCK]; } Mat16;

static unsigned parity64(uint64_t x) {
    return (unsigned)(__builtin_popcountll((unsigned long long)x) & 1u);
}

static unsigned get_bit(const Vec256 *v, unsigned i) {
    return (unsigned)((v->w[i >> 6] >> (i & 63u)) & UINT64_C(1));
}

static void set_bit(Vec256 *v, unsigned i) {
    v->w[i >> 6] |= UINT64_C(1) << (i & 63u);
}

static unsigned dot_packed(const Vec256 *a, const Vec256 *b) {
    return parity64(a->w[0] & b->w[0]) ^
           parity64(a->w[1] & b->w[1]) ^
           parity64(a->w[2] & b->w[2]) ^
           parity64(a->w[3] & b->w[3]);
}

static unsigned dot_scalar(const Vec256 *a, const Vec256 *b) {
    unsigned acc = 0u;
    for (unsigned i = 0; i < WIDTH; ++i)
        acc ^= get_bit(a, i) & get_bit(b, i);
    return acc;
}

static void matvec_packed(const Mat256 *m, const Vec256 *x, Vec256 *y) {
    memset(y, 0, sizeof(*y));
    for (unsigned block = 0; block < WORDS; ++block) {
        for (unsigned r = 0; r < 64; ++r) {
            const unsigned row = 64u * block + r;
            Vec256 rr = {{ m->row[row][0], m->row[row][1],
                           m->row[row][2], m->row[row][3] }};
            if (dot_packed(&rr, x)) set_bit(y, row);
        }
    }
}

static void matvec_scalar(const Mat256 *m, const Vec256 *x, Vec256 *y) {
    memset(y, 0, sizeof(*y));
    for (unsigned r = 0; r < WIDTH; ++r) {
        unsigned acc = 0u;
        for (unsigned c = 0; c < WIDTH; ++c) {
            const unsigned mb = (unsigned)((m->row[r][c >> 6] >> (c & 63u)) & UINT64_C(1));
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
    for (unsigned r = 0; r < WIDTH; ++r) {
        for (unsigned k = 0; k < 7; ++k) {
            const uint64_t h = mix64(seed ^ ((uint64_t)r << 12) ^ (uint64_t)k);
            const unsigned c = (unsigned)(h & 255u);
            m->row[r][c >> 6] ^= UINT64_C(1) << (c & 63u);
        }
    }
}

static void build_block(Block256x16 *b, uint64_t seed) {
    for (unsigned j = 0; j < BLOCK; ++j)
        for (unsigned w = 0; w < WORDS; ++w)
            b->col[j].w[w] = mix64(seed ^ ((uint64_t)j << 16) ^
                                   (UINT64_C(0x9e3779b97f4a7c15) * (w + 1u)));
}

static void project_packed(const Block256x16 *x, const Block256x16 *y, Mat16 *s) {
    for (unsigned i = 0; i < BLOCK; ++i) {
        uint16_t row = 0;
        for (unsigned j = 0; j < BLOCK; ++j)
            row |= (uint16_t)(dot_packed(&x->col[i], &y->col[j]) << j);
        s->row[i] = row;
    }
}

static void project_scalar(const Block256x16 *x, const Block256x16 *y, Mat16 *s) {
    for (unsigned i = 0; i < BLOCK; ++i) {
        uint16_t row = 0;
        for (unsigned j = 0; j < BLOCK; ++j)
            row |= (uint16_t)(dot_scalar(&x->col[i], &y->col[j]) << j);
        s->row[i] = row;
    }
}

static int equal_vec(const Vec256 *a, const Vec256 *b) {
    return a->w[0] == b->w[0] && a->w[1] == b->w[1] &&
           a->w[2] == b->w[2] && a->w[3] == b->w[3];
}

static int equal_mat16(const Mat16 *a, const Mat16 *b) {
    for (unsigned i = 0; i < BLOCK; ++i)
        if (a->row[i] != b->row[i]) return 0;
    return 1;
}

static uint64_t digest_mat16(const Mat16 *s, uint64_t h) {
    for (unsigned i = 0; i < BLOCK; ++i)
        h = mix64(h ^ (uint64_t)s->row[i] ^ ((uint64_t)i << 32));
    return h;
}

static void fail(const char *phase, unsigned step, unsigned column) {
    fprintf(stderr,
            "rsa260_projection_sequence_256_oracle: FAIL phase=%s step=%u column=%u\n",
            phase, step, column);
    exit(EXIT_FAILURE);
}

int main(void) {
    Mat256 m;
    Block256x16 x, yp, ys, np, ns;
    Mat16 sp, ss;
    uint64_t projection_entries_checked = 0;
    uint64_t krylov_column_steps_checked = 0;
    uint64_t digest = UINT64_C(0x243f6a8885a308d3);

    build_sparse_matrix(&m, UINT64_C(0x6a09e667f3bcc909));
    build_block(&x, UINT64_C(0xbb67ae8584caa73b));
    build_block(&yp, UINT64_C(0x3c6ef372fe94f82b));
    ys = yp;

    for (unsigned step = 0; step < STEPS; ++step) {
        project_packed(&x, &yp, &sp);
        project_scalar(&x, &ys, &ss);
        if (!equal_mat16(&sp, &ss)) fail("projection", step, 0);
        projection_entries_checked += (uint64_t)BLOCK * (uint64_t)BLOCK;
        digest = digest_mat16(&sp, digest ^ step);

        for (unsigned j = 0; j < BLOCK; ++j) {
            matvec_packed(&m, &yp.col[j], &np.col[j]);
            matvec_scalar(&m, &ys.col[j], &ns.col[j]);
            if (!equal_vec(&np.col[j], &ns.col[j])) fail("krylov", step, j);
            ++krylov_column_steps_checked;
        }
        yp = np;
        ys = ns;
    }

    printf("rsa260_projection_sequence_256_oracle: ok; width=%u; block=%u; steps=%u; projection_entries=%llu; krylov_column_steps=%llu; blocks=4x64; digest=%016llx\n",
           WIDTH, BLOCK, STEPS,
           (unsigned long long)projection_entries_checked,
           (unsigned long long)krylov_column_steps_checked,
           (unsigned long long)digest);
    return EXIT_SUCCESS;
}
