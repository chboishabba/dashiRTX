#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * BWC-shaped experiment over the repaired RSA-260 bidi candidate shadow.
 *
 * A is a 924 x 512 GF(2) sparse shadow with row excess 412.
 * B = A A^T is a 924 x 924 square left-nullspace preparation used ONLY as an
 * experimental adapter.  It is not claimed to equal CADO's production prep.
 */

enum {
  A_ROWS = 924,
  A_COLS = 512,
  A_WORDS = 8,
  B_WORDS = 15,
  PROJ_WIDTH = 8,
  KRYLOV_STEPS = 64,
  PROJECTION_TERMS = 32
};

typedef struct { uint64_t w[A_WORDS]; } VecC;
typedef struct { uint64_t w[B_WORDS]; } VecR;
typedef struct { uint64_t row[A_ROWS][A_WORDS]; } MatA;
typedef struct { uint64_t row[A_ROWS][B_WORDS]; } MatB;

static uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27;
  x *= UINT64_C(0x94d049bb133111eb);
  x ^= x >> 31;
  return x;
}

static unsigned parity64(uint64_t x) {
  return (unsigned)(__builtin_popcountll((unsigned long long)x) & 1u);
}

static unsigned dot_words(const uint64_t *a, const uint64_t *b, unsigned n) {
  unsigned p = 0;
  for (unsigned i = 0; i < n; ++i) p ^= parity64(a[i] & b[i]);
  return p;
}

static uint64_t shadow_row_base(unsigned r) {
  return ((uint64_t)r * UINT64_C(2654435761) + UINT64_C(0x9e3779b9)) % A_COLS;
}

static void build_A(MatA *a) {
  memset(a, 0, sizeof(*a));
  for (unsigned r = 0; r < A_ROWS; ++r) {
    unsigned deg = r < 6 ? 151u : 150u;
    uint64_t base = shadow_row_base(r);
    for (unsigned j = 0; j < deg; ++j) {
      unsigned c = (unsigned)((base + j) % A_COLS);
      a->row[r][c >> 6] |= UINT64_C(1) << (c & 63u);
    }
  }
}

static void build_B(const MatA *a, MatB *b) {
  memset(b, 0, sizeof(*b));
  for (unsigned r = 0; r < A_ROWS; ++r) {
    for (unsigned c = 0; c < A_ROWS; ++c) {
      if (dot_words(a->row[r], a->row[c], A_WORDS))
        b->row[r][c >> 6] |= UINT64_C(1) << (c & 63u);
    }
  }
}

static int highest_bit(const uint64_t *row, unsigned words, unsigned nbits) {
  for (int w = (int)words - 1; w >= 0; --w) {
    uint64_t x = row[w];
    if ((unsigned)w == words - 1 && (nbits & 63u))
      x &= (UINT64_C(1) << (nbits & 63u)) - 1u;
    if (x) return 64 * w + (63 - __builtin_clzll(x));
  }
  return -1;
}

static void xor_words(uint64_t *dst, const uint64_t *src, unsigned n) {
  for (unsigned i = 0; i < n; ++i) dst[i] ^= src[i];
}

static unsigned rank_A(const MatA *a) {
  uint64_t basis[A_COLS][A_WORDS];
  unsigned used[A_COLS];
  memset(basis, 0, sizeof(basis));
  memset(used, 0, sizeof(used));
  unsigned rank = 0;
  for (unsigned r = 0; r < A_ROWS; ++r) {
    uint64_t v[A_WORDS];
    memcpy(v, a->row[r], sizeof(v));
    for (;;) {
      int p = highest_bit(v, A_WORDS, A_COLS);
      if (p < 0) break;
      if (!used[p]) {
        memcpy(basis[p], v, sizeof(v));
        used[p] = 1;
        ++rank;
        break;
      }
      xor_words(v, basis[p], A_WORDS);
    }
  }
  return rank;
}

static unsigned rank_B_and_kernel(const MatB *b, VecR *kernel) {
  uint64_t basis[A_ROWS][B_WORDS];
  uint64_t comb[A_ROWS][B_WORDS];
  unsigned used[A_ROWS];
  memset(basis, 0, sizeof(basis));
  memset(comb, 0, sizeof(comb));
  memset(used, 0, sizeof(used));
  memset(kernel, 0, sizeof(*kernel));
  int found = 0;
  unsigned rank = 0;

  for (unsigned r = 0; r < A_ROWS; ++r) {
    uint64_t v[B_WORDS], c[B_WORDS];
    memcpy(v, b->row[r], sizeof(v));
    memset(c, 0, sizeof(c));
    c[r >> 6] |= UINT64_C(1) << (r & 63u);
    for (;;) {
      int p = highest_bit(v, B_WORDS, A_ROWS);
      if (p < 0) {
        if (!found) {
          memcpy(kernel->w, c, sizeof(c));
          found = 1;
        }
        break;
      }
      if (!used[p]) {
        memcpy(basis[p], v, sizeof(v));
        memcpy(comb[p], c, sizeof(c));
        used[p] = 1;
        ++rank;
        break;
      }
      xor_words(v, basis[p], B_WORDS);
      xor_words(c, comb[p], B_WORDS);
    }
  }
  return found ? rank : A_ROWS + 1u;
}

static void apply_AT(const MatA *a, const VecR *y, VecC *x) {
  memset(x, 0, sizeof(*x));
  for (unsigned r = 0; r < A_ROWS; ++r)
    if ((y->w[r >> 6] >> (r & 63u)) & 1u)
      xor_words(x->w, a->row[r], A_WORDS);
}

static void apply_A(const MatA *a, const VecC *x, VecR *y) {
  memset(y, 0, sizeof(*y));
  for (unsigned r = 0; r < A_ROWS; ++r)
    if (dot_words(a->row[r], x->w, A_WORDS))
      y->w[r >> 6] |= UINT64_C(1) << (r & 63u);
}

static void apply_B_factorized(const MatA *a, const VecR *y, VecR *out) {
  VecC tmp;
  apply_AT(a, y, &tmp);
  apply_A(a, &tmp, out);
}

static void apply_B_packed(const MatB *b, const VecR *y, VecR *out) {
  memset(out, 0, sizeof(*out));
  for (unsigned r = 0; r < A_ROWS; ++r)
    if (dot_words(b->row[r], y->w, B_WORDS))
      out->w[r >> 6] |= UINT64_C(1) << (r & 63u);
}

static void apply_B_scalar(const MatB *b, const VecR *y, VecR *out) {
  memset(out, 0, sizeof(*out));
  for (unsigned r = 0; r < A_ROWS; ++r) {
    unsigned acc = 0;
    for (unsigned c = 0; c < A_ROWS; ++c) {
      unsigned mb = (unsigned)((b->row[r][c >> 6] >> (c & 63u)) & 1u);
      unsigned yb = (unsigned)((y->w[c >> 6] >> (c & 63u)) & 1u);
      acc ^= mb & yb;
    }
    if (acc) out->w[r >> 6] |= UINT64_C(1) << (r & 63u);
  }
}

static int eqR(const VecR *a, const VecR *b) {
  for (unsigned i = 0; i < B_WORDS; ++i) if (a->w[i] != b->w[i]) return 0;
  return 1;
}

static int zeroC(const VecC *x) {
  for (unsigned i = 0; i < A_WORDS; ++i) if (x->w[i]) return 0;
  return 1;
}

static int zeroR(const VecR *x) {
  for (unsigned i = 0; i < B_WORDS; ++i) if (x->w[i]) return 0;
  return 1;
}

static void seed_vec(VecR *v, uint64_t seed) {
  for (unsigned w = 0; w < B_WORDS; ++w) v->w[w] = mix64(seed ^ (UINT64_C(0x9e3779b97f4a7c15) * (w + 1u)));
  v->w[B_WORDS - 1] &= (UINT64_C(1) << (A_ROWS & 63u)) - 1u;
}

static uint64_t projection_digest(const MatB *b) {
  VecR x[PROJ_WIDTH], y[PROJ_WIDTH], next;
  for (unsigned i = 0; i < PROJ_WIDTH; ++i) {
    seed_vec(&x[i], UINT64_C(0x1111111111111111) * (i + 1u));
    seed_vec(&y[i], UINT64_C(0x2222222222222222) * (i + 1u));
  }
  uint64_t h = UINT64_C(0x6a09e667f3bcc909);
  for (unsigned k = 0; k < PROJECTION_TERMS; ++k) {
    for (unsigned i = 0; i < PROJ_WIDTH; ++i)
      for (unsigned j = 0; j < PROJ_WIDTH; ++j)
        h = mix64(h ^ dot_words(x[i].w, y[j].w, B_WORDS) ^ ((uint64_t)i << 8) ^ ((uint64_t)j << 16) ^ k);
    for (unsigned j = 0; j < PROJ_WIDTH; ++j) {
      apply_B_packed(b, &y[j], &next);
      y[j] = next;
    }
  }
  return h;
}

int main(void) {
  MatA a;
  MatB b;
  build_A(&a);
  build_B(&a, &b);

  unsigned rankA = rank_A(&a);
  VecR kernel;
  unsigned rankB = rank_B_and_kernel(&b, &kernel);
  if (rankA != A_COLS) return 2;
  if (rankB != A_COLS) return 3;
  if (A_ROWS - rankA != 412 || A_ROWS - rankB != 412) return 4;

  VecC atKernel;
  VecR bKernel, factorKernel;
  apply_AT(&a, &kernel, &atKernel);
  apply_B_packed(&b, &kernel, &bKernel);
  apply_B_factorized(&a, &kernel, &factorKernel);
  if (!zeroC(&atKernel)) return 5;
  if (!zeroR(&bKernel) || !zeroR(&factorKernel)) return 6;

  VecR p, s, f, np, ns, nf;
  seed_vec(&p, UINT64_C(0xd15ea5e5b1d1f1e5));
  s = p;
  f = p;
  uint64_t krylov_digest = 0;
  for (unsigned k = 0; k < KRYLOV_STEPS; ++k) {
    apply_B_packed(&b, &p, &np);
    apply_B_scalar(&b, &s, &ns);
    apply_B_factorized(&a, &f, &nf);
    if (!eqR(&np, &ns) || !eqR(&np, &nf)) return 7;
    for (unsigned w = 0; w < B_WORDS; ++w)
      krylov_digest = mix64(krylov_digest ^ np.w[w] ^ ((uint64_t)k << 32) ^ w);
    p = np;
    s = ns;
    f = nf;
  }

  uint64_t proj_digest = projection_digest(&b);

  puts("RSA260_BIDI_CANDIDATE_BWC_SHADOW");
  printf("A=%dx%d rankA=%u left_nullity=%u\n", A_ROWS, A_COLS, rankA, A_ROWS-rankA);
  printf("B=AAT %dx%d rankB=%u nullity=%u\n", A_ROWS, A_ROWS, rankB, A_ROWS-rankB);
  puts("kernel_AT_zero=true kernel_B_zero=true factorized_equals_explicit=true");
  printf("krylov_steps=%d packed_scalar_factorized_agree=true digest=%016llx\n",
         KRYLOV_STEPS, (unsigned long long)krylov_digest);
  printf("projection_width=%d projection_terms=%d digest=%016llx\n",
         PROJ_WIDTH, PROJECTION_TERMS, (unsigned long long)proj_digest);
  puts("prepared_shadow_consumer_pass=true");
  puts("production_prepared_encoding_identity=false");
  puts("production_bwc_replay=false");
  return 0;
}
