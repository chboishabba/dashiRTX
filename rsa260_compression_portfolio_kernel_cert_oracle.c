#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { N = 256, C3Q = 88, V4Q = 76 };

typedef struct { uint8_t a[N][N]; } Mat256;
typedef struct { uint8_t a[C3Q][C3Q]; } Mat88;
typedef struct { uint8_t a[V4Q][V4Q]; } Mat76;

static unsigned rank_rect(uint8_t *a, unsigned n, unsigned stride) {
  unsigned r = 0;
  for (unsigned c = 0; c < n && r < n; ++c) {
    unsigned p = r;
    while (p < n && a[p * stride + c] == 0) ++p;
    if (p == n) continue;
    if (p != r) {
      for (unsigned j = c; j < n; ++j) {
        uint8_t t = a[r * stride + j];
        a[r * stride + j] = a[p * stride + j];
        a[p * stride + j] = t;
      }
    }
    for (unsigned i = 0; i < n; ++i) if (i != r && a[i * stride + c])
      for (unsigned j = c; j < n; ++j) a[i * stride + j] ^= a[r * stride + j];
    ++r;
  }
  return r;
}

static unsigned rank256(const Mat256 *m) { Mat256 t = *m; return rank_rect(&t.a[0][0], N, N); }
static unsigned rank88(const Mat88 *m) { Mat88 t = *m; return rank_rect(&t.a[0][0], C3Q, C3Q); }
static unsigned rank76(const Mat76 *m) { Mat76 t = *m; return rank_rect(&t.a[0][0], V4Q, V4Q); }

static void build_c3(Mat256 *m) {
  static const unsigned lens[7] = {9,10,11,12,13,14,15};
  memset(m, 0, sizeof(*m));
  unsigned o = 0;
  for (unsigned g = 0; g < 7; ++g) {
    for (unsigned j = 0; j < lens[g]; ++j, ++o) {
      for (unsigned p = 0; p < 3; ++p) {
        unsigned r = 3 * o + p;
        m->a[r][r] = 1;
        if (j + 1 < lens[g]) m->a[r][3 * (o + 1) + p] = 1;
      }
    }
  }
}

static void build_v4(Mat256 *m) {
  memset(m, 0, sizeof(*m));
  for (unsigned c = 0; c < 60; ++c) {
    unsigned b = 4 * c;
    for (unsigned i = 0; i < 4; ++i)
      for (unsigned j = 0; j < 4; ++j)
        if (i != j) m->a[b+i][b+j] = 1;
    if ((c % 20) + 1 < 20)
      for (unsigned i = 0; i < 4; ++i) m->a[b+i][b+4+i] = 1;
  }
  for (unsigned i = 240; i + 1 < N; ++i) m->a[i][i+1] = 1;
}

static unsigned c3_lift_coord(unsigned q, unsigned p) {
  if (q < 84) return 3 * q + p;
  return 252 + (q - 84);
}

static unsigned v4_lift_coord(unsigned q, unsigned p) {
  if (q < 60) return 4 * q + p;
  return 240 + (q - 60);
}

static int certify_c3(void) {
  Mat256 m; Mat88 q; memset(&q, 0, sizeof q); build_c3(&m);
  /* quotient is read from any representative; equality across each orbit is checked below */
  for (unsigned i = 0; i < C3Q; ++i) for (unsigned j = 0; j < C3Q; ++j) {
    unsigned ri = c3_lift_coord(i, 0);
    unsigned cj = c3_lift_coord(j, 0);
    uint8_t v = 0;
    if (j < 84) for (unsigned p = 0; p < 3; ++p) v ^= m.a[ri][c3_lift_coord(j,p)];
    else v = m.a[ri][cj];
    q.a[i][j] = v;
  }
  unsigned checks = 0;
  for (unsigned j = 0; j < C3Q; ++j) {
    uint8_t x[N] = {0}, mx[N] = {0}, y[C3Q] = {0}, ly[N] = {0};
    if (j < 84) for (unsigned p = 0; p < 3; ++p) x[c3_lift_coord(j,p)] = 1;
    else x[c3_lift_coord(j,0)] = 1;
    for (unsigned r = 0; r < N; ++r) for (unsigned c = 0; c < N; ++c) mx[r] ^= m.a[r][c] & x[c];
    for (unsigned i = 0; i < C3Q; ++i) y[i] = q.a[i][j];
    for (unsigned i = 0; i < C3Q; ++i) {
      if (i < 84) for (unsigned p = 0; p < 3; ++p) ly[c3_lift_coord(i,p)] = y[i];
      else ly[c3_lift_coord(i,0)] = y[i];
    }
    for (unsigned r = 0; r < N; ++r) { if (mx[r] != ly[r]) return 10; ++checks; }
  }
  unsigned rf = rank256(&m), rq = rank88(&q);
  unsigned nf = N - rf, nq = C3Q - rq;
  uint8_t z[C3Q] = {0}, lz[N] = {0}, mlz[N] = {0}; z[84] = 1; lz[252] = 1;
  for (unsigned r = 0; r < N; ++r) for (unsigned c = 0; c < N; ++c) mlz[r] ^= m.a[r][c] & lz[c];
  unsigned weight = 0; for (unsigned i = 0; i < N; ++i) { weight += lz[i]; if (mlz[i]) return 11; }
  printf("c3 full_rank=%u full_nullity=%u quotient_rank=%u quotient_nullity=%u intertwining_checks=%u kernel_weight=%u\n", rf,nf,rq,nq,checks,weight);
  if (nf != nq || nf != 4 || weight != 1) return 12;
  return 0;
}

static int certify_v4(void) {
  Mat256 m; Mat76 q; memset(&q, 0, sizeof q); build_v4(&m);
  for (unsigned i = 0; i < V4Q; ++i) for (unsigned j = 0; j < V4Q; ++j) {
    unsigned ri = v4_lift_coord(i, 0);
    uint8_t v = 0;
    if (j < 60) for (unsigned p = 0; p < 4; ++p) v ^= m.a[ri][v4_lift_coord(j,p)];
    else v = m.a[ri][v4_lift_coord(j,0)];
    q.a[i][j] = v;
  }
  unsigned checks = 0;
  for (unsigned j = 0; j < V4Q; ++j) {
    uint8_t x[N] = {0}, mx[N] = {0}, y[V4Q] = {0}, ly[N] = {0};
    if (j < 60) for (unsigned p = 0; p < 4; ++p) x[v4_lift_coord(j,p)] = 1;
    else x[v4_lift_coord(j,0)] = 1;
    for (unsigned r = 0; r < N; ++r) for (unsigned c = 0; c < N; ++c) mx[r] ^= m.a[r][c] & x[c];
    for (unsigned i = 0; i < V4Q; ++i) y[i] = q.a[i][j];
    for (unsigned i = 0; i < V4Q; ++i) {
      if (i < 60) for (unsigned p = 0; p < 4; ++p) ly[v4_lift_coord(i,p)] = y[i];
      else ly[v4_lift_coord(i,0)] = y[i];
    }
    for (unsigned r = 0; r < N; ++r) { if (mx[r] != ly[r]) return 20; ++checks; }
  }
  unsigned rf = rank256(&m), rq = rank76(&q);
  unsigned nf = N - rf, nq = V4Q - rq;
  uint8_t z[V4Q] = {0}, lz[N] = {0}, mlz[N] = {0}; z[60] = 1; lz[240] = 1;
  for (unsigned r = 0; r < N; ++r) for (unsigned c = 0; c < N; ++c) mlz[r] ^= m.a[r][c] & lz[c];
  unsigned weight = 0; for (unsigned i = 0; i < N; ++i) { weight += lz[i]; if (mlz[i]) return 21; }
  printf("v4 full_rank=%u full_nullity=%u quotient_rank=%u quotient_nullity=%u intertwining_checks=%u kernel_weight=%u\n", rf,nf,rq,nq,checks,weight);
  if (nf != nq || nf != 1 || weight != 1) return 22;
  return 0;
}

int main(void) {
  int a = certify_c3(); if (a) return a;
  int b = certify_v4(); if (b) return b;
  puts("ok exact_kernel_portfolio c3=88 v4=76 nullity_preserved=yes lift_verified=yes");
  return 0;
}
