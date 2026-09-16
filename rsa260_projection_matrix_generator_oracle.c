#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Synthetic right matrix-polynomial recurrence oracle for the Block-Wiedemann-
 * shaped sequence S_k = X^T M^k Y over GF(2).
 *
 * We search for the smallest degree d (up to MAXD) admitting matrices F_l,
 * l=0..d-1, such that
 *
 *   S_{k+d} + sum_{l=0}^{d-1} S_{k+l} F_l = 0
 *
 * on a training prefix.  Each output column is an independent GF(2) linear
 * system with 16*d unknowns.  The recovered recurrence is then checked on all
 * remaining withheld terms.
 *
 * This is a synthetic block-generator oracle.  It is NOT CADO-NFS's production
 * minimal-generator implementation, NOT the RSA-260 matrix, and NOT a proof
 * that the recovered degree is the production Block Wiedemann degree.
 */

enum { WIDTH=256, WORDS=4, BLOCK=16, TERMS=512, MAXD=32, MAXU=BLOCK*MAXD,
       UWORDS=(MAXU+63)/64 };
typedef struct { uint64_t w[WORDS]; } Vec256;
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;
typedef struct { Vec256 col[BLOCK]; } Block256x16;
typedef struct { uint16_t row[BLOCK]; } Mat16;
typedef struct { uint64_t bits[UWORDS]; unsigned rhs; } EqRow;

static unsigned parity64(uint64_t x){ return (unsigned)(__builtin_popcountll((unsigned long long)x)&1u); }
static void set_bit(Vec256 *v,unsigned i){ v->w[i>>6] |= UINT64_C(1)<<(i&63u); }
static unsigned dot(const Vec256*a,const Vec256*b){ return parity64(a->w[0]&b->w[0])^parity64(a->w[1]&b->w[1])^parity64(a->w[2]&b->w[2])^parity64(a->w[3]&b->w[3]); }
static uint64_t mix64(uint64_t x){ x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9); x^=x>>27; x*=UINT64_C(0x94d049bb133111eb); x^=x>>31; return x; }
static void build_sparse_matrix(Mat256*m,uint64_t seed){
  memset(m,0,sizeof(*m));
  for(unsigned r=0;r<WIDTH;++r) for(unsigned q=0;q<7;++q){
    uint64_t h=mix64(seed^((uint64_t)r<<12)^q); unsigned c=(unsigned)(h&255u);
    m->row[r][c>>6]^=UINT64_C(1)<<(c&63u);
  }
}
static void build_block(Block256x16*b,uint64_t seed){
  for(unsigned j=0;j<BLOCK;++j) for(unsigned w=0;w<WORDS;++w)
    b->col[j].w[w]=mix64(seed^((uint64_t)j<<16)^(UINT64_C(0x9e3779b97f4a7c15)*(w+1u)));
}
static void matvec(const Mat256*m,const Vec256*x,Vec256*y){
  memset(y,0,sizeof(*y));
  for(unsigned r=0;r<WIDTH;++r){ Vec256 rr={{m->row[r][0],m->row[r][1],m->row[r][2],m->row[r][3]}}; if(dot(&rr,x)) set_bit(y,r); }
}
static void project(const Block256x16*x,const Block256x16*y,Mat16*s){
  for(unsigned i=0;i<BLOCK;++i){ uint16_t row=0; for(unsigned j=0;j<BLOCK;++j) row|=(uint16_t)(dot(&x->col[i],&y->col[j])<<j); s->row[i]=row; }
}
static unsigned mbit(const Mat16 *s,unsigned i,unsigned j){ return (unsigned)((s->row[i]>>j)&1u); }
static void xor_row(EqRow *a,const EqRow *b,unsigned words){ for(unsigned w=0;w<words;++w) a->bits[w]^=b->bits[w]; a->rhs^=b->rhs; }
static unsigned bit_at(const EqRow*r,unsigned c){ return (unsigned)((r->bits[c>>6]>>(c&63u))&1u); }
static void set_eq_bit(EqRow*r,unsigned c){ r->bits[c>>6]|=UINT64_C(1)<<(c&63u); }

/* Solve A x=b over GF(2); free variables are fixed to zero. */
static int solve_gf2(EqRow *rows,unsigned nr,unsigned nu,unsigned char *x){
  const unsigned words=(nu+63u)/64u;
  unsigned *pivot_col=(unsigned*)malloc(nr*sizeof(unsigned));
  if(!pivot_col) return 0;
  for(unsigned i=0;i<nr;++i) pivot_col[i]=nu;
  unsigned rank=0;
  for(unsigned c=0;c<nu && rank<nr;++c){
    unsigned p=rank; while(p<nr && !bit_at(&rows[p],c)) ++p;
    if(p==nr) continue;
    if(p!=rank){ EqRow t=rows[p]; rows[p]=rows[rank]; rows[rank]=t; }
    pivot_col[rank]=c;
    for(unsigned r=0;r<nr;++r) if(r!=rank && bit_at(&rows[r],c)) xor_row(&rows[r],&rows[rank],words);
    ++rank;
  }
  for(unsigned r=rank;r<nr;++r){
    int any=0; for(unsigned w=0;w<words;++w) if(rows[r].bits[w]){ any=1; break; }
    if(!any && rows[r].rhs){ free(pivot_col); return 0; }
  }
  memset(x,0,nu);
  for(unsigned r=0;r<rank;++r) if(pivot_col[r]<nu) x[pivot_col[r]]=(unsigned char)(rows[r].rhs&1u);
  free(pivot_col); return 1;
}

/* F[l].row[p] bit j is F_l[p,j]. */
static int fit_degree(const Mat16 *seq,unsigned d,unsigned train_last,Mat16 *F){
  const unsigned nu=BLOCK*d;
  const unsigned kvals=train_last-d+1u;
  const unsigned nr=kvals*BLOCK;
  if(nu>MAXU || train_last<d) return 0;
  for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p) F[l].row[p]=0;
  for(unsigned outj=0;outj<BLOCK;++outj){
    EqRow *eq=(EqRow*)calloc(nr,sizeof(EqRow));
    unsigned char *sol=(unsigned char*)calloc(nu,1);
    if(!eq||!sol){ free(eq); free(sol); return 0; }
    unsigned q=0;
    for(unsigned k=0;k<kvals;++k) for(unsigned i=0;i<BLOCK;++i,++q){
      for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p)
        if(mbit(&seq[k+l],i,p)) set_eq_bit(&eq[q],l*BLOCK+p);
      eq[q].rhs=mbit(&seq[k+d],i,outj);
    }
    if(!solve_gf2(eq,nr,nu,sol)){ free(eq); free(sol); return 0; }
    for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p)
      if(sol[l*BLOCK+p]) F[l].row[p]|=(uint16_t)(1u<<outj);
    free(eq); free(sol);
  }
  return 1;
}
static int recurrence_holds(const Mat16 *seq,unsigned terms,unsigned d,const Mat16 *F,unsigned start_k){
  for(unsigned k=start_k;k+d<terms;++k){
    for(unsigned i=0;i<BLOCK;++i) for(unsigned j=0;j<BLOCK;++j){
      unsigned acc=mbit(&seq[k+d],i,j);
      for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p)
        acc ^= mbit(&seq[k+l],i,p) & mbit(&F[l],p,j);
      if(acc) return 0;
    }
  }
  return 1;
}
static uint64_t digest_generator(const Mat16 *F,unsigned d){
  uint64_t h=UINT64_C(0x13198a2e03707344);
  for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p)
    h=mix64(h ^ (uint64_t)F[l].row[p] ^ ((uint64_t)l<<40) ^ ((uint64_t)p<<24));
  return h;
}

int main(void){
  Mat256 M; Block256x16 X,Y,N; static Mat16 seq[TERMS]; Mat16 F[MAXD];
  build_sparse_matrix(&M,UINT64_C(0x6a09e667f3bcc909));
  build_block(&X,UINT64_C(0xbb67ae8584caa73b));
  build_block(&Y,UINT64_C(0x3c6ef372fe94f82b));
  for(unsigned k=0;k<TERMS;++k){ project(&X,&Y,&seq[k]); for(unsigned j=0;j<BLOCK;++j) matvec(&M,&Y.col[j],&N.col[j]); Y=N; }

  /* Train through term 383; reserve 128 terms for holdout validation. */
  const unsigned train_last=383;
  unsigned degree=0;
  for(unsigned d=1;d<=MAXD;++d){
    if(fit_degree(seq,d,train_last,F) && recurrence_holds(seq,train_last+1u,d,F,0)){
      degree=d; break;
    }
  }
  if(!degree){ fprintf(stderr,"rsa260_projection_matrix_generator_oracle: FAIL no degree <= %u\n",MAXD); return EXIT_FAILURE; }
  const unsigned holdout_start=train_last+1u-degree;
  if(!recurrence_holds(seq,TERMS,degree,F,holdout_start)){
    fprintf(stderr,"rsa260_projection_matrix_generator_oracle: FAIL holdout degree=%u\n",degree); return EXIT_FAILURE;
  }
  const unsigned train_relations=train_last-degree+1u;
  const unsigned holdout_relations=(TERMS-degree)-holdout_start;
  printf("rsa260_projection_matrix_generator_oracle: ok; block=%u; terms=%u; degree=%u; train_relations=%u; holdout_relations=%u; unknowns_per_column=%u; digest=%016llx\n",
         BLOCK,TERMS,degree,train_relations,holdout_relations,BLOCK*degree,
         (unsigned long long)digest_generator(F,degree));
  return EXIT_SUCCESS;
}
