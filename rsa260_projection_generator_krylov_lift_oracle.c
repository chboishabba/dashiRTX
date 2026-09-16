#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Lift the synthetic degree-d right matrix recurrence from the projected
 * Block-Wiedemann-shaped sequence back to the 256-dimensional Krylov carrier.
 *
 * If
 *   S_k = X^T M^k Y
 * and
 *   S_{k+d} + sum_l S_{k+l} F_l = 0,
 * define the vector-block residual
 *   W = M^d Y + sum_l M^l Y F_l.
 *
 * Projection recurrence only proves X^T M^k W = 0 on the observed window.
 * This oracle separately tests the stronger statement W = 0 on the synthetic
 * 256-dimensional carrier.  It is NOT the RSA-260 matrix or production run.
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
static unsigned get_bit(const Vec256 *v,unsigned i){ return (unsigned)((v->w[i>>6]>>(i&63u))&1u); }
static unsigned dot(const Vec256*a,const Vec256*b){ return parity64(a->w[0]&b->w[0])^parity64(a->w[1]&b->w[1])^parity64(a->w[2]&b->w[2])^parity64(a->w[3]&b->w[3]); }
static uint64_t mix64(uint64_t x){ x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9); x^=x>>27; x*=UINT64_C(0x94d049bb133111eb); x^=x>>31; return x; }
static void xor_vec(Vec256 *a,const Vec256 *b){ for(unsigned w=0;w<WORDS;++w) a->w[w]^=b->w[w]; }
static int zero_vec(const Vec256 *v){ return !(v->w[0]|v->w[1]|v->w[2]|v->w[3]); }

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
static void block_step(const Mat256 *m,const Block256x16 *x,Block256x16 *y){ for(unsigned j=0;j<BLOCK;++j) matvec(m,&x->col[j],&y->col[j]); }
static void project(const Block256x16*x,const Block256x16*y,Mat16*s){
  for(unsigned i=0;i<BLOCK;++i){ uint16_t row=0; for(unsigned j=0;j<BLOCK;++j) row|=(uint16_t)(dot(&x->col[i],&y->col[j])<<j); s->row[i]=row; }
}
static unsigned mbit(const Mat16 *s,unsigned i,unsigned j){ return (unsigned)((s->row[i]>>j)&1u); }
static void xor_row(EqRow *a,const EqRow *b,unsigned words){ for(unsigned w=0;w<words;++w) a->bits[w]^=b->bits[w]; a->rhs^=b->rhs; }
static unsigned bit_at(const EqRow*r,unsigned c){ return (unsigned)((r->bits[c>>6]>>(c&63u))&1u); }
static void set_eq_bit(EqRow*r,unsigned c){ r->bits[c>>6]|=UINT64_C(1)<<(c&63u); }

static int solve_gf2(EqRow *rows,unsigned nr,unsigned nu,unsigned char *x){
  const unsigned words=(nu+63u)/64u;
  unsigned *pivot_col=(unsigned*)malloc(nr*sizeof(unsigned)); if(!pivot_col) return 0;
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
  for(unsigned r=rank;r<nr;++r){ int any=0; for(unsigned w=0;w<words;++w) if(rows[r].bits[w]){any=1;break;} if(!any&&rows[r].rhs){free(pivot_col);return 0;} }
  memset(x,0,nu);
  for(unsigned r=0;r<rank;++r) if(pivot_col[r]<nu) x[pivot_col[r]]=(unsigned char)(rows[r].rhs&1u);
  free(pivot_col); return 1;
}
static int fit_degree(const Mat16 *seq,unsigned d,unsigned train_last,Mat16 *F){
  const unsigned nu=BLOCK*d, kvals=train_last-d+1u, nr=kvals*BLOCK;
  if(nu>MAXU || train_last<d) return 0;
  memset(F,0,MAXD*sizeof(Mat16));
  for(unsigned outj=0;outj<BLOCK;++outj){
    EqRow *eq=(EqRow*)calloc(nr,sizeof(EqRow)); unsigned char *sol=(unsigned char*)calloc(nu,1);
    if(!eq||!sol){free(eq);free(sol);return 0;}
    unsigned q=0;
    for(unsigned k=0;k<kvals;++k) for(unsigned i=0;i<BLOCK;++i,++q){
      for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p) if(mbit(&seq[k+l],i,p)) set_eq_bit(&eq[q],l*BLOCK+p);
      eq[q].rhs=mbit(&seq[k+d],i,outj);
    }
    if(!solve_gf2(eq,nr,nu,sol)){free(eq);free(sol);return 0;}
    for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p) if(sol[l*BLOCK+p]) F[l].row[p]|=(uint16_t)(1u<<outj);
    free(eq); free(sol);
  }
  return 1;
}
static int recurrence_holds(const Mat16 *seq,unsigned terms,unsigned d,const Mat16 *F,unsigned start_k){
  for(unsigned k=start_k;k+d<terms;++k) for(unsigned i=0;i<BLOCK;++i) for(unsigned j=0;j<BLOCK;++j){
    unsigned acc=mbit(&seq[k+d],i,j);
    for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p) acc^=mbit(&seq[k+l],i,p)&mbit(&F[l],p,j);
    if(acc) return 0;
  }
  return 1;
}

static void form_krylov_residual(const Block256x16 *K,unsigned d,const Mat16 *F,Block256x16 *W){
  *W=K[d];
  for(unsigned l=0;l<d;++l) for(unsigned p=0;p<BLOCK;++p) for(unsigned j=0;j<BLOCK;++j)
    if(mbit(&F[l],p,j)) xor_vec(&W->col[j],&K[l].col[p]);
}

static unsigned residual_rank(Block256x16 *w){
  Vec256 rows[BLOCK]; for(unsigned i=0;i<BLOCK;++i) rows[i]=w->col[i];
  unsigned rank=0;
  for(unsigned c=0;c<WIDTH && rank<BLOCK;++c){
    unsigned p=rank; while(p<BLOCK && !get_bit(&rows[p],c)) ++p;
    if(p==BLOCK) continue;
    if(p!=rank){Vec256 t=rows[p];rows[p]=rows[rank];rows[rank]=t;}
    for(unsigned r=0;r<BLOCK;++r) if(r!=rank && get_bit(&rows[r],c)) xor_vec(&rows[r],&rows[rank]);
    ++rank;
  }
  return rank;
}

int main(void){
  Mat256 M; Block256x16 X,Y,N; static Mat16 seq[TERMS]; Mat16 F[MAXD];
  build_sparse_matrix(&M,UINT64_C(0x6a09e667f3bcc909));
  build_block(&X,UINT64_C(0xbb67ae8584caa73b));
  build_block(&Y,UINT64_C(0x3c6ef372fe94f82b));

  Block256x16 K[MAXD+1]; K[0]=Y;
  for(unsigned l=0;l<MAXD;++l){ block_step(&M,&K[l],&K[l+1]); }

  Block256x16 Ys=Y;
  for(unsigned k=0;k<TERMS;++k){ project(&X,&Ys,&seq[k]); block_step(&M,&Ys,&N); Ys=N; }

  const unsigned train_last=383; unsigned degree=0;
  for(unsigned d=1;d<=MAXD;++d) if(fit_degree(seq,d,train_last,F)&&recurrence_holds(seq,train_last+1u,d,F,0)){degree=d;break;}
  if(!degree){fprintf(stderr,"rsa260_projection_generator_krylov_lift_oracle: FAIL no generator\n");return EXIT_FAILURE;}

  Block256x16 W; form_krylov_residual(K,degree,F,&W);
  unsigned zero_columns=0; uint64_t residual_weight=0;
  for(unsigned j=0;j<BLOCK;++j){ if(zero_vec(&W.col[j])) ++zero_columns; for(unsigned q=0;q<WORDS;++q) residual_weight+=(uint64_t)__builtin_popcountll((unsigned long long)W.col[j].w[q]); }
  unsigned rank=residual_rank(&W);

  /* Also verify the projected residual vanishes for all available compatible k. */
  unsigned projected_checks=0;
  for(unsigned k=0;k+degree<TERMS;++k){
    for(unsigned i=0;i<BLOCK;++i) for(unsigned j=0;j<BLOCK;++j){
      unsigned acc=mbit(&seq[k+degree],i,j);
      for(unsigned l=0;l<degree;++l) for(unsigned p=0;p<BLOCK;++p) acc^=mbit(&seq[k+l],i,p)&mbit(&F[l],p,j);
      if(acc){fprintf(stderr,"rsa260_projection_generator_krylov_lift_oracle: FAIL projected k=%u i=%u j=%u\n",k,i,j);return EXIT_FAILURE;}
      ++projected_checks;
    }
  }

  printf("rsa260_projection_generator_krylov_lift_oracle: ok; degree=%u; zero_residual_columns=%u; residual_rank=%u; residual_weight=%llu; projected_checks=%u; vector_relation=%s\n",
         degree,zero_columns,rank,(unsigned long long)residual_weight,projected_checks,
         zero_columns==BLOCK?"yes":"no");
  return EXIT_SUCCESS;
}
