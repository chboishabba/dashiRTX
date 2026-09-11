#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Recover a nonzero kernel vector from the shifted degree-17 Krylov relation
 * space of the synthetic 256x256 GF(2) matrix used by the RSA-260 execution
 * ladder.
 *
 * We form columns from K_l = M^l Y for l=1..17 (17*16 = 272 vectors), solve
 * for homogeneous relations among those columns, then shift each relation back
 * one Krylov step.  If
 *
 *   sum_{l=1}^{17} M^l Y a_l = 0,
 *
 * then
 *
 *   v = sum_{l=1}^{17} M^{l-1} Y a_l
 *
 * satisfies Mv=0.  We require v != 0 before accepting the candidate.
 *
 * This is a synthetic-kernel recovery oracle only.  It is NOT the RSA-260
 * matrix, NOT CADO-NFS's production kernel-recovery code, and NOT a production
 * RSA-260 linear-algebra reproduction.
 */

enum { WIDTH=256, WORDS=4, BLOCK=16, DEGREE=17, COLS=BLOCK*DEGREE,
       RWORDS=(COLS+63)/64 };
typedef struct { uint64_t w[WORDS]; } Vec256;
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;
typedef struct { Vec256 col[BLOCK]; } Block256x16;
typedef struct { uint64_t b[RWORDS]; } RelBits;
typedef struct { RelBits lhs; } EqRow;

static unsigned parity64(uint64_t x){ return (unsigned)(__builtin_popcountll((unsigned long long)x)&1u); }
static unsigned get_bit(const Vec256 *v,unsigned i){ return (unsigned)((v->w[i>>6]>>(i&63u))&1u); }
static void set_bit(Vec256 *v,unsigned i){ v->w[i>>6] |= UINT64_C(1)<<(i&63u); }
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
static void block_step(const Mat256*m,const Block256x16*x,Block256x16*y){ for(unsigned j=0;j<BLOCK;++j) matvec(m,&x->col[j],&y->col[j]); }

static unsigned rel_get(const RelBits *r,unsigned c){ return (unsigned)((r->b[c>>6]>>(c&63u))&1u); }
static void rel_set(RelBits *r,unsigned c){ r->b[c>>6]|=UINT64_C(1)<<(c&63u); }
static void rel_xor(RelBits *a,const RelBits *b){ for(unsigned w=0;w<RWORDS;++w) a->b[w]^=b->b[w]; }

static unsigned rref(EqRow *rows,unsigned nr,unsigned *pivots){
  unsigned rank=0;
  for(unsigned c=0;c<COLS && rank<nr;++c){
    unsigned p=rank; while(p<nr && !rel_get(&rows[p].lhs,c)) ++p;
    if(p==nr) continue;
    if(p!=rank){ EqRow t=rows[p]; rows[p]=rows[rank]; rows[rank]=t; }
    pivots[rank]=c;
    for(unsigned r=0;r<nr;++r) if(r!=rank && rel_get(&rows[r].lhs,c)) rel_xor(&rows[r].lhs,&rows[rank].lhs);
    ++rank;
  }
  return rank;
}

static void relation_from_free(const EqRow *rows,const unsigned *pivots,unsigned rank,unsigned free_col,RelBits *rel){
  memset(rel,0,sizeof(*rel)); rel_set(rel,free_col);
  for(unsigned r=0;r<rank;++r) if(rel_get(&rows[r].lhs,free_col)) rel_set(rel,pivots[r]);
}

static void relation_vector(const Block256x16 *K,const RelBits *rel,unsigned shift,Vec256 *out){
  memset(out,0,sizeof(*out));
  unsigned c=0;
  for(unsigned l=1;l<=DEGREE;++l) for(unsigned p=0;p<BLOCK;++p,++c)
    if(rel_get(rel,c)) xor_vec(out,&K[l-shift].col[p]);
}

static unsigned matrix_rank(Mat256 *m){
  Vec256 rows[WIDTH];
  for(unsigned r=0;r<WIDTH;++r) for(unsigned w=0;w<WORDS;++w) rows[r].w[w]=m->row[r][w];
  unsigned rank=0;
  for(unsigned c=0;c<WIDTH && rank<WIDTH;++c){
    unsigned p=rank; while(p<WIDTH && !get_bit(&rows[p],c)) ++p;
    if(p==WIDTH) continue;
    if(p!=rank){ Vec256 t=rows[p]; rows[p]=rows[rank]; rows[rank]=t; }
    for(unsigned r=0;r<WIDTH;++r) if(r!=rank && get_bit(&rows[r],c)) xor_vec(&rows[r],&rows[rank]);
    ++rank;
  }
  return rank;
}

int main(void){
  Mat256 M; Block256x16 K[DEGREE+1];
  build_sparse_matrix(&M,UINT64_C(0x6a09e667f3bcc909));
  build_block(&K[0],UINT64_C(0x3c6ef372fe94f82b));
  for(unsigned l=0;l<DEGREE;++l) block_step(&M,&K[l],&K[l+1]);

  EqRow rows[WIDTH]; memset(rows,0,sizeof(rows));
  unsigned col=0;
  for(unsigned l=1;l<=DEGREE;++l) for(unsigned p=0;p<BLOCK;++p,++col)
    for(unsigned r=0;r<WIDTH;++r) if(get_bit(&K[l].col[p],r)) rel_set(&rows[r].lhs,col);

  unsigned pivots[WIDTH]; const unsigned rank=rref(rows,WIDTH,pivots);
  unsigned is_pivot[COLS]; memset(is_pivot,0,sizeof(is_pivot));
  for(unsigned r=0;r<rank;++r) is_pivot[pivots[r]]=1u;

  unsigned relation_count=0, nonzero_kernel_relations=0, chosen_free=COLS;
  Vec256 chosen={{0,0,0,0}}; RelBits chosen_rel; memset(&chosen_rel,0,sizeof(chosen_rel));
  for(unsigned f=0;f<COLS;++f) if(!is_pivot[f]){
    RelBits rel; Vec256 zero_check,v,mv;
    relation_from_free(rows,pivots,rank,f,&rel);
    relation_vector(K,&rel,0,&zero_check); /* sum M^l Y a_l */
    if(!zero_vec(&zero_check)){
      fprintf(stderr,"rsa260_krylov_kernel_recovery_oracle: FAIL relation free=%u\n",f); return EXIT_FAILURE;
    }
    relation_vector(K,&rel,1,&v); /* shifted: sum M^(l-1)Y a_l */
    matvec(&M,&v,&mv);
    if(!zero_vec(&mv)){
      fprintf(stderr,"rsa260_krylov_kernel_recovery_oracle: FAIL kernel free=%u\n",f); return EXIT_FAILURE;
    }
    ++relation_count;
    if(!zero_vec(&v)){
      ++nonzero_kernel_relations;
      if(chosen_free==COLS){ chosen_free=f; chosen=v; chosen_rel=rel; }
    }
  }

  if(chosen_free==COLS){ fprintf(stderr,"rsa260_krylov_kernel_recovery_oracle: FAIL no nonzero kernel candidate\n"); return EXIT_FAILURE; }
  unsigned candidate_weight=0, relation_weight=0;
  for(unsigned w=0;w<WORDS;++w) candidate_weight+=(unsigned)__builtin_popcountll((unsigned long long)chosen.w[w]);
  for(unsigned w=0;w<RWORDS;++w) relation_weight+=(unsigned)__builtin_popcountll((unsigned long long)chosen_rel.b[w]);

  const unsigned mrank=matrix_rank(&M);
  uint64_t digest=UINT64_C(0x9e3779b97f4a7c15);
  for(unsigned w=0;w<WORDS;++w) digest=mix64(digest^chosen.w[w]^((uint64_t)w<<48));

  printf("rsa260_krylov_kernel_recovery_oracle: ok; matrix_rank=%u; nullity=%u; shifted_columns=%u; relation_space_dim=%u; nonzero_kernel_relations=%u; chosen_free=%u; relation_weight=%u; candidate_weight=%u; digest=%016llx\n",
         mrank,WIDTH-mrank,COLS,COLS-rank,nonzero_kernel_relations,chosen_free,relation_weight,candidate_weight,(unsigned long long)digest);
  return EXIT_SUCCESS;
}
