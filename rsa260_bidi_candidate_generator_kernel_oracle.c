#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  A_ROWS=924, A_COLS=512, A_WORDS=8, B_WORDS=15,
  BLOCK=8, TERMS=256, TRAIN_LAST=191, MAXD=80,
  MAXU=BLOCK*MAXD, UWORDS=(MAXU+63)/64,
  MAXCOLS=BLOCK*MAXD, RWORDS=(MAXCOLS+63)/64
};

typedef struct { uint64_t w[A_WORDS]; } VecC;
typedef struct { uint64_t w[B_WORDS]; } VecR;
typedef struct { uint64_t row[A_ROWS][A_WORDS]; } MatA;
typedef struct { VecR col[BLOCK]; } BlockR;
typedef struct { uint8_t row[BLOCK]; } Mat8;
typedef struct { uint64_t bits[UWORDS]; unsigned rhs; } EqRow;
typedef struct { uint64_t b[RWORDS]; } RelBits;
typedef struct { RelBits lhs; } RelEq;

static uint64_t mix64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);x^=x>>31;return x;}
static unsigned parity64(uint64_t x){return (unsigned)(__builtin_popcountll((unsigned long long)x)&1u);}
static unsigned dot_words(const uint64_t*a,const uint64_t*b,unsigned n){unsigned p=0;for(unsigned i=0;i<n;++i)p^=parity64(a[i]&b[i]);return p;}
static void xor_words(uint64_t*d,const uint64_t*s,unsigned n){for(unsigned i=0;i<n;++i)d[i]^=s[i];}
static uint64_t shadow_row_base(unsigned r){return ((uint64_t)r*UINT64_C(2654435761)+UINT64_C(0x9e3779b9))%A_COLS;}
static void build_A(MatA*a){memset(a,0,sizeof(*a));for(unsigned r=0;r<A_ROWS;++r){unsigned deg=r<6?151u:150u;uint64_t base=shadow_row_base(r);for(unsigned j=0;j<deg;++j){unsigned c=(unsigned)((base+j)%A_COLS);a->row[r][c>>6]|=UINT64_C(1)<<(c&63u);}}}
static void apply_AT(const MatA*a,const VecR*y,VecC*x){memset(x,0,sizeof(*x));for(unsigned r=0;r<A_ROWS;++r)if((y->w[r>>6]>>(r&63u))&1u)xor_words(x->w,a->row[r],A_WORDS);}
static void apply_A(const MatA*a,const VecC*x,VecR*y){memset(y,0,sizeof(*y));for(unsigned r=0;r<A_ROWS;++r)if(dot_words(a->row[r],x->w,A_WORDS))y->w[r>>6]|=UINT64_C(1)<<(r&63u);}
static void apply_B(const MatA*a,const VecR*y,VecR*out){VecC t;apply_AT(a,y,&t);apply_A(a,&t,out);}
static int zeroR(const VecR*v){for(unsigned i=0;i<B_WORDS;++i)if(v->w[i])return 0;return 1;}
static int zeroC(const VecC*v){for(unsigned i=0;i<A_WORDS;++i)if(v->w[i])return 0;return 1;}
static void xorR(VecR*a,const VecR*b){xor_words(a->w,b->w,B_WORDS);}
static void seed_vec(VecR*v,uint64_t seed){for(unsigned w=0;w<B_WORDS;++w)v->w[w]=mix64(seed^(UINT64_C(0x9e3779b97f4a7c15)*(w+1u)));v->w[B_WORDS-1]&=(UINT64_C(1)<<(A_ROWS&63u))-1u;}
static void build_block(BlockR*b,uint64_t seed){for(unsigned j=0;j<BLOCK;++j)seed_vec(&b->col[j],seed^((uint64_t)j<<32));}
static void block_step(const MatA*a,const BlockR*x,BlockR*y){for(unsigned j=0;j<BLOCK;++j)apply_B(a,&x->col[j],&y->col[j]);}
static void project(const BlockR*x,const BlockR*y,Mat8*s){for(unsigned i=0;i<BLOCK;++i){uint8_t row=0;for(unsigned j=0;j<BLOCK;++j)row|=(uint8_t)(dot_words(x->col[i].w,y->col[j].w,B_WORDS)<<j);s->row[i]=row;}}
static unsigned mbit(const Mat8*s,unsigned i,unsigned j){return (unsigned)((s->row[i]>>j)&1u);}

static unsigned bit_at(const EqRow*r,unsigned c){return (unsigned)((r->bits[c>>6]>>(c&63u))&1u);}
static void set_eq_bit(EqRow*r,unsigned c){r->bits[c>>6]|=UINT64_C(1)<<(c&63u);}
static void xor_eq(EqRow*a,const EqRow*b,unsigned words){for(unsigned w=0;w<words;++w)a->bits[w]^=b->bits[w];a->rhs^=b->rhs;}
static int solve_gf2(EqRow*rows,unsigned nr,unsigned nu,unsigned char*x){unsigned words=(nu+63u)/64u;unsigned*piv=(unsigned*)malloc(nr*sizeof(unsigned));if(!piv)return 0;for(unsigned i=0;i<nr;++i)piv[i]=nu;unsigned rank=0;for(unsigned c=0;c<nu&&rank<nr;++c){unsigned p=rank;while(p<nr&&!bit_at(&rows[p],c))++p;if(p==nr)continue;if(p!=rank){EqRow t=rows[p];rows[p]=rows[rank];rows[rank]=t;}piv[rank]=c;for(unsigned r=0;r<nr;++r)if(r!=rank&&bit_at(&rows[r],c))xor_eq(&rows[r],&rows[rank],words);++rank;}for(unsigned r=rank;r<nr;++r){int any=0;for(unsigned w=0;w<words;++w)if(rows[r].bits[w]){any=1;break;}if(!any&&rows[r].rhs){free(piv);return 0;}}memset(x,0,nu);for(unsigned r=0;r<rank;++r)if(piv[r]<nu)x[piv[r]]=(unsigned char)(rows[r].rhs&1u);free(piv);return 1;}
static int fit_degree(const Mat8*seq,unsigned d,Mat8*F){if(d>MAXD||TRAIN_LAST<d)return 0;unsigned nu=BLOCK*d,kvals=TRAIN_LAST-d+1u,nr=kvals*BLOCK;for(unsigned l=0;l<d;++l)for(unsigned p=0;p<BLOCK;++p)F[l].row[p]=0;for(unsigned outj=0;outj<BLOCK;++outj){EqRow*eq=(EqRow*)calloc(nr,sizeof(EqRow));unsigned char*sol=(unsigned char*)calloc(nu,1);if(!eq||!sol){free(eq);free(sol);return 0;}unsigned q=0;for(unsigned k=0;k<kvals;++k)for(unsigned i=0;i<BLOCK;++i,++q){for(unsigned l=0;l<d;++l)for(unsigned p=0;p<BLOCK;++p)if(mbit(&seq[k+l],i,p))set_eq_bit(&eq[q],l*BLOCK+p);eq[q].rhs=mbit(&seq[k+d],i,outj);}if(!solve_gf2(eq,nr,nu,sol)){free(eq);free(sol);return 0;}for(unsigned l=0;l<d;++l)for(unsigned p=0;p<BLOCK;++p)if(sol[l*BLOCK+p])F[l].row[p]|=(uint8_t)(1u<<outj);free(eq);free(sol);}return 1;}
static int recurrence_holds(const Mat8*seq,unsigned terms,unsigned d,const Mat8*F,unsigned start){for(unsigned k=start;k+d<terms;++k)for(unsigned i=0;i<BLOCK;++i)for(unsigned j=0;j<BLOCK;++j){unsigned acc=mbit(&seq[k+d],i,j);for(unsigned l=0;l<d;++l)for(unsigned p=0;p<BLOCK;++p)acc^=mbit(&seq[k+l],i,p)&mbit(&F[l],p,j);if(acc)return 0;}return 1;}
static uint64_t digest_generator(const Mat8*F,unsigned d){uint64_t h=UINT64_C(0x13198a2e03707344);for(unsigned l=0;l<d;++l)for(unsigned p=0;p<BLOCK;++p)h=mix64(h^(uint64_t)F[l].row[p]^((uint64_t)l<<40)^((uint64_t)p<<24));return h;}

static unsigned rel_get(const RelBits*r,unsigned c){return (unsigned)((r->b[c>>6]>>(c&63u))&1u);}
static void rel_set(RelBits*r,unsigned c){r->b[c>>6]|=UINT64_C(1)<<(c&63u);}
static void rel_xor(RelBits*a,const RelBits*b,unsigned words){for(unsigned w=0;w<words;++w)a->b[w]^=b->b[w];}
static unsigned rel_rref(RelEq*rows,unsigned nr,unsigned ncols,unsigned*piv){unsigned rank=0,words=(ncols+63u)/64u;for(unsigned c=0;c<ncols&&rank<nr;++c){unsigned p=rank;while(p<nr&&!rel_get(&rows[p].lhs,c))++p;if(p==nr)continue;if(p!=rank){RelEq t=rows[p];rows[p]=rows[rank];rows[rank]=t;}piv[rank]=c;for(unsigned r=0;r<nr;++r)if(r!=rank&&rel_get(&rows[r].lhs,c))rel_xor(&rows[r].lhs,&rows[rank].lhs,words);++rank;}return rank;}
static void relation_from_free(const RelEq*rows,const unsigned*piv,unsigned rank,unsigned freec,RelBits*rel){memset(rel,0,sizeof(*rel));rel_set(rel,freec);for(unsigned r=0;r<rank;++r)if(rel_get(&rows[r].lhs,freec))rel_set(rel,piv[r]);}
static void relation_vector(const BlockR*K,const RelBits*rel,unsigned degree,unsigned shift,VecR*out){memset(out,0,sizeof(*out));unsigned c=0;for(unsigned l=1;l<=degree;++l)for(unsigned p=0;p<BLOCK;++p,++c)if(rel_get(rel,c))xorR(out,&K[l-shift].col[p]);}
static unsigned weightR(const VecR*v){unsigned n=0;for(unsigned w=0;w<B_WORDS;++w)n+=(unsigned)__builtin_popcountll((unsigned long long)v->w[w]);return n;}

int main(void){
  MatA A; build_A(&A);
  BlockR X,Y,N; build_block(&X,UINT64_C(0xbb67ae8584caa73b));build_block(&Y,UINT64_C(0x3c6ef372fe94f82b));
  static Mat8 seq[TERMS];
  for(unsigned k=0;k<TERMS;++k){project(&X,&Y,&seq[k]);block_step(&A,&Y,&N);Y=N;}
  Mat8 F[MAXD];unsigned degree=0;
  for(unsigned d=1;d<=MAXD;++d){if(fit_degree(seq,d,F)&&recurrence_holds(seq,TRAIN_LAST+1u,d,F,0)){degree=d;break;}}
  if(!degree){fprintf(stderr,"FAIL no generator <=%u\n",MAXD);return 2;}
  unsigned holdout_start=TRAIN_LAST+1u-degree;
  if(!recurrence_holds(seq,TERMS,degree,F,holdout_start)){fprintf(stderr,"FAIL holdout d=%u\n",degree);return 3;}

  unsigned ncols=BLOCK*degree;
  BlockR *K=(BlockR*)calloc(degree+1u,sizeof(BlockR));
  if(!K) return 4;
  build_block(&K[0],UINT64_C(0x3c6ef372fe94f82b));
  for(unsigned l=0;l<degree;++l) block_step(&A,&K[l],&K[l+1]);
  RelEq *rows=(RelEq*)calloc(A_ROWS,sizeof(RelEq));unsigned*piv=(unsigned*)malloc(A_ROWS*sizeof(unsigned));unsigned*is_piv=(unsigned*)calloc(ncols,sizeof(unsigned));
  if(!rows||!piv||!is_piv)return 5;
  unsigned col=0;for(unsigned l=1;l<=degree;++l)for(unsigned p=0;p<BLOCK;++p,++col)for(unsigned r=0;r<A_ROWS;++r)if((K[l].col[p].w[r>>6]>>(r&63u))&1u)rel_set(&rows[r].lhs,col);
  unsigned rank=rel_rref(rows,A_ROWS,ncols,piv);for(unsigned r=0;r<rank;++r)is_piv[piv[r]]=1u;
  unsigned relations=0,nonzero=0,chosen=ncols;VecR chosen_v={{0}};RelBits chosen_rel;memset(&chosen_rel,0,sizeof(chosen_rel));
  for(unsigned f=0;f<ncols;++f)if(!is_piv[f]){RelBits rel;VecR z,v,mv;relation_from_free(rows,piv,rank,f,&rel);relation_vector(K,&rel,degree,0,&z);if(!zeroR(&z)){fprintf(stderr,"FAIL relation %u\n",f);return 6;}relation_vector(K,&rel,degree,1,&v);apply_B(&A,&v,&mv);if(!zeroR(&mv)){fprintf(stderr,"FAIL kernel %u\n",f);return 7;}++relations;if(!zeroR(&v)){++nonzero;if(chosen==ncols){chosen=f;chosen_v=v;chosen_rel=rel;}}}
  if(chosen==ncols){fprintf(stderr,"FAIL no nonzero kernel ncols=%u rank=%u relations=%u\n",ncols,rank,relations);return 8;}
  VecC atv;apply_AT(&A,&chosen_v,&atv);if(!zeroC(&atv)){fprintf(stderr,"FAIL lifted AT nonzero\n");return 9;}
  unsigned relw=0;for(unsigned w=0;w<(ncols+63u)/64u;++w)relw+=(unsigned)__builtin_popcountll((unsigned long long)chosen_rel.b[w]);
  uint64_t kd=UINT64_C(0x9e3779b97f4a7c15);for(unsigned w=0;w<B_WORDS;++w)kd=mix64(kd^chosen_v.w[w]^((uint64_t)w<<48));
  printf("RSA260_BIDI_CANDIDATE_GENERATOR_KERNEL ok block=%u terms=%u degree=%u train_rel=%u holdout_rel=%u gen_digest=%016llx shifted_cols=%u rel_space=%u nonzero_kernel_rel=%u chosen=%u rel_weight=%u kernel_weight=%u kernel_digest=%016llx AT_zero=true B_zero=true\n",BLOCK,TERMS,degree,TRAIN_LAST-degree+1u,(TERMS-degree)-holdout_start,(unsigned long long)digest_generator(F,degree),ncols,ncols-rank,nonzero,chosen,relw,weightR(&chosen_v),(unsigned long long)kd);
  free(K);free(rows);free(piv);free(is_piv);return 0;
}
