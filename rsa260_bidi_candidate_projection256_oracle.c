#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { A_ROWS=924, A_COLS=512, A_WORDS=8, B_WORDS=15, WIDTH=256, TERMS=16, SEQUENCES=2 };
typedef struct { uint64_t w[A_WORDS]; } VecC;
typedef struct { uint64_t w[B_WORDS]; } VecR;
typedef struct { uint64_t row[A_ROWS][A_WORDS]; } MatA;
typedef struct { uint64_t row[A_ROWS][B_WORDS]; } MatB;
typedef struct { VecR col[WIDTH]; } Block256;
static uint64_t mix64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);x^=x>>31;return x;}
static unsigned parity64(uint64_t x){return (unsigned)(__builtin_popcountll((unsigned long long)x)&1u);}
static unsigned dot_words(const uint64_t*a,const uint64_t*b,unsigned n){unsigned p=0;for(unsigned i=0;i<n;++i)p^=parity64(a[i]&b[i]);return p;}
static void xor_words(uint64_t*d,const uint64_t*s,unsigned n){for(unsigned i=0;i<n;++i)d[i]^=s[i];}
static uint64_t shadow_row_base(unsigned r){return ((uint64_t)r*UINT64_C(2654435761)+UINT64_C(0x9e3779b9))%A_COLS;}
static void build_A(MatA*a){memset(a,0,sizeof(*a));for(unsigned r=0;r<A_ROWS;++r){unsigned deg=r<6?151u:150u;uint64_t base=shadow_row_base(r);for(unsigned j=0;j<deg;++j){unsigned c=(unsigned)((base+j)%A_COLS);a->row[r][c>>6]|=UINT64_C(1)<<(c&63u);}}}
static void build_B(const MatA*a,MatB*b){memset(b,0,sizeof(*b));for(unsigned r=0;r<A_ROWS;++r)for(unsigned c=0;c<A_ROWS;++c)if(dot_words(a->row[r],a->row[c],A_WORDS))b->row[r][c>>6]|=UINT64_C(1)<<(c&63u);}
static void apply_AT(const MatA*a,const VecR*y,VecC*x){memset(x,0,sizeof(*x));for(unsigned r=0;r<A_ROWS;++r)if((y->w[r>>6]>>(r&63u))&1u)xor_words(x->w,a->row[r],A_WORDS);}
static void apply_A(const MatA*a,const VecC*x,VecR*y){memset(y,0,sizeof(*y));for(unsigned r=0;r<A_ROWS;++r)if(dot_words(a->row[r],x->w,A_WORDS))y->w[r>>6]|=UINT64_C(1)<<(r&63u);}
static void apply_B_factorized(const MatA*a,const VecR*y,VecR*out){VecC t;apply_AT(a,y,&t);apply_A(a,&t,out);}
static void apply_B_explicit(const MatB*b,const VecR*y,VecR*out){memset(out,0,sizeof(*out));for(unsigned r=0;r<A_ROWS;++r)if(dot_words(b->row[r],y->w,B_WORDS))out->w[r>>6]|=UINT64_C(1)<<(r&63u);}
static int eqR(const VecR*a,const VecR*b){for(unsigned w=0;w<B_WORDS;++w)if(a->w[w]!=b->w[w])return 0;return 1;}
static void seed_vec(VecR*v,uint64_t seed){for(unsigned w=0;w<B_WORDS;++w)v->w[w]=mix64(seed^(UINT64_C(0x9e3779b97f4a7c15)*(w+1u)));v->w[B_WORDS-1]&=(UINT64_C(1)<<(A_ROWS&63u))-1u;}
static void build_block(Block256*b,uint64_t seed){for(unsigned j=0;j<WIDTH;++j)seed_vec(&b->col[j],seed^((uint64_t)j<<32));}
static uint64_t projection_digest(const Block256*x,const Block256*y,uint64_t h,unsigned seq,unsigned term){for(unsigned i=0;i<WIDTH;++i)for(unsigned j=0;j<WIDTH;++j){unsigned bit=dot_words(x->col[i].w,y->col[j].w,B_WORDS);h=mix64(h^(uint64_t)bit^((uint64_t)seq<<60)^((uint64_t)term<<48)^((uint64_t)i<<24)^j);}return h;}
int main(void){MatA A;MatB B;build_A(&A);build_B(&A,&B);uint64_t dig[SEQUENCES]={0};uint64_t vector_steps=0;for(unsigned s=0;s<SEQUENCES;++s){Block256 X,Y,N;build_block(&X,UINT64_C(0xbb67ae8584caa73b)^((uint64_t)s<<56));build_block(&Y,UINT64_C(0x3c6ef372fe94f82b)^((uint64_t)s<<56));uint64_t h=UINT64_C(0x6a09e667f3bcc909)^s;for(unsigned k=0;k<TERMS;++k){h=projection_digest(&X,&Y,h,s,k);for(unsigned j=0;j<WIDTH;++j){VecR e,f;apply_B_explicit(&B,&Y.col[j],&e);apply_B_factorized(&A,&Y.col[j],&f);if(!eqR(&e,&f)){fprintf(stderr,"FAIL seq=%u term=%u col=%u\n",s,k,j);return 2;}N.col[j]=e;++vector_steps;}Y=N;}dig[s]=h;}printf("RSA260_BIDI_CANDIDATE_PROJECTION256 ok sequences=%u width=%u total_block_columns=%u terms=%u vector_steps=%llu explicit_factorized_agree=true digest0=%016llx digest1=%016llx production_identity=false\n",SEQUENCES,WIDTH,SEQUENCES*WIDTH,TERMS,(unsigned long long)vector_steps,(unsigned long long)dig[0],(unsigned long long)dig[1]);return 0;}
