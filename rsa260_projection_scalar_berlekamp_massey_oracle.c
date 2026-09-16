#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Entrywise GF(2) Berlekamp-Massey oracle over a synthetic Block-Wiedemann-
 * shaped projection sequence S_k = X^T M^k Y.
 *
 * This deliberately recovers one scalar recurrence for each of the 16x16
 * projected bit sequences. It does NOT compute the block/matrix minimal
 * generator used by a production Block Wiedemann solver.
 */

enum { WIDTH=256, WORDS=4, BLOCK=16, TERMS=512, MAXC=513 };
typedef struct { uint64_t w[WORDS]; } Vec256;
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;
typedef struct { Vec256 col[BLOCK]; } Block256x16;
typedef struct { uint16_t row[BLOCK]; } Mat16;

static unsigned parity64(uint64_t x){ return (unsigned)(__builtin_popcountll((unsigned long long)x)&1u); }
static void set_bit(Vec256 *v,unsigned i){ v->w[i>>6] |= UINT64_C(1)<<(i&63u); }
static unsigned dot(const Vec256*a,const Vec256*b){ return parity64(a->w[0]&b->w[0])^parity64(a->w[1]&b->w[1])^parity64(a->w[2]&b->w[2])^parity64(a->w[3]&b->w[3]); }
static uint64_t mix64(uint64_t x){ x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9); x^=x>>27; x*=UINT64_C(0x94d049bb133111eb); x^=x>>31; return x; }

static void build_sparse_matrix(Mat256*m,uint64_t seed){
    memset(m,0,sizeof(*m));
    for(unsigned r=0;r<WIDTH;++r) for(unsigned k=0;k<7;++k){
        uint64_t h=mix64(seed^((uint64_t)r<<12)^k); unsigned c=(unsigned)(h&255u);
        m->row[r][c>>6] ^= UINT64_C(1)<<(c&63u);
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

/* Standard Berlekamp-Massey over GF(2). C[0]=1; returns linear complexity L. */
static unsigned bm(const unsigned char *s,unsigned n,unsigned char *C){
    unsigned char B[MAXC]={0},T[MAXC]={0};
    memset(C,0,MAXC); C[0]=1; B[0]=1;
    unsigned L=0,m=1;
    for(unsigned N=0;N<n;++N){
        unsigned d=s[N];
        for(unsigned i=1;i<=L;++i) d ^= C[i] & s[N-i];
        if(!d){ ++m; continue; }
        memcpy(T,C,MAXC);
        for(unsigned j=0;j+m<MAXC;++j) if(B[j]) C[j+m]^=1u;
        if(2u*L<=N){ L=N+1u-L; memcpy(B,T,MAXC); m=1; } else ++m;
    }
    return L;
}
static int recurrence_holds(const unsigned char*s,unsigned n,const unsigned char*C,unsigned L){
    for(unsigned k=L;k<n;++k){ unsigned d=s[k]; for(unsigned i=1;i<=L;++i) d^=C[i]&s[k-i]; if(d) return 0; } return 1;
}
static void fail(unsigned i,unsigned j){ fprintf(stderr,"rsa260_projection_scalar_bm_oracle: FAIL entry=%u,%u\n",i,j); exit(EXIT_FAILURE); }

int main(void){
    Mat256 M; Block256x16 X,Y,N; Mat16 seq[TERMS];
    build_sparse_matrix(&M,UINT64_C(0x6a09e667f3bcc909));
    build_block(&X,UINT64_C(0xbb67ae8584caa73b));
    build_block(&Y,UINT64_C(0x3c6ef372fe94f82b));
    for(unsigned k=0;k<TERMS;++k){
        project(&X,&Y,&seq[k]);
        for(unsigned j=0;j<BLOCK;++j) matvec(&M,&Y.col[j],&N.col[j]);
        Y=N;
    }

    unsigned max_degree=0,min_degree=TERMS,sum_degree=0,validated=0;
    uint64_t degree_digest=UINT64_C(0x243f6a8885a308d3);
    for(unsigned i=0;i<BLOCK;++i) for(unsigned j=0;j<BLOCK;++j){
        unsigned char s[TERMS],C[MAXC];
        for(unsigned k=0;k<TERMS;++k) s[k]=(unsigned char)((seq[k].row[i]>>j)&1u);
        unsigned L=bm(s,TERMS,C);
        if(!recurrence_holds(s,TERMS,C,L)) fail(i,j);
        if(L>max_degree) max_degree=L; if(L<min_degree) min_degree=L; sum_degree+=L; ++validated;
        degree_digest=mix64(degree_digest ^ ((uint64_t)L<<32) ^ ((uint64_t)i<<16) ^ j);
    }
    printf("rsa260_projection_scalar_bm_oracle: ok; entries=%u; terms=%u; min_degree=%u; max_degree=%u; sum_degree=%u; digest=%016llx\n",
           validated,TERMS,min_degree,max_degree,sum_degree,(unsigned long long)degree_digest);
    return EXIT_SUCCESS;
}
