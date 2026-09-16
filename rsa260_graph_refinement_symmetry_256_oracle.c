#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Graph-derived symmetry candidate oracle for the RSA-260 research lane.
 *
 * Treat a square GF(2) matrix as a directed bipartite incidence object under a
 * simultaneous coordinate permutation.  Coordinates are refined by multiple
 * structural consumers:
 *
 *   row degree, column degree, diagonal bit,
 *   multiset of outgoing-neighbour colours,
 *   multiset of incoming-neighbour colours.
 *
 * Equal refinement colour is ONLY a candidate-equivalence relation.  Every
 * proposed permutation is still checked by the exact operator condition
 *
 *   M[p(r),p(c)] = M[r,c]  for all r,c,
 *
 * equivalent to MP = PM for the permutation matrix P.
 *
 * Two matrices are tested:
 *   (1) the same generic synthetic 256x256 matrix used by the Krylov lane;
 *   (2) the declared opposite-pair-equivariant benchmark M = Q tensor I_2.
 *
 * This is not an RSA-260 production matrix claim.
 */

enum { N=256, WORDS=4, MAX_NEIGH=N };
typedef struct { uint64_t row[N][WORDS]; } Mat256;

typedef struct {
  uint32_t oldc;
  uint32_t rowdeg;
  uint32_t coldeg;
  uint32_t diag;
  uint64_t out_hash;
  uint64_t in_hash;
} Signature;

static uint64_t mix64(uint64_t x){
  x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9);
  x^=x>>27; x*=UINT64_C(0x94d049bb133111eb);
  x^=x>>31; return x;
}

static unsigned bit(const Mat256*m,unsigned r,unsigned c){
  return (unsigned)((m->row[r][c>>6]>>(c&63u))&1u);
}
static void toggle(Mat256*m,unsigned r,unsigned c){
  m->row[r][c>>6]^=UINT64_C(1)<<(c&63u);
}

static void build_generic(Mat256*m){
  memset(m,0,sizeof(*m));
  const uint64_t seed=UINT64_C(0x6a09e667f3bcc909);
  for(unsigned r=0;r<N;++r) for(unsigned q=0;q<7;++q){
    uint64_t h=mix64(seed^((uint64_t)r<<12)^q);
    toggle(m,r,(unsigned)(h&255u));
  }
}

static void build_pair_equivariant(Mat256*m){
  memset(m,0,sizeof(*m));
  for(unsigned r=0;r<127;++r){
    for(unsigned parity=0;parity<2;++parity){
      unsigned rr=2u*r+parity;
      toggle(m,rr,2u*r+parity);
      toggle(m,rr,2u*(r+1u)+parity);
    }
  }
}

static int u32cmp(const void*a,const void*b){
  uint32_t x=*(const uint32_t*)a,y=*(const uint32_t*)b;
  return (x>y)-(x<y);
}

static uint64_t multiset_hash(uint32_t *a,unsigned n,uint64_t salt){
  qsort(a,n,sizeof(uint32_t),u32cmp);
  uint64_t h=mix64(salt^n);
  for(unsigned i=0;i<n;++i) h=mix64(h^((uint64_t)a[i]+UINT64_C(0x9e3779b97f4a7c15)*(i+1u)));
  return h;
}

static int sig_less(const Signature*a,const Signature*b){
#define CMP(f) do{ if(a->f<b->f) return 1; if(a->f>b->f) return 0; }while(0)
  CMP(oldc); CMP(rowdeg); CMP(coldeg); CMP(diag); CMP(out_hash); CMP(in_hash);
#undef CMP
  return 0;
}
static int sig_equal(const Signature*a,const Signature*b){
  return a->oldc==b->oldc && a->rowdeg==b->rowdeg && a->coldeg==b->coldeg &&
         a->diag==b->diag && a->out_hash==b->out_hash && a->in_hash==b->in_hash;
}

static void sort_indices_by_sig(unsigned idx[N],const Signature sig[N]){
  /* N is tiny: deterministic insertion sort avoids qsort context plumbing. */
  for(unsigned i=0;i<N;++i) idx[i]=i;
  for(unsigned i=1;i<N;++i){
    unsigned x=idx[i],j=i;
    while(j>0 && sig_less(&sig[x],&sig[idx[j-1]])){ idx[j]=idx[j-1]; --j; }
    idx[j]=x;
  }
}

static unsigned refine(const Mat256*m,uint32_t colour[N],unsigned *rounds){
  unsigned rowdeg[N]={0}, coldeg[N]={0};
  for(unsigned r=0;r<N;++r) for(unsigned c=0;c<N;++c) if(bit(m,r,c)){ ++rowdeg[r]; ++coldeg[c]; }

  /* Initial exact structural coordinates. */
  Signature init[N];
  for(unsigned i=0;i<N;++i){
    init[i]=(Signature){0,rowdeg[i],coldeg[i],bit(m,i,i),0,0};
  }
  unsigned order[N]; sort_indices_by_sig(order,init);
  uint32_t k=0; colour[order[0]]=0;
  for(unsigned j=1;j<N;++j){ if(!sig_equal(&init[order[j]],&init[order[j-1]])) ++k; colour[order[j]]=k; }

  for(unsigned iter=0;iter<N;++iter){
    Signature sig[N];
    for(unsigned i=0;i<N;++i){
      uint32_t out[MAX_NEIGH],in[MAX_NEIGH]; unsigned no=0,ni=0;
      for(unsigned c=0;c<N;++c) if(bit(m,i,c)) out[no++]=colour[c];
      for(unsigned r=0;r<N;++r) if(bit(m,r,i)) in[ni++]=colour[r];
      sig[i]=(Signature){colour[i],rowdeg[i],coldeg[i],bit(m,i,i),
        multiset_hash(out,no,UINT64_C(0x243f6a8885a308d3)),
        multiset_hash(in,ni,UINT64_C(0x13198a2e03707344))};
    }
    sort_indices_by_sig(order,sig);
    uint32_t next[N],nk=0; next[order[0]]=0;
    for(unsigned j=1;j<N;++j){ if(!sig_equal(&sig[order[j]],&sig[order[j-1]])) ++nk; next[order[j]]=nk; }
    int same=1; for(unsigned i=0;i<N;++i) if(next[i]!=colour[i]){ same=0; break; }
    memcpy(colour,next,sizeof(next));
    *rounds=iter+1u;
    if(same) return nk+1u;
  }
  return 0;
}

static int commutes(const Mat256*m,const unsigned p[N]){
  for(unsigned r=0;r<N;++r) for(unsigned c=0;c<N;++c)
    if(bit(m,p[r],p[c])!=bit(m,r,c)) return 0;
  return 1;
}

static void class_stats(const uint32_t colour[N],unsigned classes,
                        unsigned *singletons,unsigned *pairs,unsigned *max_class){
  unsigned count[N]={0};
  for(unsigned i=0;i<N;++i) ++count[colour[i]];
  *singletons=*pairs=*max_class=0;
  for(unsigned c=0;c<classes;++c){
    if(count[c]==1) ++*singletons;
    if(count[c]==2) ++*pairs;
    if(count[c]>*max_class) *max_class=count[c];
  }
}

static unsigned test_pair_transpositions(const Mat256*m,const uint32_t colour[N],unsigned classes){
  unsigned count[N]={0}, a[N],b[N];
  for(unsigned i=0;i<N;++i){ unsigned c=colour[i]; if(count[c]==0)a[c]=i; else if(count[c]==1)b[c]=i; ++count[c]; }
  unsigned commuting=0;
  for(unsigned c=0;c<classes;++c) if(count[c]==2){
    unsigned p[N]; for(unsigned i=0;i<N;++i)p[i]=i;
    p[a[c]]=b[c]; p[b[c]]=a[c];
    if(commutes(m,p)) ++commuting;
  }
  return commuting;
}

static int global_pair_candidate(const Mat256*m,const uint32_t colour[N],unsigned classes,unsigned *paired_classes){
  unsigned count[N]={0},a[N],b[N],p[N];
  for(unsigned i=0;i<N;++i){ p[i]=i; unsigned c=colour[i]; if(count[c]==0)a[c]=i; else if(count[c]==1)b[c]=i; ++count[c]; }
  *paired_classes=0;
  for(unsigned c=0;c<classes;++c){
    if(count[c]==2){ p[a[c]]=b[c]; p[b[c]]=a[c]; ++*paired_classes; }
    else if(count[c]!=1) return 0; /* only accept singleton/pair refined partitions */
  }
  return commutes(m,p);
}

static void run_case(const char *name,const Mat256*m,int expect_global_pair){
  uint32_t colour[N]; unsigned rounds=0;
  unsigned classes=refine(m,colour,&rounds);
  if(classes==0){ fprintf(stderr,"graph_refinement: FAIL %s no convergence\n",name); exit(EXIT_FAILURE); }
  unsigned singletons,pairs,maxc; class_stats(colour,classes,&singletons,&pairs,&maxc);
  unsigned local_commuting=test_pair_transpositions(m,colour,classes);
  unsigned paired_classes=0; int global=global_pair_candidate(m,colour,classes,&paired_classes);
  if(expect_global_pair && !global){
    fprintf(stderr,"graph_refinement: FAIL %s expected global pair symmetry; classes=%u pairs=%u max=%u rounds=%u\n",
      name,classes,pairs,maxc,rounds); exit(EXIT_FAILURE);
  }
  printf("rsa260_graph_refinement_symmetry_256_oracle: case=%s; classes=%u; singletons=%u; pairs=%u; max_class=%u; rounds=%u; commuting_local_pair_swaps=%u; global_pair_candidate=%s; paired_classes=%u\n",
    name,classes,singletons,pairs,maxc,rounds,local_commuting,global?"commuting":"not_commuting",paired_classes);
}

int main(void){
  Mat256 generic,pair;
  build_generic(&generic); build_pair_equivariant(&pair);
  run_case("generic",&generic,0);
  run_case("pair_equivariant",&pair,1);
  return EXIT_SUCCESS;
}
