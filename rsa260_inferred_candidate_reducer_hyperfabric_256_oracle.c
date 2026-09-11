#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Inferred-candidate reducer-hyperfabric oracle.
 * Synthetic 256-coordinate GF(2) benchmark only; NOT the RSA-260 matrix.
 *
 * This removes the previous planted local-pair assumption from the reduction
 * pipeline. Candidate transpositions are inferred from NDim graph refinement:
 * row degree, column degree, diagonal bit, outgoing-neighbour colour multiset,
 * and incoming-neighbour colour multiset. Only stable colour classes of size 2
 * become local reducer candidates.
 *
 * The inferred candidates then feed the existing data-driven pipeline:
 *   candidate actions
 *   -> operator-derived co-requirement edges
 *   -> requirement components
 *   -> projection-observer signatures
 *   -> consumer-relative conflicts
 *   -> largest closed conflict-free batch
 *   -> composed MP=PM verification.
 *
 * Equal refinement colour is never promoted directly to automorphism.
 */

enum { N=256, WORDS=4, MAXC=256, CHANNELS=3 };
typedef struct { uint64_t row[N][WORDS]; } Mat256;
typedef struct { uint64_t bits[CHANNELS][WORDS]; } Observer;
typedef struct { unsigned a,b; } PairCandidate;
typedef struct {
  uint32_t oldc,rowdeg,coldeg,diag;
  uint64_t out_hash,in_hash;
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
#define CMP(f) do{if(a->f<b->f)return 1;if(a->f>b->f)return 0;}while(0)
  CMP(oldc); CMP(rowdeg); CMP(coldeg); CMP(diag); CMP(out_hash); CMP(in_hash);
#undef CMP
  return 0;
}
static int sig_equal(const Signature*a,const Signature*b){
  return a->oldc==b->oldc && a->rowdeg==b->rowdeg && a->coldeg==b->coldeg &&
         a->diag==b->diag && a->out_hash==b->out_hash && a->in_hash==b->in_hash;
}
static void sort_indices(unsigned idx[N],const Signature sig[N]){
  for(unsigned i=0;i<N;++i) idx[i]=i;
  for(unsigned i=1;i<N;++i){
    unsigned x=idx[i],j=i;
    while(j>0 && sig_less(&sig[x],&sig[idx[j-1]])){idx[j]=idx[j-1];--j;}
    idx[j]=x;
  }
}

static void build_pair_benchmark(Mat256*m){
  memset(m,0,sizeof(*m));
  for(unsigned g=0;g<8;++g){
    unsigned base=g*16u;
    for(unsigned j=0;j<15;++j){
      unsigned o=base+j;
      for(unsigned p=0;p<2;++p){
        unsigned r=2u*o+p;
        toggle(m,r,2u*o+p);
        toggle(m,r,2u*(o+1u)+p);
      }
    }
  }
}
static void build_generic(Mat256*m){
  memset(m,0,sizeof(*m));
  const uint64_t seed=UINT64_C(0x6a09e667f3bcc909);
  for(unsigned r=0;r<N;++r) for(unsigned q=0;q<7;++q){
    uint64_t h=mix64(seed^((uint64_t)r<<12)^q);
    toggle(m,r,(unsigned)(h&255u));
  }
}
static void build_observer(Observer*o){
  memset(o,0,sizeof(*o));
  for(unsigned coord=0;coord<N;++coord){
    unsigned block=(coord>>1)/16u;
    unsigned ch=block%CHANNELS;
    o->bits[ch][coord>>6]|=UINT64_C(1)<<(coord&63u);
  }
}

static unsigned refine(const Mat256*m,uint32_t colour[N],unsigned *rounds){
  unsigned rowdeg[N]={0},coldeg[N]={0};
  for(unsigned r=0;r<N;++r)for(unsigned c=0;c<N;++c)if(bit(m,r,c)){++rowdeg[r];++coldeg[c];}
  Signature init[N];
  for(unsigned i=0;i<N;++i) init[i]=(Signature){0,rowdeg[i],coldeg[i],bit(m,i,i),0,0};
  unsigned order[N]; sort_indices(order,init);
  uint32_t k=0; colour[order[0]]=0;
  for(unsigned j=1;j<N;++j){if(!sig_equal(&init[order[j]],&init[order[j-1]]))++k;colour[order[j]]=k;}
  for(unsigned iter=0;iter<N;++iter){
    Signature sig[N];
    for(unsigned i=0;i<N;++i){
      uint32_t out[N],in[N];unsigned no=0,ni=0;
      for(unsigned c=0;c<N;++c)if(bit(m,i,c))out[no++]=colour[c];
      for(unsigned r=0;r<N;++r)if(bit(m,r,i))in[ni++]=colour[r];
      sig[i]=(Signature){colour[i],rowdeg[i],coldeg[i],bit(m,i,i),
        multiset_hash(out,no,UINT64_C(0x243f6a8885a308d3)),
        multiset_hash(in,ni,UINT64_C(0x13198a2e03707344))};
    }
    sort_indices(order,sig);
    uint32_t next[N],nk=0;next[order[0]]=0;
    for(unsigned j=1;j<N;++j){if(!sig_equal(&sig[order[j]],&sig[order[j-1]]))++nk;next[order[j]]=nk;}
    int same=1;for(unsigned i=0;i<N;++i)if(next[i]!=colour[i]){same=0;break;}
    memcpy(colour,next,sizeof(next));*rounds=iter+1u;
    if(same)return nk+1u;
  }
  return 0;
}

static unsigned infer_pair_candidates(const uint32_t colour[N],unsigned classes,PairCandidate cand[MAXC]){
  unsigned count[N]={0},first[N]={0};
  for(unsigned i=0;i<N;++i){unsigned c=colour[i];if(count[c]==0)first[c]=i;++count[c];}
  unsigned n=0;
  for(unsigned c=0;c<classes;++c)if(count[c]==2){
    unsigned a=first[c],b=N;
    for(unsigned i=a+1;i<N;++i)if(colour[i]==c){b=i;break;}
    if(b<N)cand[n++]=(PairCandidate){a,b};
  }
  return n;
}

static unsigned apply_candidate(unsigned x,const PairCandidate*c){
  if(x==c->a)return c->b;
  if(x==c->b)return c->a;
  return x;
}
static unsigned map_selected(unsigned x,const PairCandidate cand[MAXC],unsigned n,const unsigned sel[MAXC]){
  for(unsigned i=0;i<n;++i)if(sel[i] && (x==cand[i].a || x==cand[i].b))return apply_candidate(x,&cand[i]);
  return x;
}
static int action_commutes(const Mat256*m,const PairCandidate cand[MAXC],unsigned n,const unsigned sel[MAXC]){
  for(unsigned r=0;r<N;++r){unsigned pr=map_selected(r,cand,n,sel);
    for(unsigned c=0;c<N;++c){unsigned pc=map_selected(c,cand,n,sel);
      if(bit(m,pr,pc)!=bit(m,r,c))return 0;}}
  return 1;
}

static void infer_requirement_graph(const Mat256*m,const PairCandidate cand[MAXC],unsigned n,unsigned req[MAXC][MAXC]){
  memset(req,0,sizeof(unsigned)*MAXC*MAXC);
  for(unsigned i=0;i<n;++i)for(unsigned j=i+1;j<n;++j){
    unsigned vi[2]={cand[i].a,cand[i].b},vj[2]={cand[j].a,cand[j].b};
    int coupled=0;
    for(unsigned a=0;a<2 && !coupled;++a)for(unsigned b=0;b<2 && !coupled;++b)
      if(bit(m,vi[a],vj[b])||bit(m,vj[b],vi[a]))coupled=1;
    if(coupled)req[i][j]=req[j][i]=1u;
  }
}
static unsigned connected_components(const unsigned req[MAXC][MAXC],unsigned n,unsigned comp[MAXC]){
  for(unsigned i=0;i<n;++i)comp[i]=MAXC;
  unsigned nc=0;
  for(unsigned s=0;s<n;++s)if(comp[s]==MAXC){
    unsigned q[MAXC],h=0,t=0;q[t++]=s;comp[s]=nc;
    while(h<t){unsigned u=q[h++];for(unsigned v=0;v<n;++v)if(req[u][v]&&comp[v]==MAXC){comp[v]=nc;q[t++]=v;}}
    ++nc;
  }
  return nc;
}
static unsigned component_size(const unsigned comp[MAXC],unsigned n,unsigned c){
  unsigned z=0;for(unsigned i=0;i<n;++i)if(comp[i]==c)++z;return z;
}
static unsigned component_channel(const Observer*o,const PairCandidate cand[MAXC],const unsigned comp[MAXC],unsigned n,unsigned c){
  unsigned mask=0;
  for(unsigned i=0;i<n;++i)if(comp[i]==c){unsigned v[2]={cand[i].a,cand[i].b};
    for(unsigned t=0;t<2;++t)for(unsigned ch=0;ch<CHANNELS;++ch)
      if((o->bits[ch][v[t]>>6]>>(v[t]&63u))&1u)mask|=1u<<ch;}
  return mask;
}
static unsigned popcount(unsigned x){unsigned n=0;while(x){n+=x&1u;x>>=1;}return n;}

static void run_generic(void){
  Mat256 M;build_generic(&M);uint32_t col[N];unsigned rounds=0;
  unsigned classes=refine(&M,col,&rounds);PairCandidate cand[MAXC];
  unsigned n=infer_pair_candidates(col,classes,cand);
  if(classes!=256 || n!=0){fprintf(stderr,"inferred_candidate: FAIL generic classes=%u candidates=%u\n",classes,n);exit(EXIT_FAILURE);}
  printf("rsa260_inferred_candidate_reducer_hyperfabric_256_oracle: case=generic; classes=%u; inferred_pair_candidates=%u; rounds=%u; fail_closed=1\n",classes,n,rounds);
}

static void run_pair(void){
  Mat256 M;Observer O;build_pair_benchmark(&M);build_observer(&O);
  uint32_t col[N];unsigned rounds=0;unsigned classes=refine(&M,col,&rounds);
  PairCandidate cand[MAXC];unsigned n=infer_pair_candidates(col,classes,cand);
  if(classes!=128 || n!=128){fprintf(stderr,"inferred_candidate: FAIL pair classes=%u candidates=%u\n",classes,n);exit(EXIT_FAILURE);}

  unsigned individual=0;
  for(unsigned i=0;i<n;++i){unsigned sel[MAXC]={0};sel[i]=1;if(action_commutes(&M,cand,n,sel))++individual;}
  if(individual!=0)return (void)exit(EXIT_FAILURE);

  unsigned req[MAXC][MAXC];infer_requirement_graph(&M,cand,n,req);
  unsigned edges=0;for(unsigned i=0;i<n;++i)for(unsigned j=i+1;j<n;++j)edges+=req[i][j];
  unsigned comp[MAXC];unsigned nc=connected_components(req,n,comp);
  if(nc!=8 || edges!=120)return (void)exit(EXIT_FAILURE);
  for(unsigned c=0;c<nc;++c)if(component_size(comp,n,c)!=16)return (void)exit(EXIT_FAILURE);

  unsigned channel[MAXC]={0};
  for(unsigned c=0;c<nc;++c){channel[c]=component_channel(&O,cand,comp,n,c);if(popcount(channel[c])!=1)return (void)exit(EXIT_FAILURE);}
  unsigned conflict[MAXC][MAXC]={0},conflict_edges=0;
  for(unsigned a=0;a<nc;++a)for(unsigned b=a+1;b<nc;++b)if(channel[a]==channel[b]){conflict[a][b]=conflict[b][a]=1;++conflict_edges;}
  if(conflict_edges!=7)return (void)exit(EXIT_FAILURE);

  unsigned best=0,bmask=0,admissible=0;
  for(unsigned mask=0;mask<(1u<<nc);++mask){
    int ok=1;for(unsigned a=0;a<nc && ok;++a)if(mask&(1u<<a))for(unsigned b=a+1;b<nc;++b)if((mask&(1u<<b))&&conflict[a][b]){ok=0;break;}
    if(!ok)continue;
    unsigned sel[MAXC]={0};for(unsigned i=0;i<n;++i)if(mask&(1u<<comp[i]))sel[i]=1;
    if(!action_commutes(&M,cand,n,sel))continue;
    ++admissible;unsigned k=popcount(mask);if(k>best){best=k;bmask=mask;}
  }
  if(best!=3)return (void)exit(EXIT_FAILURE);
  unsigned selected=0;for(unsigned c=0;c<nc;++c)if(bmask&(1u<<c))selected+=component_size(comp,n,c);
  if(selected!=48)return (void)exit(EXIT_FAILURE);

  printf("rsa260_inferred_candidate_reducer_hyperfabric_256_oracle: case=pair_equivariant; classes=%u; inferred_pair_candidates=%u; refinement_rounds=%u; individual_commuting=%u; inferred_requirement_edges=%u; inferred_components=%u; component_size=16; inferred_conflict_edges=%u; admissible_closed_batches=%u; selected_components=%u; selected_local_reducers=%u; quotient_coordinates=%u; global_action=commuting\n",
    classes,n,rounds,individual,edges,nc,conflict_edges,admissible,best,selected,N-selected);
}

int main(void){run_generic();run_pair();return EXIT_SUCCESS;}
