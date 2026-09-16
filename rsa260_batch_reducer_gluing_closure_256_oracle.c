#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Hierarchical batch-reducer / gluing-closure oracle for the RSA-260 research
 * lane.  This uses the declared pair-equivariant 256-coordinate benchmark,
 * NOT the RSA-260 production matrix.
 *
 * There are 128 local candidate reducers: swap orbit pair (2i,2i+1).  No
 * individual candidate commutes with M.  The quotient adjacency Q is a path,
 * so swapping orbit i forces the same basis relabelling on every adjacent
 * orbit.  This induces a co-requirement/gluing graph whose connected closure
 * from any non-isolated seed is the full 128-candidate family.  The composed
 * family DOES commute with M.
 *
 * This is deliberately richer than a conflict graph:
 *   - conflict edge: reducers cannot coexist;
 *   - requirement edge: reducers must be composed together to close seams.
 *
 * Candidate IDs also receive a five-trit address (0..242) so the 128 occupied
 * candidates can be viewed through nested 3-adic cylinders.  The 115 unused
 * addresses are explicit padding/tail state, not candidate reducers.
 */

enum { WIDTH=256, WORDS=4, ORBITS=128, PADIC_DEPTH=5, PADIC_CAPACITY=243 };
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;

static unsigned bit(const Mat256*m,unsigned r,unsigned c){
  return (unsigned)((m->row[r][c>>6]>>(c&63u))&1u);
}
static void toggle(Mat256*m,unsigned r,unsigned c){
  m->row[r][c>>6]^=UINT64_C(1)<<(c&63u);
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

static unsigned local_swap_index(unsigned x,unsigned orbit){
  unsigned a=2u*orbit,b=a+1u;
  if(x==a) return b;
  if(x==b) return a;
  return x;
}

static int action_commutes(const Mat256*m,const unsigned active[ORBITS]){
  for(unsigned r=0;r<WIDTH;++r){
    unsigned pr=r;
    unsigned ro=r>>1;
    if(active[ro]) pr=local_swap_index(r,ro);
    for(unsigned c=0;c<WIDTH;++c){
      unsigned pc=c;
      unsigned co=c>>1;
      if(active[co]) pc=local_swap_index(c,co);
      if(bit(m,pr,pc)!=bit(m,r,c)) return 0;
    }
  }
  return 1;
}

static void build_requirement_graph(const Mat256*m,unsigned req[ORBITS][ORBITS]){
  memset(req,0,sizeof(unsigned)*ORBITS*ORBITS);
  /*
   * Two orbit-reducers are glued when M has a cross-orbit incidence between
   * them.  For this benchmark that recovers the quotient path adjacency.
   */
  for(unsigned i=0;i<ORBITS;++i){
    for(unsigned j=i+1;j<ORBITS;++j){
      int linked=0;
      for(unsigned pi=0;pi<2 && !linked;++pi)
        for(unsigned pj=0;pj<2 && !linked;++pj)
          if(bit(m,2u*i+pi,2u*j+pj) || bit(m,2u*j+pj,2u*i+pi)) linked=1;
      if(linked) req[i][j]=req[j][i]=1u;
    }
  }
}

static unsigned closure_from_seed(const unsigned req[ORBITS][ORBITS],unsigned seed,unsigned active[ORBITS]){
  memset(active,0,sizeof(unsigned)*ORBITS);
  active[seed]=1u;
  unsigned changed=1u;
  while(changed){
    changed=0u;
    for(unsigned i=0;i<ORBITS;++i) if(active[i])
      for(unsigned j=0;j<ORBITS;++j) if(req[i][j] && !active[j]){
        active[j]=1u; changed=1u;
      }
  }
  unsigned n=0; for(unsigned i=0;i<ORBITS;++i)n+=active[i];
  return n;
}

static unsigned ternary_prefix(unsigned id,unsigned depth){
  unsigned digits[PADIC_DEPTH]={0,0,0,0,0};
  for(int k=PADIC_DEPTH-1;k>=0;--k){ digits[k]=id%3u; id/=3u; }
  unsigned p=0;
  for(unsigned k=0;k<depth;++k)p=3u*p+digits[k];
  return p;
}

static unsigned occupied_cylinders(unsigned depth){
  unsigned seen[PADIC_CAPACITY]={0};
  unsigned count=0;
  for(unsigned i=0;i<ORBITS;++i){
    unsigned p=ternary_prefix(i,depth);
    if(!seen[p]){ seen[p]=1u; ++count; }
  }
  return count;
}

int main(void){
  Mat256 M; build_pair_equivariant(&M);

  unsigned individual_commuting=0;
  for(unsigned i=0;i<ORBITS;++i){
    unsigned active[ORBITS]={0}; active[i]=1u;
    if(action_commutes(&M,active)) ++individual_commuting;
  }
  if(individual_commuting!=0){
    fprintf(stderr,"rsa260_batch_reducer_gluing_closure_256_oracle: FAIL individual=%u\n",individual_commuting);
    return EXIT_FAILURE;
  }

  unsigned req[ORBITS][ORBITS]; build_requirement_graph(&M,req);
  unsigned requirement_edges=0;
  for(unsigned i=0;i<ORBITS;++i)for(unsigned j=i+1;j<ORBITS;++j)requirement_edges+=req[i][j];
  if(requirement_edges!=127){
    fprintf(stderr,"rsa260_batch_reducer_gluing_closure_256_oracle: FAIL requirement_edges=%u\n",requirement_edges);
    return EXIT_FAILURE;
  }

  unsigned active[ORBITS];
  unsigned closure=closure_from_seed(req,0,active);
  if(closure!=ORBITS || !action_commutes(&M,active)){
    fprintf(stderr,"rsa260_batch_reducer_gluing_closure_256_oracle: FAIL closure=%u commutes=%d\n",closure,action_commutes(&M,active));
    return EXIT_FAILURE;
  }

  /* Every seed lies in the same requirement component and closes globally. */
  unsigned all_seed_closures=0;
  for(unsigned s=0;s<ORBITS;++s){
    unsigned a[ORBITS];
    unsigned n=closure_from_seed(req,s,a);
    if(n==ORBITS && action_commutes(&M,a)) ++all_seed_closures;
  }
  if(all_seed_closures!=ORBITS){
    fprintf(stderr,"rsa260_batch_reducer_gluing_closure_256_oracle: FAIL seed_closures=%u\n",all_seed_closures);
    return EXIT_FAILURE;
  }

  unsigned c1=occupied_cylinders(1),c2=occupied_cylinders(2),c3=occupied_cylinders(3),
           c4=occupied_cylinders(4),c5=occupied_cylinders(5);
  if(c5!=ORBITS || PADIC_CAPACITY-ORBITS!=115){
    fprintf(stderr,"rsa260_batch_reducer_gluing_closure_256_oracle: FAIL padic c5=%u\n",c5);
    return EXIT_FAILURE;
  }

  printf("rsa260_batch_reducer_gluing_closure_256_oracle: ok; local_candidates=%u; individual_commuting=%u; conflict_edges=0; requirement_edges=%u; closure_from_seed=%u; all_seed_global_closures=%u; global_action=commuting; quotient_factor=2; padic_depth=%u; padic_capacity=%u; padic_tail=%u; occupied_cylinders=%u,%u,%u,%u,%u\n",
    ORBITS,individual_commuting,requirement_edges,closure,all_seed_closures,
    PADIC_DEPTH,PADIC_CAPACITY,PADIC_CAPACITY-ORBITS,c1,c2,c3,c4,c5);
  return EXIT_SUCCESS;
}
