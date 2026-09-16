#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Adaptive reducer-hyperfabric oracle.
 *
 * Synthetic 256-coordinate GF(2) benchmark only; NOT the RSA-260 matrix.
 *
 * The carrier is 8 disconnected 32-coordinate regions.  Each region contains
 * 16 opposite-pair orbits.  Inside one region, local pair swaps are glued by
 * quotient-path incidences, so requirement closure turns any seed into the
 * whole 16-reducer region.  Region closures are independent at the operator
 * level.
 *
 * To exercise the selection layer rather than a trivial all-components case,
 * each region also carries a declared consumer resource class 0..2.  A batch
 * may contain at most one region of each resource class.  This is an explicit
 * synthetic consumer constraint, not inferred from MP=PM.  The selector must
 * therefore distinguish:
 *
 *   requirement closure  : what MUST be selected together;
 *   conflict constraint  : what MUST NOT be selected together;
 *   operator equivariance: what the composed action must still satisfy.
 *
 * Regions receive coarse/fine ternary addresses.  Refinement is adaptive:
 * depth increases only while unresolved resource/conflict ambiguity remains.
 */

enum { WIDTH=256, WORDS=4, REGIONS=8, ORBITS_PER_REGION=16,
       ORBITS=128, PADIC_DEPTH=3, PADIC_CAPACITY=27 };
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;

typedef struct {
  unsigned region;
  unsigned resource_class;
  unsigned trits[PADIC_DEPTH];
} RegionMeta;

static unsigned bit(const Mat256*m,unsigned r,unsigned c){
  return (unsigned)((m->row[r][c>>6]>>(c&63u))&1u);
}
static void toggle(Mat256*m,unsigned r,unsigned c){
  m->row[r][c>>6]^=UINT64_C(1)<<(c&63u);
}

static void build_matrix(Mat256*m){
  memset(m,0,sizeof(*m));
  for(unsigned g=0;g<REGIONS;++g){
    unsigned base_orbit=g*ORBITS_PER_REGION;
    for(unsigned j=0;j<ORBITS_PER_REGION-1;++j){
      unsigned o=base_orbit+j;
      for(unsigned p=0;p<2;++p){
        unsigned r=2u*o+p;
        toggle(m,r,2u*o+p);
        toggle(m,r,2u*(o+1u)+p);
      }
    }
  }
}

static unsigned map_index(unsigned x,const unsigned selected_region[REGIONS]){
  unsigned orbit=x>>1;
  unsigned region=orbit/ORBITS_PER_REGION;
  if(selected_region[region]) return x^1u;
  return x;
}

static int action_commutes(const Mat256*m,const unsigned selected_region[REGIONS]){
  for(unsigned r=0;r<WIDTH;++r){
    unsigned pr=map_index(r,selected_region);
    for(unsigned c=0;c<WIDTH;++c){
      unsigned pc=map_index(c,selected_region);
      if(bit(m,pr,pc)!=bit(m,r,c)) return 0;
    }
  }
  return 1;
}

static void metadata(RegionMeta meta[REGIONS]){
  /* Deliberately repeated resource classes generate real batch conflicts. */
  static const unsigned resource[REGIONS]={0,1,2,0,1,2,0,1};
  for(unsigned g=0;g<REGIONS;++g){
    meta[g].region=g;
    meta[g].resource_class=resource[g];
    unsigned x=g;
    for(int k=PADIC_DEPTH-1;k>=0;--k){ meta[g].trits[k]=x%3u; x/=3u; }
  }
}

static unsigned prefix(const RegionMeta*m,unsigned depth){
  unsigned p=0; for(unsigned k=0;k<depth;++k)p=3u*p+m->trits[k]; return p;
}

static unsigned unresolved_pairs_at_depth(const RegionMeta meta[REGIONS],unsigned depth){
  unsigned unresolved=0;
  for(unsigned i=0;i<REGIONS;++i)for(unsigned j=i+1;j<REGIONS;++j)
    if(prefix(&meta[i],depth)==prefix(&meta[j],depth) &&
       meta[i].resource_class!=meta[j].resource_class) ++unresolved;
  return unresolved;
}

static unsigned requirement_edges(void){
  /* Each 16-orbit path contributes 15 local co-requirement edges. */
  return REGIONS*(ORBITS_PER_REGION-1u);
}

static unsigned conflict_edges(const RegionMeta meta[REGIONS]){
  unsigned n=0;
  for(unsigned i=0;i<REGIONS;++i)for(unsigned j=i+1;j<REGIONS;++j)
    if(meta[i].resource_class==meta[j].resource_class) ++n;
  return n;
}

static unsigned popcount8(unsigned x){
  unsigned n=0; while(x){n+=x&1u;x>>=1;} return n;
}

static int conflict_free_mask(unsigned mask,const RegionMeta meta[REGIONS]){
  unsigned used[3]={0,0,0};
  for(unsigned g=0;g<REGIONS;++g)if(mask&(1u<<g)){
    unsigned r=meta[g].resource_class;
    if(used[r]) return 0;
    used[r]=1u;
  }
  return 1;
}

int main(void){
  Mat256 M; RegionMeta meta[REGIONS];
  build_matrix(&M); metadata(meta);

  /* Local reducers remain invalid alone; each full requirement component works. */
  unsigned individual_commuting=0;
  for(unsigned orbit=0;orbit<ORBITS;++orbit){
    unsigned selected[REGIONS]={0};
    unsigned region=orbit/ORBITS_PER_REGION;
    /* Test one local pair manually rather than whole-region map. */
    int ok=1;
    for(unsigned r=0;r<WIDTH && ok;++r){
      unsigned pr=r;
      if((r>>1)==orbit) pr=r^1u;
      for(unsigned c=0;c<WIDTH;++c){
        unsigned pc=c;
        if((c>>1)==orbit) pc=c^1u;
        if(bit(&M,pr,pc)!=bit(&M,r,c)){ ok=0; break; }
      }
    }
    if(ok) ++individual_commuting;
    (void)selected;
  }
  if(individual_commuting!=0){
    fprintf(stderr,"rsa260_adaptive_reducer_hyperfabric_256_oracle: FAIL local=%u\n",individual_commuting);
    return EXIT_FAILURE;
  }

  unsigned component_commuting=0;
  for(unsigned g=0;g<REGIONS;++g){
    unsigned selected[REGIONS]={0}; selected[g]=1u;
    if(action_commutes(&M,selected)) ++component_commuting;
  }
  if(component_commuting!=REGIONS){
    fprintf(stderr,"rsa260_adaptive_reducer_hyperfabric_256_oracle: FAIL components=%u\n",component_commuting);
    return EXIT_FAILURE;
  }

  /* Adaptive p-adic refinement: stop when another depth no longer resolves
     cross-resource ambiguity or after full declared depth. */
  unsigned unresolved[PADIC_DEPTH+1];
  unresolved[0]=0;
  for(unsigned i=0;i<REGIONS;++i)for(unsigned j=i+1;j<REGIONS;++j)
    if(meta[i].resource_class!=meta[j].resource_class) ++unresolved[0];
  unsigned stop_depth=PADIC_DEPTH;
  for(unsigned d=1;d<=PADIC_DEPTH;++d){
    unresolved[d]=unresolved_pairs_at_depth(meta,d);
    if(unresolved[d]==0){ stop_depth=d; break; }
    if(d>1 && unresolved[d]==unresolved[d-1]){ stop_depth=d-1; break; }
  }

  /* Exhaustive selector over the 8 requirement-closed components.  Objective:
     maximize number of local reducers, equivalently selected regions here. */
  unsigned best_mask=0,best_regions=0,admissible_masks=0;
  for(unsigned mask=0;mask<(1u<<REGIONS);++mask){
    if(!conflict_free_mask(mask,meta)) continue;
    unsigned selected[REGIONS]={0};
    for(unsigned g=0;g<REGIONS;++g)selected[g]=(mask>>g)&1u;
    if(!action_commutes(&M,selected)) continue;
    ++admissible_masks;
    unsigned n=popcount8(mask);
    if(n>best_regions || (n==best_regions && mask<best_mask)){
      best_regions=n; best_mask=mask;
    }
  }

  if(best_regions!=3){
    fprintf(stderr,"rsa260_adaptive_reducer_hyperfabric_256_oracle: FAIL best_regions=%u mask=%u\n",best_regions,best_mask);
    return EXIT_FAILURE;
  }

  unsigned selected[REGIONS]={0};
  for(unsigned g=0;g<REGIONS;++g)selected[g]=(best_mask>>g)&1u;
  if(!action_commutes(&M,selected)) return EXIT_FAILURE;

  unsigned local_reducers=best_regions*ORBITS_PER_REGION;
  unsigned quotient_coordinates=WIDTH-local_reducers; /* each pair swap quotient deletes one coordinate */
  if(local_reducers!=48 || quotient_coordinates!=208) return EXIT_FAILURE;

  printf("rsa260_adaptive_reducer_hyperfabric_256_oracle: ok; local_candidates=%u; individual_commuting=%u; requirement_components=%u; requirement_edges=%u; conflict_edges=%u; component_commuting=%u; adaptive_stop_depth=%u; unresolved=%u,%u,%u,%u; admissible_closed_batches=%u; selected_regions=%u; selected_local_reducers=%u; quotient_coordinates=%u; structural_reduction=%u/256; global_action=commuting\n",
    ORBITS,individual_commuting,REGIONS,requirement_edges(),conflict_edges(meta),component_commuting,
    stop_depth,unresolved[0],unresolved[1],unresolved[2],unresolved[3],admissible_masks,
    best_regions,local_reducers,quotient_coordinates,local_reducers);
  return EXIT_SUCCESS;
}
