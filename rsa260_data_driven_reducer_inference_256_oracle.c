#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Data-driven reducer inference oracle.
 * Synthetic 256-coordinate GF(2) benchmark only; NOT the RSA-260 matrix.
 *
 * Unlike rsa260_adaptive_reducer_hyperfabric_256_oracle.c, this program does
 * not receive region/resource labels as metadata.  It infers:
 *
 *   1. local pair-reducer candidates from the fixed pair coordinate fibre;
 *   2. co-requirement edges from actual cross-orbit matrix coupling;
 *   3. requirement-closed connected components from that inferred graph;
 *   4. component consumer signatures from an explicit projection observer;
 *   5. conflicts from collisions of those inferred observer signatures;
 *   6. the largest conflict-free closed family;
 *   7. the global composed action and MP=PM directly upstairs.
 *
 * The observer is part of the benchmark input, not a hidden resource-class
 * annotation.  This separates operator-derived requirements from
 * consumer-derived conflicts.
 */

enum { WIDTH=256, WORDS=4, ORBITS=128, CHANNELS=3, MAX_COMPONENTS=128 };
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;
typedef struct { uint64_t bits[CHANNELS][WORDS]; } Observer;

static unsigned bit(const Mat256*m,unsigned r,unsigned c){
  return (unsigned)((m->row[r][c>>6]>>(c&63u))&1u);
}
static void toggle(Mat256*m,unsigned r,unsigned c){
  m->row[r][c>>6]^=UINT64_C(1)<<(c&63u);
}

static void build_matrix(Mat256*m){
  memset(m,0,sizeof(*m));
  /* Eight disconnected 16-orbit path components are encoded only in M. */
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

static void build_observer(Observer*o){
  memset(o,0,sizeof(*o));
  /* Three projection channels.  Coordinates, not component labels, determine
     membership.  The repeated channel pattern creates consumer conflicts that
     must be inferred after components are discovered from M. */
  for(unsigned coord=0;coord<WIDTH;++coord){
    unsigned orbit=coord>>1;
    unsigned block=orbit/16u;
    unsigned channel=block%CHANNELS;
    o->bits[channel][coord>>6] |= UINT64_C(1)<<(coord&63u);
  }
}

static unsigned map_index(unsigned x,const unsigned selected_orbit[ORBITS]){
  return selected_orbit[x>>1] ? (x^1u) : x;
}

static int action_commutes(const Mat256*m,const unsigned selected_orbit[ORBITS]){
  for(unsigned r=0;r<WIDTH;++r){
    unsigned pr=map_index(r,selected_orbit);
    for(unsigned c=0;c<WIDTH;++c){
      unsigned pc=map_index(c,selected_orbit);
      if(bit(m,pr,pc)!=bit(m,r,c)) return 0;
    }
  }
  return 1;
}

static void infer_requirement_graph(const Mat256*m,unsigned req[ORBITS][ORBITS]){
  memset(req,0,sizeof(unsigned)*ORBITS*ORBITS);
  for(unsigned i=0;i<ORBITS;++i){
    for(unsigned j=i+1;j<ORBITS;++j){
      int coupled=0;
      for(unsigned pi=0;pi<2 && !coupled;++pi)
        for(unsigned pj=0;pj<2 && !coupled;++pj)
          if(bit(m,2u*i+pi,2u*j+pj) || bit(m,2u*j+pj,2u*i+pi)) coupled=1;
      if(coupled) req[i][j]=req[j][i]=1u;
    }
  }
}

static unsigned connected_components(const unsigned req[ORBITS][ORBITS],unsigned component_of[ORBITS]){
  for(unsigned i=0;i<ORBITS;++i) component_of[i]=MAX_COMPONENTS;
  unsigned components=0;
  for(unsigned seed=0;seed<ORBITS;++seed){
    if(component_of[seed]!=MAX_COMPONENTS) continue;
    unsigned q[ORBITS],head=0,tail=0;
    q[tail++]=seed; component_of[seed]=components;
    while(head<tail){
      unsigned u=q[head++];
      for(unsigned v=0;v<ORBITS;++v){
        if(req[u][v] && component_of[v]==MAX_COMPONENTS){
          component_of[v]=components; q[tail++]=v;
        }
      }
    }
    ++components;
  }
  return components;
}

static unsigned component_size(const unsigned component_of[ORBITS],unsigned c){
  unsigned n=0; for(unsigned i=0;i<ORBITS;++i) if(component_of[i]==c) ++n; return n;
}

static unsigned infer_component_channel(const Observer*o,const unsigned component_of[ORBITS],unsigned c){
  unsigned seen_mask=0;
  for(unsigned orbit=0;orbit<ORBITS;++orbit) if(component_of[orbit]==c){
    for(unsigned p=0;p<2;++p){
      unsigned coord=2u*orbit+p;
      for(unsigned ch=0;ch<CHANNELS;++ch)
        if((o->bits[ch][coord>>6]>>(coord&63u))&1u) seen_mask|=1u<<ch;
    }
  }
  return seen_mask;
}

static int observer_invariant_for_component(const Observer*o,const unsigned component_of[ORBITS],unsigned c){
  /* Pair swaps preserve this observer iff every swapped pair has identical
     membership in every observer channel. */
  for(unsigned orbit=0;orbit<ORBITS;++orbit) if(component_of[orbit]==c){
    unsigned a=2u*orbit,b=a+1u;
    for(unsigned ch=0;ch<CHANNELS;++ch){
      unsigned ba=(unsigned)((o->bits[ch][a>>6]>>(a&63u))&1u);
      unsigned bb=(unsigned)((o->bits[ch][b>>6]>>(b&63u))&1u);
      if(ba!=bb) return 0;
    }
  }
  return 1;
}

static unsigned popcount(unsigned x){ unsigned n=0; while(x){n+=x&1u;x>>=1;} return n; }

int main(void){
  Mat256 M; Observer O; build_matrix(&M); build_observer(&O);

  unsigned req[ORBITS][ORBITS]; infer_requirement_graph(&M,req);
  unsigned requirement_edges=0;
  for(unsigned i=0;i<ORBITS;++i)for(unsigned j=i+1;j<ORBITS;++j) requirement_edges+=req[i][j];

  unsigned component_of[ORBITS];
  unsigned components=connected_components(req,component_of);
  if(components!=8 || requirement_edges!=120){
    fprintf(stderr,"rsa260_data_driven_reducer_inference_256_oracle: FAIL topology components=%u edges=%u\n",components,requirement_edges);
    return EXIT_FAILURE;
  }
  for(unsigned c=0;c<components;++c) if(component_size(component_of,c)!=16) return EXIT_FAILURE;

  unsigned component_channel[MAX_COMPONENTS]={0};
  unsigned observer_invariant_components=0;
  for(unsigned c=0;c<components;++c){
    component_channel[c]=infer_component_channel(&O,component_of,c);
    if(popcount(component_channel[c])!=1) return EXIT_FAILURE;
    if(observer_invariant_for_component(&O,component_of,c)) ++observer_invariant_components;
  }
  if(observer_invariant_components!=components) return EXIT_FAILURE;

  unsigned conflict[MAX_COMPONENTS][MAX_COMPONENTS];
  memset(conflict,0,sizeof(conflict));
  unsigned conflict_edges=0;
  for(unsigned a=0;a<components;++a)for(unsigned b=a+1;b<components;++b){
    if(component_channel[a]==component_channel[b]){
      conflict[a][b]=conflict[b][a]=1u; ++conflict_edges;
    }
  }
  if(conflict_edges!=7) return EXIT_FAILURE;

  unsigned best_mask=0,best_components=0,admissible=0;
  for(unsigned mask=0;mask<(1u<<components);++mask){
    int ok=1;
    for(unsigned a=0;a<components && ok;++a) if(mask&(1u<<a))
      for(unsigned b=a+1;b<components;++b) if((mask&(1u<<b)) && conflict[a][b]){ok=0;break;}
    if(!ok) continue;

    unsigned selected_orbit[ORBITS]={0};
    for(unsigned orbit=0;orbit<ORBITS;++orbit)
      if(mask&(1u<<component_of[orbit])) selected_orbit[orbit]=1u;
    if(!action_commutes(&M,selected_orbit)) continue;

    ++admissible;
    unsigned n=popcount(mask);
    if(n>best_components || (n==best_components && mask<best_mask)){
      best_components=n; best_mask=mask;
    }
  }

  if(best_components!=3) return EXIT_FAILURE;
  unsigned best_reducers=0;
  for(unsigned c=0;c<components;++c) if(best_mask&(1u<<c)) best_reducers+=component_size(component_of,c);
  if(best_reducers!=48) return EXIT_FAILURE;

  unsigned quotient_coordinates=WIDTH-best_reducers;
  if(quotient_coordinates!=208) return EXIT_FAILURE;

  printf("rsa260_data_driven_reducer_inference_256_oracle: ok; local_candidates=%u; inferred_requirement_edges=%u; inferred_components=%u; component_size=16; observer_invariant_components=%u; inferred_conflict_edges=%u; admissible_closed_batches=%u; selected_components=%u; selected_local_reducers=%u; quotient_coordinates=%u; global_action=commuting; production=0\n",
    ORBITS,requirement_edges,components,observer_invariant_components,conflict_edges,
    admissible,best_components,best_reducers,quotient_coordinates);
  return EXIT_SUCCESS;
}
