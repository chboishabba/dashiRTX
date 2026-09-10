#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Symmetry census for the SAME synthetic 256x256 GF(2) matrix used by the
 * RSA-260 Krylov/kernel-recovery oracle.
 *
 * Candidate family:
 *   - all 24 permutations of the four contiguous 64-coordinate blocks;
 *   - all 64 common cyclic shifts within each block.
 *
 * For each coordinate permutation P we test the exact commutation condition
 *
 *      M P = P M,
 *
 * equivalently M[p(r),p(c)] = M[r,c] for every matrix entry.
 *
 * This is a symmetry census, not an RSA-260 production claim.
 */

enum { WIDTH=256, WORDS=4, BLOCKS=4, BLOCK_WIDTH=64 };
typedef struct { uint64_t row[WIDTH][WORDS]; } Mat256;

static uint64_t mix64(uint64_t x){
  x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9);
  x^=x>>27; x*=UINT64_C(0x94d049bb133111eb);
  x^=x>>31; return x;
}

static void build_sparse_matrix(Mat256*m,uint64_t seed){
  memset(m,0,sizeof(*m));
  for(unsigned r=0;r<WIDTH;++r) for(unsigned q=0;q<7;++q){
    uint64_t h=mix64(seed^((uint64_t)r<<12)^q);
    unsigned c=(unsigned)(h&255u);
    m->row[r][c>>6]^=UINT64_C(1)<<(c&63u);
  }
}

static unsigned get_entry(const Mat256*m,unsigned r,unsigned c){
  return (unsigned)((m->row[r][c>>6]>>(c&63u))&1u);
}

static unsigned permute_index(unsigned i,const unsigned bp[4],unsigned shift){
  unsigned b=i>>6, j=i&63u;
  return (bp[b]<<6) | ((j+shift)&63u);
}

static int commutes(const Mat256*m,const unsigned bp[4],unsigned shift){
  for(unsigned r=0;r<WIDTH;++r){
    unsigned pr=permute_index(r,bp,shift);
    for(unsigned c=0;c<WIDTH;++c){
      unsigned pc=permute_index(c,bp,shift);
      if(get_entry(m,pr,pc)!=get_entry(m,r,c)) return 0;
    }
  }
  return 1;
}

static int next_perm(unsigned a[4]){
  int i=2;
  while(i>=0 && a[i]>=a[i+1]) --i;
  if(i<0) return 0;
  int j=3;
  while(a[j]<=a[i]) --j;
  unsigned t=a[i]; a[i]=a[j]; a[j]=t;
  for(int l=i+1,r=3;l<r;++l,--r){ t=a[l]; a[l]=a[r]; a[r]=t; }
  return 1;
}

int main(void){
  Mat256 M;
  build_sparse_matrix(&M,UINT64_C(0x6a09e667f3bcc909));

  unsigned bp[4]={0,1,2,3};
  uint64_t tested=0, commuting=0, nontrivial=0;
  uint64_t digest=UINT64_C(0x243f6a8885a308d3);

  do {
    for(unsigned shift=0;shift<64;++shift){
      int ok=commutes(&M,bp,shift);
      ++tested;
      if(ok){
        ++commuting;
        int identity=(shift==0 && bp[0]==0 && bp[1]==1 && bp[2]==2 && bp[3]==3);
        if(!identity) ++nontrivial;
        uint64_t code=((uint64_t)bp[0]<<24)|((uint64_t)bp[1]<<16)|((uint64_t)bp[2]<<8)|bp[3];
        digest=mix64(digest^code^((uint64_t)shift<<32));
      }
    }
  } while(next_perm(bp));

  if(tested!=1536){
    fprintf(stderr,"rsa260_symmetry_census_256_oracle: FAIL tested=%llu\n",(unsigned long long)tested);
    return EXIT_FAILURE;
  }
  if(commuting==0){
    fprintf(stderr,"rsa260_symmetry_census_256_oracle: FAIL identity missing\n");
    return EXIT_FAILURE;
  }

  printf("rsa260_symmetry_census_256_oracle: ok; candidates=%llu; commuting=%llu; nontrivial=%llu; family=S4_blocks_x_C64_shift; digest=%016llx\n",
         (unsigned long long)tested,(unsigned long long)commuting,
         (unsigned long long)nontrivial,(unsigned long long)digest);
  return EXIT_SUCCESS;
}
