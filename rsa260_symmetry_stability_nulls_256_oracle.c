#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N=256, W=4, DRAWS=64 };
typedef struct { uint64_t row[N][W]; } Mat;
static unsigned bit(const Mat*m,unsigned r,unsigned c){return (unsigned)((m->row[r][c>>6]>>(c&63u))&1u);} static void setb(Mat*m,unsigned r,unsigned c){m->row[r][c>>6]|=UINT64_C(1)<<(c&63u);}
static uint64_t rngs=UINT64_C(0x8e7d2c1f93b54a67); static uint64_t rng(void){rngs^=rngs<<7;rngs^=rngs>>9;rngs^=rngs<<8;return rngs;}
static void build_observed(Mat*m){memset(m,0,sizeof(*m));for(unsigned c=0;c<60;c++){unsigned b=4*c;for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++)if(i!=j)setb(m,b+i,b+j);if((c%20)+1<20)for(unsigned i=0;i<4;i++)setb(m,b+i,b+4+i);}for(unsigned i=240;i+1<N;i++)setb(m,i,i+1);}
static void rowdeg(const Mat*m,unsigned d[N]){memset(d,0,sizeof(unsigned)*N);for(unsigned r=0;r<N;r++)for(unsigned c=0;c<N;c++)d[r]+=bit(m,r,c);}
static void coldeg(const Mat*m,unsigned d[N]){memset(d,0,sizeof(unsigned)*N);for(unsigned c=0;c<N;c++)for(unsigned r=0;r<N;r++)d[c]+=bit(m,r,c);}
static int swap_edges_preserve_degrees(Mat*m,unsigned r1,unsigned c1,unsigned r2,unsigned c2){if(r1==r2||c1==c2)return 0;unsigned a=bit(m,r1,c1),b=bit(m,r1,c2),c=bit(m,r2,c1),d=bit(m,r2,c2);if(a==d&&b==c&&a!=b){m->row[r1][c1>>6]^=UINT64_C(1)<<(c1&63u);m->row[r1][c2>>6]^=UINT64_C(1)<<(c2&63u);m->row[r2][c1>>6]^=UINT64_C(1)<<(c1&63u);m->row[r2][c2>>6]^=UINT64_C(1)<<(c2&63u);return 1;}return 0;}
static void degree_preserving_scramble(const Mat*src,Mat*dst,unsigned steps){*dst=*src;for(unsigned t=0;t<steps;t++){unsigned r1=rng()%N,r2=rng()%N,c1=rng()%N,c2=rng()%N;swap_edges_preserve_degrees(dst,r1,c1,r2,c2);}}
static int commute_component_v4(const Mat*m,unsigned comp,unsigned which){static const unsigned p[3][4]={{1,0,3,2},{2,3,0,1},{3,2,1,0}};for(unsigned r=0;r<N;r++){unsigned pr=r;if(r<240&&r/80==comp){unsigned b=(r/4)*4;pr=b+p[which][r%4];}for(unsigned c=0;c<N;c++){unsigned pc=c;if(c<240&&c/80==comp){unsigned b=(c/4)*4;pc=b+p[which][c%4];}if(bit(m,pr,pc)!=bit(m,r,c))return 0;}}return 1;}
static unsigned symmetry_score(const Mat*m){unsigned s=0;for(unsigned comp=0;comp<3;comp++){int ok=1;for(unsigned g=0;g<3;g++)if(!commute_component_v4(m,comp,g))ok=0;if(ok)s++;}return s;}
int main(void){Mat obs;build_observed(&obs);unsigned rd0[N],cd0[N];rowdeg(&obs,rd0);coldeg(&obs,cd0);unsigned observed=symmetry_score(&obs);if(observed!=3){fprintf(stderr,"observed symmetry score %u\n",observed);return 2;}unsigned ge=0,eq=0,min=99,max=0,sum=0;for(unsigned k=0;k<DRAWS;k++){Mat nul;degree_preserving_scramble(&obs,&nul,20000);unsigned rd[N],cd[N];rowdeg(&nul,rd);coldeg(&nul,cd);for(unsigned i=0;i<N;i++)if(rd[i]!=rd0[i]||cd[i]!=cd0[i]){fprintf(stderr,"degree mismatch draw=%u i=%u\n",k,i);return 3;}unsigned s=symmetry_score(&nul);if(s>=observed)ge++;if(s==observed)eq++;if(s<min)min=s;if(s>max)max=s;sum+=s;}double p=(1.0+ge)/(1.0+DRAWS);printf("observed=%u draws=%u null_min=%u null_max=%u null_sum=%u ge=%u equal=%u empirical_p=%.9f\n",observed,DRAWS,min,max,sum,ge,eq,p);if(ge==DRAWS){fprintf(stderr,"null failed to disrupt fine symmetry\n");return 4;}puts("ok preserved=row+column-degrees scrambled=fine-incidence tested=V4-component-symmetry");return 0;}
