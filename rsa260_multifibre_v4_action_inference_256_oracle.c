#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N=256, W=4, CLASSES=60, COMPS=3, PERCOMP=20, TAIL=16 };
typedef struct { uint64_t row[N][W]; } Mat;
static unsigned bit(const Mat*m,unsigned r,unsigned c){return (unsigned)((m->row[r][c>>6]>>(c&63u))&1u);}
static void setb(Mat*m,unsigned r,unsigned c){m->row[r][c>>6]|=UINT64_C(1)<<(c&63u);}
static unsigned ord(const unsigned*p){unsigned cur[4]={0,1,2,3};for(unsigned t=1;t<=12;t++){unsigned nxt[4];for(unsigned i=0;i<4;i++)nxt[i]=p[cur[i]];for(unsigned i=0;i<4;i++)cur[i]=nxt[i];int id=1;for(unsigned i=0;i<4;i++)if(cur[i]!=i)id=0;if(id)return t;}return 0;}
static void build_primary(Mat*m){memset(m,0,sizeof(*m));for(unsigned c=0;c<CLASSES;c++){unsigned b=4*c;for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++)if(i!=j)setb(m,b+i,b+j);if((c%PERCOMP)+1<PERCOMP)for(unsigned i=0;i<4;i++)setb(m,b+i,b+4+i);}for(unsigned i=240;i+1<N;i++)setb(m,i,i+1);}
static void build_fibres(Mat f[3]){memset(f,0,sizeof(Mat)*3);static const unsigned mate[3][4]={{1,0,3,2},{2,3,0,1},{3,2,1,0}};for(unsigned k=0;k<3;k++)for(unsigned c=0;c<CLASSES;c++){unsigned b=4*c;for(unsigned i=0;i<4;i++)setb(&f[k],b+i,b+mate[k][i]);}}
static int preserve_local(const Mat*m,unsigned cls,const unsigned*p){unsigned b=4*cls;for(unsigned i=0;i<4;i++)for(unsigned j=0;j<4;j++)if(bit(m,b+i,b+j)!=bit(m,b+p[i],b+p[j]))return 0;return 1;}
static void enum_rec(unsigned d,unsigned used,unsigned p[4],const Mat*m,const Mat f[3],unsigned cls,unsigned*primary,unsigned*all,unsigned*o4,unsigned*o2){if(d==4){if(!preserve_local(m,cls,p))return;(*primary)++;for(unsigned k=0;k<3;k++)if(!preserve_local(&f[k],cls,p))return;(*all)++;unsigned o=ord(p);if(o==4)(*o4)++;if(o==2)(*o2)++;return;}for(unsigned v=0;v<4;v++)if(!(used&(1u<<v))){p[d]=v;enum_rec(d+1,used|(1u<<v),p,m,f,cls,primary,all,o4,o2);}}
static unsigned map_sel(unsigned x,unsigned comp,const unsigned*p){if(x>=240)return x;unsigned cls=x/4;if(cls/PERCOMP!=comp)return x;unsigned b=4*cls;return b+p[x%4];}
static int global_commutes(const Mat*m,unsigned comp,const unsigned*p){for(unsigned r=0;r<N;r++){unsigned pr=map_sel(r,comp,p);for(unsigned c=0;c<N;c++){unsigned pc=map_sel(c,comp,p);if(bit(m,pr,pc)!=bit(m,r,c))return 0;}}return 1;}
int main(void){Mat M,F[3];build_primary(&M);build_fibres(F);unsigned p[4],primary=0,all=0,o4=0,o2=0;enum_rec(0,0,p,&M,F,0,&primary,&all,&o4,&o2);printf("primary_local_group=%u intersected_group=%u order2=%u order4=%u\n",primary,all,o2,o4);if(primary!=24||all!=4||o2!=3||o4!=0)return 2;static const unsigned gens[3][4]={{1,0,3,2},{2,3,0,1},{3,2,1,0}};unsigned individual=0,closed=0;for(unsigned g=0;g<3;g++){unsigned q[4]={0,1,2,3};memcpy(q,gens[g],sizeof q);if(global_commutes(&M,0,q))closed++;}unsigned b=0;int single_ok=1;for(unsigned r=0;r<N&&single_ok;r++)for(unsigned c=0;c<N;c++){unsigned pr=r,pc=c;if(r>=b&&r<b+4)pr=b+gens[0][r-b];if(c>=b&&c<b+4)pc=b+gens[0][c-b];if(bit(&M,pr,pc)!=bit(&M,r,c)){single_ok=0;break;}}if(single_ok)individual=1;printf("classes=%u components=%u per_component=%u tail=%u single_class_commuting=%u closed_v4_generators_commuting=%u\n",CLASSES,COMPS,PERCOMP,TAIL,individual,closed);if(individual!=0||closed!=3)return 3;puts("ok multifibre_intersection=S4_to_V4 requirement_closure=20classes");return 0;}
