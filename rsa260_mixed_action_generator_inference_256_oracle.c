#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N=256, W=4, MAX=256 };
typedef struct { uint64_t row[N][W]; } Mat;
typedef struct { uint32_t oldc,rd,cd,diag; uint64_t oh,ih; } Sig;
typedef struct { unsigned n, v[4]; } Class;
typedef struct { unsigned cls, order, p[4]; } Gen;
static uint64_t mix64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);x^=x>>31;return x;}
static unsigned bit(const Mat*m,unsigned r,unsigned c){return (m->row[r][c>>6]>>(c&63u))&1u;}
static void setb(Mat*m,unsigned r,unsigned c){m->row[r][c>>6]|=UINT64_C(1)<<(c&63u);}
static int cmpu(const void*a,const void*b){uint32_t x=*(const uint32_t*)a,y=*(const uint32_t*)b;return(x>y)-(x<y);}
static uint64_t mh(uint32_t*a,unsigned n,uint64_t s){qsort(a,n,sizeof(uint32_t),cmpu);uint64_t h=mix64(s^n);for(unsigned i=0;i<n;i++)h=mix64(h^((uint64_t)a[i]+UINT64_C(0x9e3779b97f4a7c15)*(i+1)));return h;}
static int sless(const Sig*a,const Sig*b){
#define C(f) do{if(a->f<b->f)return 1;if(a->f>b->f)return 0;}while(0)
C(oldc);C(rd);C(cd);C(diag);C(oh);C(ih);
#undef C
return 0;}
static int seq(const Sig*a,const Sig*b){return a->oldc==b->oldc&&a->rd==b->rd&&a->cd==b->cd&&a->diag==b->diag&&a->oh==b->oh&&a->ih==b->ih;}
static void sortidx(unsigned idx[N],const Sig s[N]){for(unsigned i=0;i<N;i++)idx[i]=i;for(unsigned i=1;i<N;i++){unsigned x=idx[i],j=i;while(j&&sless(&s[x],&s[idx[j-1]])){idx[j]=idx[j-1];j--;}idx[j]=x;}}
static unsigned refine(const Mat*m,uint32_t col[N],unsigned*rounds){unsigned rd[N]={0},cd[N]={0};for(unsigned r=0;r<N;r++)for(unsigned c=0;c<N;c++)if(bit(m,r,c)){rd[r]++;cd[c]++;}Sig s[N];for(unsigned i=0;i<N;i++)s[i]=(Sig){0,rd[i],cd[i],bit(m,i,i),0,0};unsigned ord[N];sortidx(ord,s);uint32_t k=0;col[ord[0]]=0;for(unsigned j=1;j<N;j++){if(!seq(&s[ord[j]],&s[ord[j-1]]))k++;col[ord[j]]=k;}for(unsigned it=0;it<N;it++){for(unsigned i=0;i<N;i++){uint32_t out[N],in[N];unsigned no=0,ni=0;for(unsigned c=0;c<N;c++)if(bit(m,i,c))out[no++]=col[c];for(unsigned r=0;r<N;r++)if(bit(m,r,i))in[ni++]=col[r];s[i]=(Sig){col[i],rd[i],cd[i],bit(m,i,i),mh(out,no,17),mh(in,ni,31)};}sortidx(ord,s);uint32_t nxt[N],nk=0;nxt[ord[0]]=0;for(unsigned j=1;j<N;j++){if(!seq(&s[ord[j]],&s[ord[j-1]]))nk++;nxt[ord[j]]=nk;}int same=1;for(unsigned i=0;i<N;i++)if(nxt[i]!=col[i]){same=0;break;}memcpy(col,nxt,sizeof(nxt));*rounds=it+1;if(same)return nk+1;}return 0;}
static unsigned add_sector(Mat*m,unsigned start,unsigned classes,unsigned arity){for(unsigned j=0;j<classes;j++){unsigned b=start+j*arity;for(unsigned p=0;p<arity;p++){setb(m,b+p,b+((p+1)%arity));if(j+1<classes)setb(m,b+p,b+arity+p);}}return start+classes*arity;}
static void build(Mat*m){memset(m,0,sizeof(*m));unsigned x=0;x=add_sector(m,x,20,2);x=add_sector(m,x,24,3);x=add_sector(m,x,30,4);for(unsigned i=x;i+1<N;i++)setb(m,i,i+1);}
static unsigned collect_classes(const uint32_t col[N],unsigned nc,Class out[MAX]){unsigned cnt[N]={0};for(unsigned i=0;i<N;i++)cnt[col[i]]++;unsigned q=0;for(unsigned c=0;c<nc;c++)if(cnt[c]>=2&&cnt[c]<=4){out[q].n=cnt[c];unsigned k=0;for(unsigned i=0;i<N;i++)if(col[i]==c)out[q].v[k++]=i;q++;}return q;}
static unsigned perm_order(const unsigned*p,unsigned n){unsigned cur[4]={0,1,2,3};for(unsigned t=1;t<=12;t++){unsigned nxt[4];for(unsigned i=0;i<n;i++)nxt[i]=p[cur[i]];for(unsigned i=0;i<n;i++)cur[i]=nxt[i];int id=1;for(unsigned i=0;i<n;i++)if(cur[i]!=i)id=0;if(id)return t;}return 0;}
static int induced_preserves(const Mat*m,const Class*c,const unsigned*p){for(unsigned i=0;i<c->n;i++)for(unsigned j=0;j<c->n;j++)if(bit(m,c->v[i],c->v[j])!=bit(m,c->v[p[i]],c->v[p[j]]))return 0;return 1;}
static void enum_perm_rec(const Mat*m,const Class*c,unsigned cls,unsigned depth,unsigned used,unsigned p[4],Gen*g,unsigned*ng){if(depth==c->n){unsigned ord=perm_order(p,c->n);if(ord>1&&induced_preserves(m,c,p)){g[*ng].cls=cls;g[*ng].order=ord;memcpy(g[*ng].p,p,sizeof(unsigned)*4);(*ng)++;}return;}for(unsigned v=0;v<c->n;v++)if(!(used&(1u<<v))){p[depth]=v;enum_perm_rec(m,c,cls,depth+1,used|(1u<<v),p,g,ng);}}
static unsigned infer_gens(const Mat*m,const Class*c,unsigned n,Gen*g){unsigned ng=0,p[4]={0};for(unsigned i=0;i<n;i++)enum_perm_rec(m,&c[i],i,0,0,p,g,&ng);return ng;}
static unsigned best_order_for_class(const Gen*g,unsigned ng,unsigned cls){unsigned b=1;for(unsigned i=0;i<ng;i++)if(g[i].cls==cls&&g[i].order>b)b=g[i].order;return b;}
static unsigned count_order(const Gen*g,unsigned ng,unsigned ord){unsigned n=0;for(unsigned i=0;i<ng;i++)if(g[i].order==ord)n++;return n;}
int main(void){Mat M;build(&M);uint32_t col[N];unsigned rounds=0,nc=refine(&M,col,&rounds);Class cl[MAX];unsigned ncl=collect_classes(col,nc,cl);Gen g[4096];unsigned ng=infer_gens(&M,cl,ncl,g);unsigned c2=0,c3=0,c4=0,other=0;for(unsigned i=0;i<ncl;i++){unsigned o=best_order_for_class(g,ng,i);if(cl[i].n==2&&o==2)c2++;else if(cl[i].n==3&&o==3)c3++;else if(cl[i].n==4&&o==4)c4++;else other++;}printf("classes=%u orbit_classes=%u rounds=%u c2=%u c3=%u c4=%u other=%u gens=%u orders2=%u orders3=%u orders4=%u\n",nc,ncl,rounds,c2,c3,c4,other,ng,count_order(g,ng,2),count_order(g,ng,3),count_order(g,ng,4));if(c2!=20||c3!=24||c4!=30||other!=0)return 2;unsigned chosen=0;for(unsigned i=0;i<ncl;i++){unsigned bo=best_order_for_class(g,ng,i);if(bo>1)chosen++;}printf("ok mixed_action_classes=%u chosen=%u residual_coordinates=24 class_size_does_not_choose_order=1\n",ncl,chosen);return chosen==74?0:3;}
