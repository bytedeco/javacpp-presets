#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "blosc.h"
static unsigned char gen(size_t i){ return (unsigned char)((i/4)*7 + (i%4)); }
int main(int argc, char**argv){
  DIR*d=opendir(argv[1]); struct dirent*e; int ok=0,bad=0; char path[1024];
  blosc_init();
  while((e=readdir(d))){
    if(!strstr(e->d_name,".bin")) continue;
    snprintf(path,sizeof path,"%s/%s",argv[1],e->d_name);
    FILE*f=fopen(path,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); rewind(f);
    unsigned char*c=malloc(n); fread(c,1,n,f); fclose(f);
    size_t nb,cb,bs; blosc_cbuffer_sizes(c,&nb,&cb,&bs);
    unsigned char*o=malloc(nb?nb:1); int r=blosc_decompress(c,o,nb);
    int good = (r==(int)nb);
    for(size_t i=0;good&&i<nb;i++) if(o[i]!=gen(i)) good=0;
    if(good) ok++; else { bad++; if(bad<=8) printf("FAIL %s (ver byte %d) ret=%d\n",e->d_name,c[0],r);}
    free(c); free(o);
  }
  printf("c-blosc %s: %d ok, %d failed\n", BLOSC_VERSION_STRING, ok, bad);
  return bad?1:0;
}
