using System;
using System.IO;
public static class Stage2DuplicateFingerprint {
    public static ulong[] Read(string path) {
        byte[] b=File.ReadAllBytes(path);
        if(b.Length==0 || b.Length%72!=0) throw new Exception("Invalid 9x8 gray frames");
        ulong[] hashes=new ulong[b.Length/72];
        for(int f=0;f<hashes.Length;f++) {
            ulong h=0;
            for(int y=0;y<8;y++) for(int x=0;x<8;x++)
                if(b[f*72+y*9+x]>b[f*72+y*9+x+1]) h|=1UL<<(y*8+x);
            hashes[f]=h;
        }
        return hashes;
    }
    public static int Distance(ulong a,ulong b) {
        ulong v=a^b; int n=0; while(v!=0){v&=v-1;n++;} return n;
    }
    // Three successive half-second samples, each <=8 bits, total <=18.
    // Output: sample offsets and summed distance; only a screening heuristic.
    public static int[] Match(ulong[] a,ulong[] b) {
        int best=193, ai=-1,bi=-1;
        for(int i=0;i+2<a.Length;i++) for(int j=0;j+2<b.Length;j++) {
            int d0=Distance(a[i],b[j]); if(d0>8)continue;
            int d1=Distance(a[i+1],b[j+1]); if(d1>8)continue;
            int d2=Distance(a[i+2],b[j+2]); if(d2>8)continue;
            int sum=d0+d1+d2;
            if(sum<=18 && sum<best){best=sum;ai=i;bi=j;}
        }
        return new int[]{ai,bi,best};
    }
}
