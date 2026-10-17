/* kv_tq_parity_q36.c — gates for the qwen36 KV_TQ port (same role as kv8_parity.c).
 * The codec itself ships in kv_tq.h (colibri GLM side, tests/test_kv_tq.c); this
 * file gates what the q36 PORT depends on, at the two production widths:
 *   1. FWHT self-inverse + norm preservation (n = 64 student, 128 real-27B).
 *   2. Radius exactness: quant returns the row's L2 norm (both codecs) — it
 *      rides the SAME Kscale/Vscale per-row slot KV8 already allocates.
 *   3. Direction coverage: mean rel L2 err bounded — codec 1 (rotated int4)
 *      < 0.15, PolarQuant-4 < 0.45, PolarQuant-3 < 0.65 (thresholds shared
 *      with the GLM-side test).
 *   4. Scale-hoisting linearity — the port's core premise: dequant(r=1)*s ==
 *      dequant(r=s) element-wise for random rows and scales, because the
 *      radius scales the reconstruction linearly. The engine decodes at
 *      radius 1 and multiplies by Kscale/Vscale outside the element loop,
 *      exactly like KV8's per-row scale.
 *   5. Inert rows: zero / NaN / Inf quantize to radius 0 and decode to exact
 *      zeros (a poisoned row must never reach the FWHT).
 *   6. Accounting: packed row bytes < fp8 row bytes < f32 row bytes at both
 *      widths, and the int4 codec is exactly n/2.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kv_tq.h"

static int fails = 0;
#define CHECK(cond, ...) do{ if(!(cond)){ fails++; \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fputc('\n',stderr);} }while(0)

static uint32_t rng = 0x12345678u;
static float frand(void){ rng = rng*1664525u+1013904223u; return (float)((rng>>8)&0xFFFFFF)/(float)0x1000000 * 2.f - 1.f; }
static float gauss(void){ float u=(frand()+1.f)*0.5f+1e-7f, v=(frand()+1.f)*0.5f; return sqrtf(-2.f*logf(u))*cosf(2.f*COLI_TQ_PI*v); }
static double norm2d(const float *a,int n){ double s=0; for(int i=0;i<n;i++) s+=(double)a[i]*a[i]; return sqrt(s); }
static double l2d(const float *a,const float *b,int n){ double s=0; for(int i=0;i<n;i++){ double d=(double)a[i]-b[i]; s+=d*d; } return sqrt(s); }

static float avg_relerr(int n,int bits,int codec,int rows){
    int rb = coli_kvq_row_bytes(n,bits,codec);
    float *x=malloc(sizeof(float)*n), *y=malloc(sizeof(float)*n);
    uint8_t *q=malloc((size_t)rb);
    double acc=0;
    for(int r=0;r<rows;r++){
        for(int i=0;i<n;i++) x[i]=gauss()*3.7f;
        float rad=coli_kvq_quant_row(x,q,n,bits,codec);
        CHECK(rad > 0.f, "n=%d codec=%d row %d radius %g", n, codec, r, rad);
        coli_kvq_dequant_row(q,rad,y,n,bits,codec);
        double nx=norm2d(x,n); acc += (nx>0)? l2d(x,y,n)/nx : 0;
    }
    free(x); free(y); free(q);
    return (float)(acc/rows);
}

int main(void){
    for (int n = 64; n <= 128; n <<= 1) {
        /* 1. FWHT self-inverse + orthonormal */
        {
            float a[128], b[128];
            for(int i=0;i<n;i++){ a[i]=gauss(); b[i]=a[i]; }
            coli_tq_fwht(b,n); coli_tq_fwht(b,n);
            CHECK(l2d(a,b,n) <= 1e-4*(norm2d(a,n)+1.0), "n=%d FWHT self-inverse err %g", n, l2d(a,b,n));
            for(int i=0;i<n;i++) b[i]=a[i];
            coli_tq_fwht(b,n);
            CHECK(fabs(norm2d(b,n)-norm2d(a,n)) <= 1e-4*(norm2d(a,n)+1.0), "n=%d FWHT preserves norm", n);
        }
        /* 6. accounting */
        {
            int rb_c1 = coli_kvq_row_bytes(n,4,1);
            int rb_p4 = coli_kvq_row_bytes(n,4,0);
            int rb_p3 = coli_kvq_row_bytes(n,3,0);
            CHECK(rb_c1 == n/2, "n=%d codec1 row bytes %d != n/2=%d", n, rb_c1, n/2);
            CHECK(rb_p4 < n, "n=%d polar-4 row %d B not < fp8 %d B", n, rb_p4, n);
            CHECK(rb_p3 < rb_p4, "n=%d polar-3 %d not < polar-4 %d", n, rb_p3, rb_p4);
            fprintf(stderr, "[info] n=%d row bytes: f32=%d fp8=%d polar3=%d polar4=%d int4=%d\n",
                    n, n*4, n, rb_p3, rb_p4, rb_c1);
        }
        /* 3. direction coverage, both codecs */
        {
            float e_c1 = avg_relerr(n,4,1,300);
            float e_p4 = avg_relerr(n,4,0,300);
            float e_p3 = avg_relerr(n,3,0,300);
            fprintf(stderr,"[info] n=%d mean rel L2 err: rot-int4=%.4f polar4=%.4f polar3=%.4f\n", n, e_c1, e_p4, e_p3);
            CHECK(e_c1 < 0.15f, "n=%d codec1 rel err %g (bound 0.15)", n, e_c1);
            CHECK(e_c1 < e_p4, "n=%d codec1 (%g) should beat polar-4 (%g)", n, e_c1, e_p4);
            CHECK(e_p4 < 0.45f, "n=%d polar-4 rel err %g (bound 0.45)", n, e_p4);
            CHECK(e_p3 < 0.65f, "n=%d polar-3 rel err %g (bound 0.65)", n, e_p3);
        }
        /* 2+4. radius exactness + scale-hoisting linearity, both codecs */
        for (int codec = 0; codec <= 1; codec++) {
            int bits = 4;
            int rb = coli_kvq_row_bytes(n,bits,codec);
            float x[128], y1[128], y2[128], ys[128];
            uint8_t q[128];
            for(int i=0;i<n;i++) x[i]=gauss()*3.7f;
            float rad = coli_kvq_quant_row(x,q,n,bits,codec);
            CHECK(fabsf(rad-(float)norm2d(x,n)) <= 1e-4f*rad, "n=%d codec=%d radius %g vs ||x|| %g", n, codec, rad, (float)norm2d(x,n));
            (void)rb;
            coli_kvq_dequant_row(q,1.f,y1,n,bits,codec);
            for (int t=0;t<50;t++){
                float s = 0.01f + 9.f*fabsf(frand());
                coli_kvq_dequant_row(q,s,y2,n,bits,codec);
                for(int i=0;i<n;i++) ys[i] = y1[i]*s;
                double d = l2d(ys,y2,n), nr = norm2d(y2,n);
                CHECK(d <= 1e-5*(nr+1e-30), "n=%d codec=%d linearity s=%g: rel diff %g (norm %g)", n, codec, s, d/(nr+1e-30), nr);
            }
        }
        /* 5. inert rows */
        {
            uint8_t q[128]; float y[128]; float x[128];
            memset(x,0,sizeof x);
            CHECK(coli_kvq_quant_row(x,q,n,4,1) == 0.f, "n=%d zero row radius != 0", n);
            coli_kvq_dequant_row(q,0.f,y,n,4,1);
            for(int i=0;i<n;i++) CHECK(y[i]==0.f, "n=%d zero row decode[%d]=%g", n, i, y[i]);
            for(int i=0;i<n;i++) x[i] = (i&1) ? NAN : INFINITY;
            CHECK(coli_kvq_quant_row(x,q,n,4,1) == 0.f, "n=%d NaN/Inf row radius != 0", n);
            coli_kvq_dequant_row(q,0.f,y,n,4,1);
            for(int i=0;i<n;i++) CHECK(y[i]==0.f, "n=%d NaN/Inf decode[%d]=%g", n, i, y[i]);
        }
    }
    if (fails) { fprintf(stderr, "PARITY FAILED: %d gate(s)\n", fails); return 1; }
    fprintf(stderr, "PARITY OK: FWHT, radius, direction, linearity, inert, accounting — all gates passed\n");
    return 0;
}
