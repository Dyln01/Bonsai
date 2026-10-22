#!/usr/bin/env python3
"""patch_tq_q38.py — Phase 4c slice 3: KV_TQ tiers + absorbed rotation for qwen38.

Applies AFTER patch_kv8_q38.py (extends its branches). Knobs mirror qwen36:
KV_TQ=<bits 2..8>, KV_TQ_CODEC=0|1 (1 = rotated-int4, default), plus
Q38_TQ_ABSORB=0|1 (default 1): the 3b-rot pattern — rotate each query head row
once, dot raw rotated-space levels, accumulate V in rotated space, unrotate the
output once per (position, head). Codec math is kv_tq.h (engine-agnostic, gated
by kv_tq_parity); absorb math gated by tq_absorb_parity.

Gates (CI): tq-off self-determinism; absorb 0/1 id-identity + prefill cosine
>=0.999; tq-on vs off prefill cosine inside the e4m3/Lloyd-Max noise band.

Usage: python3 patch_tq_q38.py <colibri-c-dir>
Backups: *.q38pretq. Refuses twice.
"""
import os
import shutil
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/colibri/c")
FILES = {"qwen38.c": os.path.join(D, "qwen38.c"),
         "qwen38_core.h": os.path.join(D, "qwen38_core.h")}
srcs = {k: open(p, encoding="utf8", errors="surrogateescape").read() for k, p in FILES.items()}
if "tq_absorb" in srcs["qwen38.c"]:
    raise SystemExit("patch_tq_q38: already applied")
if "m->kv8" not in srcs["qwen38.c"]:
    raise SystemExit("patch_tq_q38: apply patch_kv8_q38.py first")

EDITS = []
def edit(fn, old, new):
    EDITS.append((fn, old, new))

edit("qwen38_core.h", '#include "kv8_fused.h"    /* Bonsai KV8 e4m3 decode helpers (overlay from Bonsai engine/) */\n',
     '#include "kv8_fused.h"    /* Bonsai KV8 e4m3 decode helpers (overlay from Bonsai engine/) */\n'
     '#include "kv_tq.h"        /* Bonsai KV_TQ codecs + rotation (overlay) */\n')

edit("qwen38_core.h", "    int kv8; float **Kscale, **Vscale;   /* Bonsai KV8 tier (patch_kv8_q38) */\n",
     "    int kv8; float **Kscale, **Vscale;   /* Bonsai KV8 tier (patch_kv8_q38) */\n"
     "    int kv_tq, tq_bits, tq_codec, tq_absorb, kvq_row_b;   /* Bonsai KV_TQ tier (patch_tq_q38) */\n")

edit("qwen38.c",
     """      if (m->kv8 && getenv("SERVE") && getenv("SERVE")[0] == '1') {
          fprintf(stderr, "[kv8] serve/mux unsupported in slice 2 — rerun without KV8=1\\n"); exit(1); } }
""",
     """      if (m->kv8 && getenv("SERVE") && getenv("SERVE")[0] == '1') {
          fprintf(stderr, "[kv8] serve/mux unsupported in slice 2 — rerun without KV8=1\\n"); exit(1); } }
    { const char *te = getenv("KV_TQ"); int bits = te && *te ? atoi(te) : 0;
      if (bits && m->kv8) { fprintf(stderr, "[kvq] KV8 and KV_TQ are mutually exclusive; KV_TQ wins\\n"); m->kv8 = 0; }
      m->kv_tq = bits >= 2 && bits <= 8;
      if (bits && !m->kv_tq) { fprintf(stderr, "[kvq] KV_TQ must be 2..8\\n"); exit(1); }
      const char *ce = getenv("KV_TQ_CODEC"); m->tq_codec = (ce && *ce == '0') ? 0 : 1;
      const char *ae = getenv("Q38_TQ_ABSORB"); m->tq_absorb = (ae && ae[0] == '0') ? 0 : 1;
      m->tq_bits = bits ? bits : 4;
      m->kvq_row_b = m->kv_tq ? coli_kvq_row_bytes(c->head_dim, m->tq_bits, m->tq_codec) : 0;
      if (m->kv_tq && getenv("SERVE") && getenv("SERVE")[0] == '1') {
          fprintf(stderr, "[kvq] serve/mux unsupported in slice 3 — rerun without KV_TQ\\n"); exit(1); } }
""")

edit("qwen38_core.h",
     "        if(m->kv8){size_t by=(size_t)c->kv_heads*m->max_t*c->head_dim;m->K[i]=(float*)malloc(by);m->V[i]=(float*)malloc(by);m->Kscale[i]=falloc((int64_t)c->kv_heads*m->max_t);m->Vscale[i]=falloc((int64_t)c->kv_heads*m->max_t);}\n",
     "        if(m->kv8||m->kv_tq){size_t rb=(size_t)(m->kv_tq?m->kvq_row_b:c->head_dim);size_t by=(size_t)c->kv_heads*m->max_t*rb;m->K[i]=(float*)malloc(by);m->V[i]=(float*)malloc(by);m->Kscale[i]=falloc((int64_t)c->kv_heads*m->max_t);m->Vscale[i]=falloc((int64_t)c->kv_heads*m->max_t);}\n")

edit("qwen38_core.h",
     """            if(m->kv8){uint8_t*K8=(uint8_t*)Kl,*V8=(uint8_t*)Vl;float*Ks=m->Kscale[layer],*Vs=m->Vscale[layer];
                Ks[(int64_t)h*m->kv_cap+pos]=coli_kv8_quant_row(kh,K8+((int64_t)h*m->kv_cap+pos)*D,D);
                Vs[(int64_t)h*m->kv_cap+pos]=coli_kv8_quant_row(vp+(int64_t)s*KVH*D+(int64_t)h*D,V8+((int64_t)h*m->kv_cap+pos)*D,D);}
""",
     """            if(m->kv8){uint8_t*K8=(uint8_t*)Kl,*V8=(uint8_t*)Vl;float*Ks=m->Kscale[layer],*Vs=m->Vscale[layer];
                Ks[(int64_t)h*m->kv_cap+pos]=coli_kv8_quant_row(kh,K8+((int64_t)h*m->kv_cap+pos)*D,D);
                Vs[(int64_t)h*m->kv_cap+pos]=coli_kv8_quant_row(vp+(int64_t)s*KVH*D+(int64_t)h*D,V8+((int64_t)h*m->kv_cap+pos)*D,D);}
            else if(m->kv_tq){uint8_t*K8=(uint8_t*)Kl,*V8=(uint8_t*)Vl;float*Ks=m->Kscale[layer],*Vs=m->Vscale[layer];int64_t rb=m->kvq_row_b;
                Ks[(int64_t)h*m->kv_cap+pos]=coli_kvq_quant_row(kh,K8+((int64_t)h*m->kv_cap+pos)*rb,D,m->tq_bits,m->tq_codec);
                Vs[(int64_t)h*m->kv_cap+pos]=coli_kvq_quant_row(vp+(int64_t)s*KVH*D+(int64_t)h*D,V8+((int64_t)h*m->kv_cap+pos)*rb,D,m->tq_bits,m->tq_codec);}
""")

# read side: dequant scratch covers kv8 and tq (absorb = rotated-space levels)
edit("qwen38_core.h",
     """        float *Kd=NULL,*Vd=NULL;
        if(m->kv8){Kd=falloc((int64_t)KVH*nsel*D);Vd=falloc((int64_t)KVH*nsel*D);
            for(int kvh=0;kvh<KVH;kvh++)for(int j=0;j<nsel;j++){
                kv8_dequant_row_fast((const uint8_t*)Kl+((int64_t)kvh*m->kv_cap+selected[j])*D,m->Kscale[layer][(int64_t)kvh*m->kv_cap+selected[j]],Kd+((int64_t)kvh*nsel+j)*D,D);
                kv8_dequant_row_fast((const uint8_t*)Vl+((int64_t)kvh*m->kv_cap+selected[j])*D,m->Vscale[layer][(int64_t)kvh*m->kv_cap+selected[j]],Vd+((int64_t)kvh*nsel+j)*D,D);}}
""",
     """        float *Kd=NULL,*Vd=NULL;
        int absorb=m->kv_tq&&m->tq_absorb&&m->tq_codec==1;
        if(m->kv8||m->kv_tq){int64_t rb=m->kv_tq?m->kvq_row_b:D;Kd=falloc((int64_t)KVH*nsel*D);Vd=falloc((int64_t)KVH*nsel*D);
            for(int kvh=0;kvh<KVH;kvh++)for(int j=0;j<nsel;j++){
                const uint8_t *kr=(const uint8_t*)Kl+((int64_t)kvh*m->kv_cap+selected[j])*rb;
                const uint8_t *vr=(const uint8_t*)Vl+((int64_t)kvh*m->kv_cap+selected[j])*rb;
                float ksc=m->Kscale[layer][(int64_t)kvh*m->kv_cap+selected[j]];
                float vsc=m->Vscale[layer][(int64_t)kvh*m->kv_cap+selected[j]];
                if(m->kv_tq&&absorb){q38_tq_rot_noscale(kr,Kd+((int64_t)kvh*nsel+j)*D,D);q38_tq_rot_noscale(vr,Vd+((int64_t)kvh*nsel+j)*D,D);}
                else if(m->kv_tq){coli_kvq_dequant_row(kr,ksc,Kd+((int64_t)kvh*nsel+j)*D,D,m->tq_bits,m->tq_codec);coli_kvq_dequant_row(vr,vsc,Vd+((int64_t)kvh*nsel+j)*D,D,m->tq_bits,m->tq_codec);}
                else{kv8_dequant_row_fast(kr,ksc,Kd+((int64_t)kvh*nsel+j)*D,D);kv8_dequant_row_fast(vr,vsc,Vd+((int64_t)kvh*nsel+j)*D,D);}}}
""")

# helper: rotated-space levels without radius (radius hoisted in Kscale/Vscale)
edit("qwen38_core.h", '#include "kv_tq.h"        /* Bonsai KV_TQ codecs + rotation (overlay) */\n',
     '#include "kv_tq.h"        /* Bonsai KV_TQ codecs + rotation (overlay) */\n'
     "static inline void q38_tq_rot_noscale(const uint8_t *src, float *dst, int n){\n"
     "    float inv=1.f/sqrtf((float)n);\n"
     "    for(int i=0;i<n;i++){int c=(src[i>>1]>>((i&1)*4))&0xF;dst[i]=coli_q4_lev[c]*inv;}\n"
     "}\n")

# absorb: rotate the query row, scale scores/V by the hoisted radius, unrotate output
edit("qwen38_core.h",
     "            float *qh=falloc(D);memcpy(qh,qraw,(size_t)D*sizeof(float));q38_rms0(qh,qh,l->qn,D,c->eps);q38_rope(qh,D,c->rotary_dim,pos,theta);\n",
     "            float *qh=falloc(D);memcpy(qh,qraw,(size_t)D*sizeof(float));q38_rms0(qh,qh,l->qn,D,c->eps);q38_rope(qh,D,c->rotary_dim,pos,theta);\n"
     "            if(absorb){float *qrh=falloc(D);coli_tq_rotate(qh,qrh,D,COLI_TQ_SEED);free(qh);qh=qrh;}\n")

edit("qwen38_core.h",
     "int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=m->kv8?Kd+((int64_t)khidx*nsel+j)*D:Kl+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);score[j]=a;if(a>mx)mx=a;}\n",
     "int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=(m->kv8||m->kv_tq)?Kd+((int64_t)khidx*nsel+j)*D:Kl+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);if(absorb)a*=m->Kscale[layer][(int64_t)khidx*m->kv_cap+selected[j]];score[j]=a;if(a>mx)mx=a;}\n")

edit("qwen38_core.h",
     "for(int j=0;j<nsel;j++){float a=score[j]/den;const float *vh=m->kv8?Vd+((int64_t)khidx*nsel+j)*D:Vl+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}\n",
     "for(int j=0;j<nsel;j++){float a=score[j]/den;if(absorb)a*=m->Vscale[layer][(int64_t)khidx*m->kv_cap+selected[j]];const float *vh=(m->kv8||m->kv_tq)?Vd+((int64_t)khidx*nsel+j)*D:Vl+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}\n"
     "            if(absorb)coli_tq_unrotate(oh,D,COLI_TQ_SEED);\n")

for fn, old, new in EDITS:
    if srcs[fn].count(old) != 1:
        raise SystemExit("patch_tq_q38: %s anchor count=%d (want 1): %r" % (fn, srcs[fn].count(old), old[:80]))
    srcs[fn] = srcs[fn].replace(old, new, 1)

for k, p in FILES.items():
    shutil.copy(p, p + ".q38pretq")
    open(p, "wb").write(srcs[k].encode("utf8", "surrogateescape"))
print("patch_tq_q38 applied (%d edits, backups *.q38pretq)" % len(EDITS))
