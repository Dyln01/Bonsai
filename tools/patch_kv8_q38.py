#!/usr/bin/env python3
"""patch_kv8_q38.py — Phase 4c slice 2: KV8 tier for the qwen38 engine.

Applies to a colibri checkout (bf24429): edits qwen38.c (knob, fields, alloc,
prefix guard) and qwen38_core.h (codec include, quantize-on-write, dequant-
once-per-selected-row-on-read in the QSA gather). Design: docs/CROSS_ENGINE_KVQ.md.

Semantics mirror qwen36 KV8: fp8-e4m3 rows + per-row f32 scale in the same
pool slots (pool becomes byte-wide under KV8=1); the QSA read path dequantizes
each SELECTED row once per (position, kv-head) into scratch and every query
head of the group dots the scratch — the sparse gather makes the 3a-style
amortization fall out for free.

Slice-2 scope limits (enforced, loud): SERVE/mux and kv_prefix reuse refuse
KV8=1 (the segment/prefix machinery exposes K/V as float spans); MTP-head
containers refuse KV8=1. KV8=0 is byte-inert (CI id-identity vs stock binary).

Usage: python3 patch_kv8_q38.py <colibri-c-dir>
Backups: <dir>/qwen38.c.q38prekv8, <dir>/qwen38_core.h.q38prekv8. Refuses twice.
"""
import os
import shutil
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/colibri/c")
FILES = {
    "qwen38.c": os.path.join(D, "qwen38.c"),
    "qwen38_core.h": os.path.join(D, "qwen38_core.h"),
}
srcs = {k: open(p, encoding="utf8", errors="surrogateescape").read() for k, p in FILES.items()}
if "kv8" in srcs["qwen38.c"] and "m->kv8" in srcs["qwen38.c"]:
    raise SystemExit("patch_kv8_q38: already applied")

EDITS = []
def edit(fn, old, new):
    EDITS.append((fn, old, new))

# ---- qwen38.c ----
edit("qwen38_core.h",
     "    const Q38Row *mux_rows;\n",
     "    const Q38Row *mux_rows;\n"
     "    int kv8; float **Kscale, **Vscale;   /* Bonsai KV8 tier (patch_kv8_q38) */\n")

edit("qwen38.c",
     '    m->max_t = q38_context_total(c, np, n_new, "generation");\n',
     '    m->max_t = q38_context_total(c, np, n_new, "generation");\n'
     '    { const char *kve = getenv("KV8"); m->kv8 = kve && kve[0] == \'1\';\n'
     '      if (m->kv8 && getenv("SERVE") && getenv("SERVE")[0] == \'1\') {\n'
     '          fprintf(stderr, "[kv8] serve/mux unsupported in slice 2 — rerun without KV8=1\\n"); exit(1); } }\n')

edit("qwen38.c",
     "q38_prefix_copy_state(m,0);m->kv_len=g_q38_prefix.len;\n",
     "if(!m->kv8){q38_prefix_copy_state(m,0);m->kv_len=g_q38_prefix.len;}else{m->kv_len=0;}  /* KV8: prefix reuse off in slice 2 */\n")

# ---- qwen38_core.h ----
edit("qwen38_core.h",
     '#include "spec_draft.h"   /* prompt-lookup drafts and the gate that decides when drafting pays */\n',
     '#include "spec_draft.h"   /* prompt-lookup drafts and the gate that decides when drafting pays */\n'
     '#include "kv8_fused.h"    /* Bonsai KV8 e4m3 decode helpers (overlay from Bonsai engine/) */\n')

edit("qwen38_core.h",
     "    m->K=(float**)calloc((size_t)c->layers+1,sizeof(float*));m->V=(float**)calloc((size_t)c->layers+1,sizeof(float*));m->IK=(float**)calloc((size_t)c->layers+1,sizeof(float*));\n",
     "    m->K=(float**)calloc((size_t)c->layers+1,sizeof(float*));m->V=(float**)calloc((size_t)c->layers+1,sizeof(float*));m->IK=(float**)calloc((size_t)c->layers+1,sizeof(float*));\n"
     "    m->Kscale=(float**)calloc((size_t)c->layers+1,sizeof(float*));m->Vscale=(float**)calloc((size_t)c->layers+1,sizeof(float*));\n")

edit("qwen38_core.h",
     "    if(m->K){for(int i=0;i<rows;i++){free(m->K[i]);free(m->V[i]);free(m->IK[i]);}free(m->K);free(m->V);free(m->IK);}\n",
     "    if(m->K){for(int i=0;i<rows;i++){free(m->K[i]);free(m->V[i]);free(m->IK[i]);if(m->Kscale)free(m->Kscale[i]);if(m->Vscale)free(m->Vscale[i]);}free(m->K);free(m->V);free(m->IK);if(m->Kscale)free(m->Kscale);if(m->Vscale)free(m->Vscale);}\n")

edit("qwen38_core.h",
     "        m->K[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->V[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->IK[i]=falloc((int64_t)m->max_t*c->idx_dim);\n",
     "        if(m->kv8){size_t by=(size_t)c->kv_heads*m->max_t*c->head_dim;m->K[i]=(float*)malloc(by);m->V[i]=(float*)malloc(by);m->Kscale[i]=falloc((int64_t)c->kv_heads*m->max_t);m->Vscale[i]=falloc((int64_t)c->kv_heads*m->max_t);}\n"
     "        else{m->K[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->V[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);}\n"
     "        m->IK[i]=falloc((int64_t)m->max_t*c->idx_dim);\n")

edit("qwen38_core.h",
     "            memcpy(Kl+((int64_t)h*m->kv_cap+pos)*D,kh,(size_t)D*sizeof(float));\n"
     "            memcpy(Vl+((int64_t)h*m->kv_cap+pos)*D,vp+(int64_t)s*KVH*D+(int64_t)h*D,(size_t)D*sizeof(float));\n",
     "            if(m->kv8){uint8_t*K8=(uint8_t*)Kl,*V8=(uint8_t*)Vl;float*Ks=m->Kscale[layer],*Vs=m->Vscale[layer];\n"
     "                Ks[(int64_t)h*m->kv_cap+pos]=coli_kv8_quant_row(kh,K8+((int64_t)h*m->kv_cap+pos)*D,D);\n"
     "                Vs[(int64_t)h*m->kv_cap+pos]=coli_kv8_quant_row(vp+(int64_t)s*KVH*D+(int64_t)h*D,V8+((int64_t)h*m->kv_cap+pos)*D,D);}\n"
     "            else{memcpy(Kl+((int64_t)h*m->kv_cap+pos)*D,kh,(size_t)D*sizeof(float));\n"
     "            memcpy(Vl+((int64_t)h*m->kv_cap+pos)*D,vp+(int64_t)s*KVH*D+(int64_t)h*D,(size_t)D*sizeof(float));}\n")

edit("qwen38_core.h",
     "        index_dt+=now_s()-phase_started; phase_started=now_s();\n"
     "        for(int h=0;h<QH;h++){\n",
     "        index_dt+=now_s()-phase_started; phase_started=now_s();\n"
     "        float *Kd=NULL,*Vd=NULL;\n"
     "        if(m->kv8){Kd=falloc((int64_t)KVH*nsel*D);Vd=falloc((int64_t)KVH*nsel*D);\n"
     "            for(int kvh=0;kvh<KVH;kvh++)for(int j=0;j<nsel;j++){\n"
     "                kv8_dequant_row_fast((const uint8_t*)Kl+((int64_t)kvh*m->kv_cap+selected[j])*D,m->Kscale[layer][(int64_t)kvh*m->kv_cap+selected[j]],Kd+((int64_t)kvh*nsel+j)*D,D);\n"
     "                kv8_dequant_row_fast((const uint8_t*)Vl+((int64_t)kvh*m->kv_cap+selected[j])*D,m->Vscale[layer][(int64_t)kvh*m->kv_cap+selected[j]],Vd+((int64_t)kvh*nsel+j)*D,D);}}\n"
     "        for(int h=0;h<QH;h++){\n")

edit("qwen38_core.h",
     "int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=Kl+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);score[j]=a;if(a>mx)mx=a;}\n",
     "int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=m->kv8?Kd+((int64_t)khidx*nsel+j)*D:Kl+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);score[j]=a;if(a>mx)mx=a;}\n")

edit("qwen38_core.h",
     "for(int j=0;j<nsel;j++){float a=score[j]/den;const float *vh=Vl+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}\n",
     "for(int j=0;j<nsel;j++){float a=score[j]/den;const float *vh=m->kv8?Vd+((int64_t)khidx*nsel+j)*D:Vl+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}\n")

edit("qwen38_core.h",
     "        free(qidx);free(pool);free(selected);\n",
     "        free(qidx);free(pool);free(selected);if(Kd){free(Kd);free(Vd);}\n")

for fn, old, new in EDITS:
    if srcs[fn].count(old) != 1:
        raise SystemExit("patch_kv8_q38: %s anchor count=%d (want 1): %r" % (fn, srcs[fn].count(old), old[:70]))
    srcs[fn] = srcs[fn].replace(old, new, 1)

for k, p in FILES.items():
    shutil.copy(p, p + ".q38prekv8")
    open(p, "wb").write(srcs[k].encode("utf8", "surrogateescape"))
print("patch_kv8_q38 applied (%d edits across 2 files, backups *.q38prekv8)" % len(EDITS))
