# KV-prefix reuse across agent turns — measured (qwen36)

**Date:** 2026-10-16 · engine `qwen36` @ bf24429 + Bonsai patch stack
**Driver:** `scripts/exp_prefix_q36.py` (raw serve protocol: SUBMIT/ACCEPT/DATA/DONE)
**Model:** synthetic Qwen3.8-27B-shaped container (24L, hidden 1024) — no download.
**Host:** WSL2, Ryzen 7 7735HS, 4 threads. Q36_MAXT=4096, max_tok as noted.

## What was never measured

The engine's `kv_prefix` machinery (kv_prefix.h) records every fed token id
where it is fed, swaps the record **with the state** when a serve_mux
conversation parks/unparks (`q36_seq_swap`), and offers an all-or-nothing
prefix match: if the next prompt begins with the exact ids the state was
built from, prefill only the new tail. It existed, the `[PREFIX]` log existed
(`COLI_PREFIX_LOG=1`), the `COLI_KV_PREFIX=0` A/B hatch existed — but no
measurement on q36 had ever shown it firing under agent-shaped traffic. This
bench does.

## Traffic shapes

- **append-only** (max_tok=1): each turn resends the whole transcript plus a
  ~104-byte tool-result tail. The single sampled token is emitted but never
  fed back, so the record is *exactly* the resent prefix — the case reuse
  exists for, and the case a synthetic container can measure id-exactly.
- **reply-inclusive** (max_tok=24): the full agent pattern including the
  model's own reply in history. On this container the reply's id-stream is
  not recoverable from text (256-piece byte tokenizer vs 4096-wide vocab:
  ids ≥ 256 decode to no bytes), so the match diverges — a synthetic artifact,
  documented below, not an engine bug.

## Measured (3 turns × 2 conversations, interleaved A1 B1 A2 B2 A3 B3)

| arm | config | c0t1 | c0t2 | c0t3 | c1t2 |
|---|---|---|---|---|---|
| **mux_on** | KV_SLOTS=2 | 4.03 s | **0.37 s (reuse 93%)** | **0.39 s (93%)** | 0.39 s (93%) |
| mux_off | KV_SLOTS=2, COLI_KV_PREFIX=0 | 4.02 s | 4.29 s | 4.85 s | 4.48 s |
| single_on | no KV_SLOTS | 4.09 s | 4.30 s (diverged) | 5.05 s (diverged) | 4.52 s |
| mux_agent | KV_SLOTS=2, max_tok=24 | 4.38 s | 4.82 s (diverged) | 4.98 s (diverged) | 4.89 s |

(All arms rc=0; full rows in `results/results_prefix_q36.json`.)

## Findings

1. **Reuse works through the slot swap, and it is a 10–12× wall-clock cut.**
   Two interleaved conversations on serve_mux: every turn ≥ 2 reuses 93% of
   its prompt and runs in ~0.4 s instead of ~4.3–4.8 s. The record follows its
   own conversation through every park/unpark (held counts match each
   conversation's own history exactly), so interleaving costs nothing.
2. **The saving compounds with conversation length.** Turn 3 (1864 tokens,
   1864-token transcript resent): 5.0 s → 0.39 s. On a streaming dense model
   the saved prefill is the dominant per-turn cost — the upstream anecdote in
   kv_prefix.h (DeepSeek V4, 82% reuse: 61 s instead of 320 s) is the same
   effect on a bigger engine.
3. **Shared system prompts reuse across conversations.** On the single-slot
   path, conversation 1's first turn reused conversation 0's identical
   1608-token system prefix (98%) — matching ids produce matching state, by
   construction safe.
4. **Interleaved conversations on ONE slot thrash** — each conversation's
   turn destroys the other's held prefix (all later turns diverge, both
   conversations re-prefill from scratch every turn). This is what KV_SLOTS
   exists to prevent; now quantified.
5. **Reply-inclusive traffic diverges on the synthetic container** — the
   record includes fed-back reply ids, and ids ≥ 256 decode to no bytes, so no
   text-level resend can reconstruct them. On a real BPE model every sampled
   id decodes to recoverable text, the resent bytes re-encode to the same
   ids, and the full agent pattern takes the 93%-reuse path. Real-model
   confirmation is queued for the gated Phase 5 (real-weights validation).

## The prompt-structuring contract for agent clients

Reuse is all-or-nothing on the **exact fed id prefix**. To keep it firing:

1. **Resend history byte-for-byte, append-only.** Any edit to an earlier
   message (rewriting, re-tokenizing, NFC drift) diverges the whole match.
2. **Keep one slot per conversation** (`KV_SLOTS` ≥ concurrent conversations,
   `Q36_SLOTS` in the launcher). A second conversation on the same slot
   evicts the first's state.
3. **Q36_MAXT must cover the whole transcript**, not just the prompt: past
   the cap the record is dropped (`pos0+n > cap → len=0`) and every turn
   re-prefills.
4. **Verify with `COLI_PREFIX_LOG=1`** — "it did not get faster" is otherwise
   indistinguishable from "reuse is not wired up". A/B against
   `COLI_KV_PREFIX=0`.

## Files

- Driver: `scripts/exp_prefix_q36.py` (arms: mux_on, mux_off, single_on,
  single_off, mux_agent)
- Results: `results/results_prefix_q36.json`
- Launcher: `Q36_SLOTS=<n>` now maps to the engine's KV_SLOTS on serve.
