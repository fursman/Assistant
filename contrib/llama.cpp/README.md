# llama-server `/judge` route

`judge.patch` (against upstream llama.cpp `9f0e5098a5d5d8adb8dd5c59036ae2a38ab07db9`) and
`server-judge.h` add a prefill-only typed-question endpoint to llama-server, used by the
assistant's turn router (`turn_router.py`):

    POST /judge  {"system": "...", "context": "...", "questions": [{"id": "q1", "text": "...?", "options": ["yes", "no"]}]}

Start the server with `--judge-slots N` (reserves 1+N sequences in the same context as the
completion slot and forces `--kv-unified`). The context is prefilled once, the questions are
decoded as their own sequences in padded batched forward passes, and the option log-probs at
each answer position come back; no token is generated. `setup.sh` pins llama.cpp to that
commit and applies the patch when it builds llama-server, replacing an older judge build.

This is v3 (November 2026). Options, as environment variables or per request in
`"judge_opts"`: `LLAMA_JUDGE_FAST_TOK`, `_CUT_CTX`, `_SYS_CKPT`, `_CKPT_DEV`, `_PAD`
(none/full/bucket), `_TRIE`, `_SORT`, `_C0`, `_C1`, `_MIN_CKPT`. With MTP on a 16 GB card
use `--judge-slots 2 --ubatch-size 256` and `SYS_CKPT=0 TRIE=0 PAD=full` (see
`../qwen38.service`). `"debug": true` returns per-decode timings. `n_prefix_tokens` now
counts the tokens up to the end of the context.

Background, measurements and the standalone engine: https://github.com/fursman/Pre-fill
