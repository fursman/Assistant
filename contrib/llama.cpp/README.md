# llama-server `/judge` route

`judge.patch` (against upstream llama.cpp `e70802a01f03f0ed31a26338a5664796f3824371`) and
`server-judge.h` add a prefill-only typed-question endpoint to llama-server, used by the
assistant's turn router (`turn_router.py`):

    POST /judge  {"system": "...", "context": "...", "questions": [{"id": "q1", "text": "...?", "options": ["yes", "no"]}]}

Start the server with `--judge-slots N` (reserves 1+N sequences in the same context as the
completion slot and forces `--kv-unified`). The context is prefilled once, every question is
decoded as its own sequence in one batched forward pass, and the option log-probs at each answer
position come back; no token is generated. `setup.sh` pins llama.cpp to that commit and applies
the patch when it builds llama-server. Background, measurements and the standalone engine:
https://github.com/fursman/Pre-fill
