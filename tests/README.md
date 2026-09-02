# Tests

- `unit/` — pure synthetic tests without LM/GPU.
- `integration/` — synthetic interaction between common components; real GPU smoke tests stay in `scripts/` and are run explicitly.
- `conformance/` — comparison with author/reference implementations and fixed vectors.
