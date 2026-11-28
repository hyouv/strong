# Stronghold Protocol Simulator (`act3autochess@V3_1`)

Headless game simulator for **明日方舟：卫戍协议（规程·序列Ⅰ `act3autochess@V3_1`）**, migrated and refactored from `stronghold-training`.

## Directory Structure

```text
.
├── config/
│   └── regulation.json          # Pinned ruleset metadata (act3autochess@V3_1)
├── simulator/
│   ├── engine/                  # Vendored Stronghold-Protocol JS core (~12.5 MB)
│   └── js/                      # Refactored headless JS worker & RL interface
│       ├── context.mjs          # Pure-JS rules/data loader (no git subprocesses)
│       ├── strategy_actions.mjs # Strategy/command action enumeration & C2S validation
│       ├── candidates.mjs       # Optimized legal action enumeration (exact order preserved)
│       ├── examples.mjs         # Optimized bot demonstration generator
│       ├── session.mjs          # Encapsulated RoomSession class
│       └── worker.mjs           # NDJSON stdio worker CLI
├── stronghold_sim/              # Pure-Python simulator client & parallel pool (no PyTorch dependency)
│   ├── protocol.py
│   ├── runtime.py               # Project-local Node.js discovery & CPU affinity helpers
│   └── worker.py                # RoomClient & RoomPool
└── tests/
    ├── test_js_simulator.mjs    # Node unit tests
    ├── test_python_client.py    # Python client & pool tests
    └── test_parity.py           # Strict differential parity tests vs original simulator
```

## Key Refactorings (Zero Behavior Change)

1. **Minimal self-contained engine (`simulator/engine/`)**: Extracted only the runtime simulation core (`server/data.js`, `server/match/`, `server/sim/`, `server/ai/`, `shared/`, `types/`, and `data/versions/act3autochess-v3_1/`), reducing footprint from ~85 MB (2,230 files + nested `.git`) to ~12.5 MB (544 files) with zero npm dependencies and unmodified game rules.
2. **Removed `git` subprocess calls on worker startup (`simulator/js/context.mjs`)**: Validates version and upstream revision directly from JSON metadata instead of spawning `git rev-parse` and `git status` per worker process.
3. **Encapsulated session state (`simulator/js/session.mjs`)**: Replaced module-level globals in `worker.mjs` with a reusable `RoomSession` class, and skipped per-candidate `JSON.stringify` filtering when no actions have been rejected (`rejected.size === 0`).
4. **Hot-path optimizations in `candidates.mjs` & `examples.mjs`**: Hoisted repeated `parseKey`, board UID set, chess piece filtering, and shop slot checks out of inner loops; replaced `structuredClone` + `JSON.stringify` in `examples.mjs` with direct pre-action JSON serialization and fast action-type pre-filtering.
5. **Decoupled Python package (`stronghold_sim/`)**: Separated worker lifecycle (`RoomClient` / `RoomPool`) from PyTorch and RL training code, added automatic Node.js discovery (`.tools/node`, `STRONGHOLD_NODE`, or `PATH`), context manager support, optional CPU affinity/logging, and stderr diagnostics on worker failures.

## Local Development

All dependencies are confined inside the repository:
- Python virtual environment: `.venv/`
- Portable Node.js v22 runtime: `.tools/node/`

Run tests:

```powershell
.\.tools\node\node.exe --test tests/test_js_simulator.mjs tests/test_headless.mjs
.\.venv\Scripts\python.exe -m pytest -v
```

Simulator profiling, fixed-action replay benchmarks, and the headless optimizations are
documented in [docs/SIMULATOR_PERFORMANCE.md](docs/SIMULATOR_PERFORMANCE.md).
