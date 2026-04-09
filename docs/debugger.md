# calcite-debugger

HTTP debug server for stepping through CSS execution, inspecting state, and
comparing compiled vs interpreted evaluation paths.

## Quick start

```sh
cargo run --release -p calcite-debugger -- -i path/to/program.css
```

Server starts on port 3333 (change with `-p PORT`).

## Usage

```sh
# Step forward 100 ticks
curl -X POST localhost:3333/tick -d '{"count":100}'

# See full register + property state
curl localhost:3333/state

# Jump to any tick (uses checkpoints for speed)
curl -X POST localhost:3333/seek -d '{"tick":8178}'

# Read IVT memory (INT 0x10 at 0x40 = addresses 64-71)
curl -X POST localhost:3333/memory -d '{"addr":64,"len":8}'

# Render video memory
curl -X POST localhost:3333/screen

# Compare compiled vs interpreted at current tick
curl localhost:3333/compare-paths

# Compare against a reference trace (stops at first divergence)
curl -X POST localhost:3333/compare -d '{"reference":[...],"stop_at_first":true}'

# Shutdown
curl -X POST localhost:3333/shutdown
```

## Endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| GET | `/info` | — | Session metadata, property/function counts, snapshot list |
| GET | `/state` | — | Current tick, all registers, all computed property values |
| POST | `/tick` | `{"count": N}` | Advance N ticks (default 1). Returns changes. |
| POST | `/seek` | `{"tick": N}` | Jump to tick N. Restores from nearest checkpoint, replays forward. |
| POST | `/memory` | `{"addr": N, "len": N}` | Hex + byte + word dump of memory region (default 256 bytes). |
| POST | `/screen` | `{"addr": N, "width": N, "height": N}` | Render text-mode video memory. Auto-detects config. |
| POST | `/compare` | `{"reference": [...], "stop_at_first": bool}` | Diff registers against a reference trace (JSON array of tick objects). |
| GET | `/compare-paths` | — | Run current tick through BOTH compiled and interpreted paths, diff all registers + memory. |
| POST | `/snapshot` | — | Create a manual checkpoint at the current tick. |
| GET | `/snapshots` | — | List all checkpoint ticks. |
| POST | `/shutdown` | — | Stop the server. |

## Checkpoints

Automatic checkpoints are created every `--snapshot-interval` ticks (default
1000). `/seek` uses the nearest checkpoint to avoid replaying from tick 0.
Create manual checkpoints with `/snapshot` before investigating a specific tick.

## Typical debugging workflow

### Finding a compiled vs interpreted divergence

```sh
# Start server
cargo run --release -p calcite-debugger -- -i program.css

# Binary search for first diverging tick
for tick in 0 10 100 1000 5000; do
    curl -sX POST localhost:3333/seek -d "{\"tick\":$tick}" > /dev/null
    curl -s localhost:3333/compare-paths | python3 -c "
import json,sys; d=json.load(sys.stdin)
print(f'tick {d[\"tick\"]}: {d[\"total_diffs\"]} diffs')
"
done

# Once found, inspect the divergence
curl -s localhost:3333/compare-paths | python3 -m json.tool
```

### Conformance testing against the reference emulator

The debugger is the backbone of conformance testing — tools like
`fulldiff.mjs` and `diagnose.mjs` drive it via HTTP. See
`docs/conformance-testing.md` for the full tool reference and workflows.

### Inspecting memory regions

```sh
# IVT (interrupt vector table) at 0x0000
curl -sX POST localhost:3333/memory -d '{"addr":0,"len":1024}'

# Stack (SP-relative)
SP=$(curl -s localhost:3333/state | python3 -c "import json,sys; print(json.load(sys.stdin)['registers']['SP'])")
curl -sX POST localhost:3333/memory -d "{\"addr\":$SP,\"len\":32}"

# Video memory
curl -sX POST localhost:3333/screen
```

## CLI options

```
-i, --input <PATH>              CSS file to debug
-p, --port <PORT>               HTTP port (default: 3333)
    --snapshot-interval <N>     Ticks between auto-checkpoints (default: 1000)
```
