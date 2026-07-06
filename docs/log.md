# Performance Log

Running notes on calcite performance work. Read this before doing any
optimisation — it has the current bottleneck analysis and the tooling
to measure impact of changes.

## Tooling reference

See [docs/benchmarking.md](benchmarking.md) for full usage of `calcite-bench`
and the Criterion benchmarks.

---

## 2026-07-06 — windowed byte array grows a writable (packed-cell) backing; affine packed families

Cross-cutting with CSS-DOS's session-writable disk (`disk.writable`,
see CSS-DOS LOGBOOK 2026-07-06). A writable window's cabinet shape:
the window dispatch's inner function reads packed `--mc{N}` cells
whose NAME indices sit at a constant offset from the key space, and
the cells' `--applySlot` write cascades key on a second, window-local
address property. Four generic (shape-only) extensions:

- **`recognise_windowed_byte_array`** (`compile.rs`): arm runs no
  longer need offset-0 anchors — any constant `offset − (key −
  window_base)` delta works (carried as `key_offset`); the inner
  function may now be a near-packed-byte dispatch, giving the
  descriptor a `PackedCells { table_id, pack }` backing next to the
  literal-array one. The `name == "--readMem"` guard is gone — the
  recogniser is attempted on any dispatch with inline exceptions
  (first success wins; purely structural).
- **`state::WindowedByteArray`** gains `backing: WindowBacking`
  (`Literal` | `PackedCells`). `read_mem` resolves window reads
  through either; **`write_mem` now routes window writes** into
  `PackedCells` backings (same key-remapped splice the cabinet's own
  write rules perform) and keeps dropping them on `Literal` (rom
  semantics).
- **`classify_near_packed_byte` / `get_or_build_packed_cell_table`**:
  a dispatch's packed arms may reference cells at `key/pack + C` for
  a constant `C` derived from the arms (was: C = 0 only). The dense
  addr table stays keyed in key-space; only the name lookup shifts.
- **`recognise_packed_broadcast`**: port FAMILIES. Cells may split
  into disjoint groups keyed on different address properties, each
  port carrying its own constant name/arith delta; `address_map` is
  now keyed by the ARITHMETIC byte address (the value the addr
  property actually takes), so the runtime is unchanged. The old
  "every port covers every cell" check became per-port completeness.
- **`fast_path` assignment runs are segmented**: a physical `--mc`
  run holding two structurally-distinct templates back to back (two
  write families) now absorbs the leading uniform segment and leaves
  the rest to the slow parser instead of bailing wholesale. On the
  writable test cabinet: 329,680 RAM cells absorbed prebuilt, 184,320
  disk cells slow-parsed then port-recognised (parse 1.7 → 2.6 s).
- **`rep_applier` Copy**: a destination range entirely inside a
  cell-backed window is serviced per-byte via `state.write_mem`
  (bulk `REP MOVSW` into the window — the BIOS write path); every
  other virtual-region destination still refuses loudly.

Genericity: no new property-name knowledge anywhere (the one literal
name test in the windowed recogniser was removed); deltas/offsets are
derived from CSS shape per table/port. Verified: 288+ unit tests
green, CSS-DOS smoke 6/6, writable e2e (`run.mjs writable` — batch
writes a file via INT 13h AH=03h, TYPEs it back) green on calcite
AND on the JS reference machine; rom-cabinet behaviour unchanged
(smoke + zero-diff shapes — rom runs recognise with `key_offset: 0`,
`Literal` backing, delta-0 single family, exactly as before).
Throughput on the writable cabinet ~290K t/s vs ~584K rom (the
shadow's extra state + per-tick remap props); rom carts unaffected.

## 2026-07-03 — deleted leftover compile diagnostics that flooded the browser console

Two ungated debug blocks in `compile.rs` are gone: the `[linear
branch]` op dump in `compile_style_condition` (an old "opcode 214
memAddr" debugging aid — up to 31 warn lines per 10..5000-op branch,
~1,100 console lines on a doom-scale cabinet) and the
`[slot-compaction]` shared-slot report (which recomputed full
liveness on every compile just to log). Both were warn-level, and the
CSS-DOS bridge worker silences info but keeps warn — so they were the
bulk of what users saw during an in-browser compile, making healthy
compiles look wedged (see CSS-DOS logbook 2026-07-03 FINDING).
`[compile detail]` info-level phase timings stay; the wasm
`compile_phase_report()` records phases via compile_stats, not by
console scraping. Verified: cargo build + `cargo test -p calcite-core
--release` green, wasm-pack rebuilt, CSS-DOS smoke green.

## 2026-06-12 — cabinet compile wall 30.0 → 10.6 s (wasm, −64%): data-as-code fast paths + copy elimination

The doom cabinet (332 MB) is ~90% byte data encoded as CSS (rom-disk
dispatch functions, memory-cell scaffolding); the compile spent most
of its wall building ASTs for that data and then deep-copying the
recognised structures. Five commits on main, each A/B'd same-day with
single `compile-only` runs per the owner's directive (compile wall is
consistent to ~±0.5 s; deltas below that are noise):

| Step | Change | wasm compile |
|---|---|---|
| baseline (`788389d`) | — | 30.0 s |
| `6228955` | fast-path: dense literal `@function` dispatch runs (`style(--k: N): V;`) absorbed at byte level, merged into DispatchTable | 31.5 s¹ |
| `22efecd` | stop cloning dispatch_tables in Compiler::new (&mut borrow); stop cloning branches of table-recognised functions | **19.6 → 10.1 s** |
| `89ce961` | FxHashMap for DispatchTable entries | 9.7 s |
| `3099813` | blank buffer-copy assignment runs (--__N*, dropped-by-name anyway); fix run-prefix split (trailing digit run) | ~10.1 s (noise)² |

¹ Step 1 alone didn't move the wasm wall because the absorbed entries
still landed in the same HashMaps and clones — but it's what made the
clone removals possible/cheap, and it cut native parse 4.7 → 3.6 s.
² Clear native win (parse 3.59 → 1.41 s, assignments 362,242 → 178,
78.2% of input blanked); wasm single runs straddle step 4's number.

Final official driver numbers (same day, same host):
`compile-only` **29.96 s (pre) → 10.64 s (post)**. JSONs: CSS-DOS
`docs/benches/compile-only-2026-06-12-fnfast-{baseline,final}.json`.

**Addendum (same day): owned from_parsed (`4b107d1`) → 4.59 s
(−85% total).** `from_parsed(&ParsedProgram)` forced clones of
everything the evaluator kept; the owned form drains dispatch
branches straight into table entries (no 656K readMem expr clones —
dispatch recognition 3.9 → 0.20 s) and moves the prebuilt packed
ports (broadcast recognition 1.5 → 0.04 s). wasm/CLI/debugger use it;
the borrow form stays as a clone-wrapper for small inputs. Driver
JSON `...fnfast-final2.json`. Remaining ~4.6 s: fast-path byte scan
1.5 s, cssparser 1.4 s, compile passes 1.7 s — next lever is a
templated-expr dispatch-run fast path (readMem mod/round pairs) or
fusing the three full-file scans.

**Phase visibility (`4cc3d1a`)**: the wasm compile breakdown was
unobservable (worker console doesn't reach the page; CDP console
capture adds ~12 s to a 30 s compile). New `compile_stats` module
records per-phase seconds; `CalciteEngine::compile_phase_report()`
returns JSON; the CSS-DOS bridge answers `{type:'phase-report'}` and
`web/tests/compile-phase-capture.playwright.mjs` prints it. The
breakdown is what redirected the work: parse was only ~3.5 s of the
31 s — `Compiler::new` table clone (11.1 s), dispatch
recognition+merge (12.3 s incl. functions clone 4.7 s) dominated.

**Where the remaining ~10 s lives** (phase report, post-landing):
fast-path byte scan 1.3 s, cssparser 1.2 s, dispatch recognition
~3-4 s (recognise_dispatch still clones 656K readMem branch Exprs;
an owned `from_parsed` could move instead), broadcast recognition
1.5 s (cloning prebuilt packed-port address maps — Arc them),
compile passes 1.6 s, engine setup remainder. A templated-expr
dispatch-run fast path (readMem's mod/round pairs) is the next
structural lever if compile wall matters more later.

**Verification.** Every step: calcite-cli full state dump at 2M ticks
byte-identical to the pre-change path on doom8088; 305 lib + 30
integration tests (7 new); CSS-DOS smoke 7/7.
`CALCITE_NO_FN_DISPATCH_FAST=1` disables the dispatch-run fast path
(native only) for future A/B. Cardinal-rule note: all recognition is
learned from input bytes (anchors + digit-run splitting), no cabinet
names hardcoded; the buffer-copy blank mirrors the evaluator's
existing name-prefix filter.

---

## 2026-06-12 — column_drawer_fast_forward DELETED: last x86-aware code block removed

Release-audit cleanup; closes the "remaining genericity residue" item
from the 2026-06-10 merge note. Deleted from
`crates/calcite-core/src/compile.rs`: the `post_tick_apply` call site,
`FusionDiag` + its thread-local diag fns, both `fusion_fastfwd_enabled`
definitions (env gate `CALCITE_FUSION_FASTFWD`), `COLUMN_DRAWER_BODY`
(21-byte doom8088 opcode pattern), `column_drawer_fast_forward()` and
`rom_match()` — ~285 lines, default-off since 2026-04-29 (perf
net-loss), hard-`false` on wasm, so no behaviour change. Also removed
the `CALCITE_FUSION_DIAG` hooks + fire-count print from
`calcite-cli/src/main.rs`. A pre-deletion audit swept crates/ for other
upstream knowledge: none found outside comments and test fixtures —
with this deletion the cardinal rule holds tree-wide.

**Verification.** `cargo build --release` green; `cargo test -p
calcite-core --release` all targets pass; CSS-DOS smoke gate run
post-deletion. (`cargo test --workspace` fails on this host compiling
`windows-sys`/`chrono` test deps — `dlltool.exe` missing — pre-existing
environment issue, unrelated.) Cross-link: CSS-DOS LOGBOOK 2026-06-12.

## 2026-06-11 — short dense dispatch chains (`f2c8615`): chain threshold 6→2 on the flat path, ~+3-5% web throughput

Follows the 2026-06-10 headroom note (BIfNEL = top op bucket, 22.6%).
Runtime adjacency profiling showed 62% of `BranchIfNotEqLit`
executions are immediately followed by another — chain-walking in
probe runs shorter than `MIN_CHAIN_LEN = 6`, which
`build_dispatch_chains` refused to convert. New `MIN_FLAT_CHAIN_LEN
= 2`: chains of 2–5 probes convert **only when the dense flat-array
path applies** (range ≤ 256, ≤ 3× count); the HashMap path keeps
threshold 6 (linear probes beat a map lookup at that size).

**doom8088.** Chains 208 → 358; dispatched ops/tick 700 → 678
(−3.1%); BIfNEL 178 → 137/tick. All short chains on this cabinet are
dense — a sparse-short-chain linear-scan variant (`small_table`) was
built, exposed a remap hazard (passes that reindex ops remap
chain-table PCs via `entries`/`flat_table` and must cover any new
field), then **reverted as dead weight** (zero sparse short chains;
chain count unchanged). Remaining BIfNEL self-follow (58%) is
cross-slot guard sequences, not convertible chains.

**Verification.** 300 lib tests; doom8088 state dump byte-identical
@2M ticks; cycles+IP identical @8M; smoke 7/7. **Host caveat:** the
bench host ran ~35% below the 2026-06-10 baseline all day (310K vs
477K t/s web) — cross-day absolute comparison invalid. Same-day
3-run web A/B medians (ref vs new wasm, same host state): **+4.8%
t/s / −5.8% runMsToInGame / −4.4% doomLoad**; CLI A/B +2–3%; one ref
run overlapped the new band, so quote this as ~+3–5%. JSONs: CSS-DOS
`docs/benches/doom-all-2026-06-11-chainlen{2,-ref}-run{1,2,3}.json`.
Re-baseline STATUS numbers on a healthy host before next perf claim.

---

## 2026-06-10 — copy propagation + DCE landed (`967ddad` + `9ecc6de`): −17.8% dispatched ops, +4.6% web throughput

Executes the copy-elimination proposal from the 2026-06-09 op-profile
entry. New pass in `compile.rs` between `inline_calls` and the fuse
peepholes, run per op stream (main, dispatch entries/fallbacks,
broadcast value/spillover): (1) forward copy propagation — facts merge
by intersection at labels, one linear pass since streams are
forward-only CFGs; (2) backward liveness DCE for pure ops, plus
adjacent store-forwarding (retarget producer → copy dst when the
copied slot dies; kills dispatch-entry result copies). Soundness rests
on the same invariants `compact_slots` already assumes (tick-scratch
slots, declared cross-stream channels); `Dispatch` drops all facts and
its table's transitive read-set is always-live; streams with backward
edges are skipped; `CALCITE_NO_COPY_ELIM=1` disables (native only).

**doom8088 numbers.** Static: 122,859 ops removed, 234,022 reads
redirected. Dispatched ops/tick 846 → 695 (−17.8%; profile CSVs were
tmp-only, regenerate with `--op-profile`). The reduction is entirely
LoadSlot (472M → 341M per 2M ticks): propagation+DCE killed call-site
arg/result copies, forwarding killed entry result copies; survivors
are dispatch-param (barrier) and cross-label copies. CLI 6M ticks:
351.8K vs 307.4K t/s (+14.4%, dev-only). Web 3-run `doom-all --headed`
medians vs the 2026-06-09 baseline (75.0 s / 456.0K t/s / doomLoad
63.65 s): **70.5 s runMsToInGame (−6.1%) / 477.2K t/s (+4.6%) /
doomLoad 60.8 s (−4.4%)** — every run beat every clean baseline run.
JSONs: CSS-DOS `docs/benches/doom-all-2026-06-10-copyelim-run{1,2,3}`.

**Compile-cost note (and a baseline-drift warning).** First cut cost
14.9 s of native compile — fixed to 0.11 s by keeping external/barrier
slots out of the tracked live set, reusing scratch buffers across the
~100K entry streams, and capping the facts map (`9ecc6de`,
output-identical fingerprint). In wasm the pass costs **~1.7 s** of
cabinet compile, measured same-day A/B with the pass hard-disabled
(33.3 vs 31.6 s, `compile-only` profile). Caveat: the 2026-06-09
baseline recorded compileMs ≈ 24.6 s, but pass-OFF today measures
31.6 s — web compile wall drifts day-to-day on this host; only
same-day A/B is trustworthy for compile-cost claims. Runtime metrics
(t/s, doomLoad) were stable across days.

**Verification.** 300 lib tests (9 new, incl. forwarding negatives);
doom8088 full state dump (every state var + cell, 4.5 MB) byte-identical
vs pass-off after 2M ticks; ticks+cycles+IP identical on all 8 cached
harness cabinets; CSS-DOS smoke 7/7. dos-smoke-p2 note: that cabinet
is ~7 t/s in the CLI with or without the pass (pre-existing; it has
tens of millions of ops — 11M reads redirected, 9.5M removed, 4.4 s
native compile) — worth its own investigation someday.

**Remaining headroom in the profile (ops/tick, post-pass):** BIfNEL
22.6% + DispatchChain 4.5% (chain probing), LoadSlot 21.9% (params +
cross-label), LoadState 11.7%, LoadLit 9.0%. Next structural lever is
per-dispatch-key specialisation (CSS-DOS plans), or cross-label copy
propagation if profiling shows the surviving LoadSlots are cheap wins.

---

## 2026-06-10 — rep-generic MERGED to main (`cc729b2`): hardcoded x86 string-op path is gone

Cross-link: CSS-DOS LOGBOOK 2026-06-10. Closes the 2026-06-09 entry's
"remaining step". Two parts:

**Merge-review fixes (branch `b2dc52d`)** — the 2026-06-09 warts, fixed
before merging rather than carried as debt. No literal cabinet-property
name remains anywhere in the generic rep path:

- `LoopDescriptor.per_iter_cycles` is now `Option<CycleCharge>
  { property, per_iter }` — the structurally identified cycle-counter
  member's *name* rides with the per-key literal, and
  `commit_ip_and_cycles` charges through it (and advances IP through
  `descriptor.ip_property`) instead of hardcoded `"IP"`/`"cycleCount"`.
  New test: a `--pc_y`/`--zorch` cabinet commits to its own slots —
  could not pass before.
- The dispatcher routes on each descriptor's `key_property` instead of
  literal `--opcode` (a last-read cache keeps it one slot read per tick
  when descriptors share a family).
- Panic diagnostics and `CALCITE_REP_DIAG` counters are descriptor-
  driven: the x86 opname/REP-prefix decode tables are deleted, fires
  tally per dispatch-key value, and `describe_loop_state` dumps the
  loop's own slots by descriptor-carried names.

**Verification.** Branch: 288 unit tests; calcite-cli A/B vs the
pre-merge main binary **byte-identical** (cycles+IP, 7 smoke cabinets ×
2M ticks); CSS-DOS smoke 7/7 (109 s). Post-merge from `main`: 288 unit
tests; smoke 7/7 (113 s); doom8088 title screen verified via
fast-shoot at tick 6M (mode 13h render correct). No new bench — the
binary behaviour is byte-identical to the branch benched 2026-06-09
(+0.50% wall / −1.49% t/s, inside gate/noise).

**Remaining genericity residue on main (not blockers, tracked):**
`column_drawer_fast_forward` + `COLUMN_DRAWER_BODY` (~280 lines) was
upstream-knowledge code, **default-off** (env `CALCITE_FUSION_FASTFWD`,
disabled 2026-04-29 as a perf net-loss; hard-`false` on wasm) —
**DELETED 2026-06-12** (see entry above). LODS-shape `Full` commit
still refuses loudly (accumulator target not modelled; unreached by
any current cart, proven by the A/B).

## 2026-06-09 — rep-generic: smoke 7/7 + byte-identical A/B + bench — cheat removal verified on the branch

Branch `feat/rep-generic` (`247b274` → `17fe7da`, pushed). Cross-link:
CSS-DOS LOGBOOK 2026-06-09. Continues the 2026-06-08 FINDING below; the
"one recogniser gap" turned out to be the first of **five** layered
defects, each only visible once the previous was fixed:

1. **`ip_extra_advance_slot` captured from the wrong shape** (the known
   gap). Correct capture is the *stay branch's subtrahend* in the
   per-key IP body. Fixed-point argument: on a stay tick the IP slot's
   new value equals its old value, so any wrapper addend W cancels
   (`self − S + W = IP` pins `self`), and the exit branch is always
   `IP + S + L`. The old top-level-`Add` rule was unreachable on real
   cabinets (the Add sits inside the TF/IRQ wrapper's else-`calc`) and
   wrong even where it matched (captures W, which cancels).
2. **Mirror names unresolvable at runtime.** `--__1DI` etc. resolve in
   compiled code via the buffer-copy rule; the applier's `read_prop`
   didn't know it → "pointer self_property unresolved" panic. Now
   routed via `eval::to_bare_name` (pure fn — the first attempt used
   `property_to_address`, whose thread-local ADDRESS_MAP is empty on
   the debugger's tokio workers: CLI passed, debugger hung).
3. **No loop-continuation gate.** The dispatcher applied on *any* tick
   whose opcode had a descriptor — single-shot (non-rep) string ops got
   bulk-applied with stale counters (first observed as cycleCount going
   hugely negative). Now: `evaluate_loop_predicate` — the descriptor's
   own predicate with polarity (`predicate_means_stay`) — evaluated
   against the post-tick slot view, reproducing the CSS's stay/advance
   decision exactly. Generalises the deleted path's hasREP +
   already-exited-via-ZF guards.
4. **Source-base scaling.** `IndirectRead` / `ComparisonShape` now
   record whether the captured base sat inside `* 16` in the shape.
   Kiln's `--_strSrcSeg` is a *flat* base (override-aware, already
   `DS*16`); unconditionally rescaling it read sources 16× off (CMPS
   exited early on garbage → stale-flags Jcc divergence at tick 450375
   of dos-smoke). High-byte `+ 1` literal offsets now peel during
   decomposition, modelled positionally like the dst side.
5. **CMPS/SCAS flags commit silently no-opped** — `flag_property` is
   the mirror name (`--__1flags`); bare-strip missed the state var.
   Same `to_bare_name` routing fixes it. Also: LODS-shape Full commit
   now *refuses* (accumulator target not modelled) instead of silently
   leaving the accumulator stale — loud-not-wrong; no current cart
   reaches it (proven by the A/B below).

**Verification.** (a) `cargo test -p calcite-core`: 287 green, incl.
new shape-true tests (real kiln IP shape through the TF/IRQ wrapper;
subtrahend-not-wrapper pin; predicate polarity; gate behaviour; flat vs
×16 base scaling). The old `ip_extra_advance_slot_*` tests encoded the
wrong model and were rewritten. (b) **A/B vs `main`: byte-identical** —
calcite-cli on all 7 smoke cabinets for 2M ticks gives identical
cycleCount and IP on both builds (tick-count bisection was the
diagnostic; forks at ticks 445075 / 450375 of dos-smoke closed by
fixes 4 / 5). (c) CSS-DOS smoke **7/7 PASS** (111 s — previously the
panicked debugger hung the whole run). (d) Web bench, 3-run `doom-all
--headed` medians (JSONs: CSS-DOS `docs/benches/doom-all-2026-06-09-*`):
rep-generic 73.8 s runMsToInGame / 459.6K t/s / 63.3 s doomLoad vs
fresh main baseline 73.4 s / 466.6K / 62.8 s → **+0.50% wall /
−1.49% t/s / +0.85% doomLoad**. Wall metrics inside the ±1% gate; the
t/s median delta is inside the documented ±3% run noise and the run
distributions overlap. Since the A/B state evolution is byte-identical,
any real residual cost is the dispatcher gate's predicate evaluation on
string-op ticks — optimisable (pre-resolve predicate slot indices at
descriptor build) without touching semantics. A further 3 alternating
bench pairs were attempted but the machine suspended overnight
mid-session; the contaminated runs (main-run5 failed,
repgeneric-run5/6 degraded) are kept alongside, self-identifying.

**Residual warts (on the branch, logged for merge review, not
blockers):**
- `commit_ip_and_cycles` still reads/writes literal `"IP"` /
  `"cycleCount"` names instead of descriptor-carried ones; per-iter
  cycle extraction captures the constant but not the cycle slot's name.
  Both should become descriptor fields.
- The dispatcher reads `--opcode` and panic diagnostics by literal name
  (pre-existing, documented inline).
- LODS-shape Full commit is a deliberate loud hole (see 5).

## 2026-06-09 — FINDING: per-tick op profile — 845 ops/tick, >⅓ pure data movement; copy elimination is the biggest unlisted lever

Read-only analysis on `main` `8de61a8`, no code changes. Cross-link:
CSS-DOS LOGBOOK 2026-06-09 (this entry + the doomLoad tick-share
correction — kernel ~49%, `__I4D` ~22%, not 46%).

Measured with the existing `--op-profile` (doom8088, 2M-tick boot
window, 1.69 G dispatched ops) — first time anyone pulled the
per-tick magnitude from it:

- **845 dispatched ops per tick** (= per guest instruction). The
  "280 K ops" figure is program size; branches skip most of it.
- **LoadSlot 27.9% + LoadLit 8.3% — over ⅓ of runtime is moves.**
  #2 adjacency overall is LoadSlot→LoadSlot (10%): copy *chains*.
  Sources read from the emit code: `flatten_calls` copies args into
  per-function shared param slots and the result back out
  (`compile.rs:3450`/`3461` on `8de61a8`); dispatch entries copy
  `entry_result` → dispatch dst. **Proposed pass:** per-call-site
  copy propagation at inline time + dead-copy elimination over the
  flat stream. Linear-time, shape-only, zero cardinal-rule surface.
- Branch/dispatch machinery ≈30%: BIfNEL 20.9%, DispatchChain 4.1%
  (35/tick), Dispatch 2.8% (24/tick — the 2026-05-14 list's "~50
  probes/tick" guess was accurate), Jump 2.3%.
- Per-op cost ≈2.6 ns in WASM at ~448 K t/s — the interpreter is
  already lean per op. The lever is op *count* (845 → ~50
  semantically needed): copy elimination first, per-dispatch-key
  specialisation as the structural follow-up, load-time WASM codegen
  as the long-arc multiplier.

Corrections to the record: (1) the 2026-05-14 list's item 5
("bounds-checked Vec<i64> slot access", rated easiest win) was stale
when written — unchecked slot macros landed 2026-04-14 (`490a8bf`)
and slots are `i32`; that explains the "item 5 → −30%, never retry"
DEAD entry (it re-attempted a done thing). (2) `COLUMN_DRAWER_BODY`
(`compile.rs` ~5787) hardcodes 21 bytes of Doom's x86 column drawer
in the engine — default-off and dead on wasm32, but it is upstream
knowledge; delete at genericity-ship time alongside
`rep_fast_forward`.

## 2026-06-08 — FINDING: rep-generic dispatcher landed, but recogniser's ip_extra_advance_slot shape is wrong for real cabinets

Branch `feat/rep-generic` (commit `247b274`, recovered from a wedged
2026-05-29 session that hung awaiting a never-delivered task
notification). Cross-link: CSS-DOS LOGBOOK / STATUS 2026-06-08.

**What landed (recovered + committed + pushed):** the Task 3.5
descriptor-driven dispatcher rewrite in `compile.rs` `rep_fast_forward`
(+120 / −522): deletes the hardcoded x86 opcode-table path, routes
purely through a `LoopDescriptor` lookup keyed by opcode value →
`BulkClass` → `apply_{fill,copy,read_only}_with_commit`. No
`matches!(opcode, 0xAA | ...)`, no env toggle, "this is the only path";
unsupported/`PerIter` shapes panic loudly (no silent slow path). Also
deleted the now-obsolete `tests/rep_fast_forward.rs` integration test
(it asserted the deleted opcode-table behaviour). Full lib unit suite
green: 281 + 28 + 7 + 5 + 10, 0 failed. calcite-core / calcite-cli /
calcite-debugger all build.

**The gap (verified against real cabinet CSS, not argued):** smoke
6/7 carts panic —
`rep_fast_forward: applier-unsupported — REPE/REP STOSW (op=0xab) ...
ip_extra_advance_slot not captured`. `hello-text` passes (little REP
work). Root cause is a **recogniser shape mismatch**, not the
dispatcher:

- `extract_ip_extra_advance_slot` (`pattern/loop_descriptor.rs:1266`)
  models IP-advance as a top-level `Calc(Add(dispatch, bareVar))` and
  captures the `Var` as `ip_extra_advance_slot`. The
  `ip_extra_advance_slot_*` unit tests all hand-build exactly that
  shape (`add(ip_dispatch, var("--prefixLen"))`) → they pass while
  every real cabinet panics. The tests encode the recogniser's *wrong
  model*, so green unit tests gave false confidence.
- The real cabinet (cga4-stripes, `--IP` assignment) is a different
  shape entirely: `--IP: if(--_tf; --_irqActive; else: calc(<dispatch
  on --opcode>))`, and the **per-opcode body** for STOSW (171) is
  `if(--_repContinue:1 : calc(__1IP − prefixLen); else: calc(__1IP +
  1))`. So `prefixLen` appears as a **subtraction inside a
  `_repContinue`-gated per-key body** (IP backs up by prefixLen to
  re-run the REP each iteration; +1 after the loop ends) — NOT as a
  top-level `Add(dispatch, Var)`.

**Why it's not a one-liner.** Grabbing the slot name is trivial;
modelling it *correctly* is not. The applier must reproduce Chrome's
post-REP IP, which here is the `else` branch (`__1IP + 1`) reached when
`_repContinue` goes false. The descriptor + applier have to model that
two-branch IP semantics, not just capture a name — getting it wrong
silently corrupts IP after every REP. This is correctness-critical
recogniser design the dead session never reached; my first hypothesis
(just descend through the TF/IRQ `StyleCondition` wrappers, reusing
`collect_override_branches`) was **disproven** by the real CSS — the
wrapper descent isn't the issue, the per-key gated-subtraction shape
is.

**Next step (unfinished):** teach the recogniser the
`_repContinue ? IP − prefixLen : IP + 1` per-key IP body shape, capture
`prefixLen` from the subtraction, and make the applier commit the
post-REP `+1` branch. Then re-run smoke 7/7 and the ±1% perf bench
(neither has passed yet — do NOT call this branch "landed"). Open
question still unaddressed: whether all 6 failing carts share this one
shape or there are further independent recogniser gaps behind it.

## 2026-05-28 — input-edge apply moved off per-tick path (Phase 4)

Branch `feat/keyboard-pseudo-input`. Cross-link: CSS-DOS LOGBOOK
2026-05-28.

The per-tick `needs_input_edge_apply` gate + `apply_input_edges` body
ran on every tick (~34 M times per doom-loading run). After the 2026-05-26
fix the gate was cheap (a gen-counter compare, then a bool fast-path),
but the bandwidth still cost something on the web bench: 9-run avg
~377 K t/s vs the 2026-05-08 master baseline of ~446 K t/s.

Two commits in this session:

1. **`dcc7dd5` — Phase 1: collapse gate to single bool field load.**
   Replaced `pseudo_active_gen: u32` on State + `last_apply_gen: u32`
   on Evaluator with `pseudo_active_dirty: bool` on State. The gate
   read becomes one byte load + branch instead of two u32 loads +
   compare. Native A/B (6 runs) showed wash within noise — confirmed
   the gate cost was not the binding constraint. Kept anyway: simpler
   shape, sets up Phase 4.

2. **`f4da585` — Phase 4: apply-on-transition.** Moved the entire
   apply path off the per-tick loop. `State::set_pseudo_class_active`
   now directly recomputes the affected gated state-var slots at the
   moment of mutation. Per-tick cost is zero — no gate, no apply.

   Mechanics:
   - `State::input_edge_groups: Vec<InputEdgeGroup>` installed by
     `Evaluator::wire_state_for_input_edges(&mut state)` at engine
     construction (and on `reset()`). Same wiring pattern as
     `wire_state_for_packed_memory` / `windowed_byte_array`.
   - `set_pseudo_class_active` toggles HashSet membership, then for
     each group sums values of edges whose `(pseudo, selector)` is
     currently active and writes the slot. Inverted iteration over
     the small active set avoids HashSet allocations.

   Deletions: `needs_input_edge_apply` gate, `apply_input_edges`
   body, `build_input_edge_groups` helper, `InputEdgeGroup` /
   `InputEdgeGroupEntry` structs (moved to state.rs, pub),
   `input_edge_groups` / `last_apply_was_nonzero` / `pseudo_active_dirty`
   fields, 4 per-tick call sites. Net −73 lines.

Web doom-loading bench (4 runs after Phase 4):
- 394 K / 87.0 s
- 431 K / 79.4 s
- 436 K / 79.5 s
- 432 K / 79.9 s

Median ~432 K / ~79.5 s, vs 2026-05-08 master baseline 446 K / 77.1 s.
**Within ~3 % of master** on doomLoad-shape work. Smoke 7/7 PASS.
Unit + integration input-edge tests pass.

Cabinet-genericity preserved: the apply path operates over CSS pseudo
classes, selectors, and state-var slots — no knowledge of what those
slots represent above the CSS layer.

---

## 2026-05-26 — apply_input_edges hot-path: inversion + gen cache (doomLoad fix)

Branch `feat/keyboard-pseudo-input`. Cross-link: CSS-DOS LOGBOOK
2026-05-26.

The 2026-05-22 inline-gate fix (`763d6cd`) recovered the boot path but
did NOT fix the doomLoad-phase regression. Root cause for the residual:
during doomLoad the `doom-all` bench's `title_tap` watch fires every
poll (cond `menuactive=0,gamestate=3,bdamode=0x13,repeat`), so
`pseudo_active` is non-empty most of the time and the body of
`apply_input_edges` runs every tick. Each tick walked all 59 input
edges and called `state.pseudo_class_active_pair(&pseudo, &selector)`,
which did `self.pseudo_active.contains(&(p.to_string(), s.to_string()))`
— two `String::from(&str)` allocations per lookup, 118 allocations per
tick, ~30 M/sec at 250 K t/s. **That** was the doomLoad bottleneck.

Two changes in this commit:

1. **Inverted iteration.** The slow path now walks `pseudo_active`
   (small — 0-2 entries during gameplay) and looks up matching edges
   by reference-compare, rather than walking every edge and probing
   the HashSet. Zero allocations on the hot path.

2. **Generation cache.** Added `State::pseudo_active_gen` bumped on
   every mutation, and `Evaluator::last_apply_gen` recording the gen
   at the last apply. `needs_input_edge_apply` now short-circuits on
   `gen == last_apply_gen` — so during the long quiet stretches
   between pulse-release boundaries (~50 K ticks at a time), we skip
   the recompute entirely.

Bench numbers on doom-all web (5 runs each, post-fix vs the fresh-wasm
pre-fix baseline established the same day):

| Metric          | pre-fix avg | post-fix avg | master 2026-05-08 |
|-----------------|------------:|-------------:|------------------:|
| runMsToInGame   | 174.6 s     | ~92 s        | 77.1 s            |
| doomLoad        | 155.1 s     | ~78 s        | 70.0 s            |
| ticksPerSecAvg  | 193 K       | ~377 K       | ~446 K            |
| ingameFps       | 0.50        | 0.9-1.7      | ~1.9              |

Most of the 1.78× regression closed. Residual ~12 % gap vs master is
within the range plausibly explained by struct-layout cache effects
from the additional `pseudo_active` HashSet field — not worth chasing
unless it widens.

Files: `crates/calcite-core/src/{eval.rs,state.rs}`. Unit tests:
`input_edges_drive_state_var` still passes (it tests the apply path
end-to-end including the new inversion + gen-cache).

## 2026-05-19 — script: WatchKind::Stride regression fix (wasm poll cadence)

Branch `feat/keyboard-pseudo-input`. Cross-link: CSS-DOS
`docs/logbook/LOGBOOK.md` 2026-05-19 (same investigation, the
companion bridge fix lives there).

`baf3086` ("keyboard: :active pseudo-class input model, split from
genericity bundle") silently reverted `WatchKind::Stride` from the
elapsed-since-last-fire form (introduced by `e442f74`) back to
`tick % every == 0`. The CLI watch runner is immune — it advances
its cursor in `min_stride`-sized steps, so the tick passed to
`poll()` is always a multiple of `every`. The wasm path
(`run_batch_watched`) polls at `frame_counter + cumulative chunk`
boundaries that are adaptively sized and essentially never a
multiple of 50_000, so `tick % 50_000 == 0` was almost never true:
the `poll` stride watch never fired, every `gate=poll` cond watch
(title/menu/loading/ingame in the CSS-DOS doom-loading profile)
starved, and the web bench appeared stuck even though the engine
was executing fine (1B+ cycles, mode 0x13).

Fix: restored the elapsed-since-last-fire `Stride { every,
last_fired_at: Cell<u32> }` form across `script.rs` (enum + doc),
`script_eval.rs` (evaluate logic), `script_spec.rs` (parse
constructor + test pattern), `tests/script_primitives.rs` (5
constructors), and the `calcite-cli` `min_stride` match arm
(`{ every, .. }`). Added `script_eval::tests::
stride_fires_on_unaligned_poll_cadence` — polls at a 137-tick
cadence (never a multiple of `every`) and asserts the watch still
fires, reproducing the exact wasm failure mode (would fail on the
`% ` form, passes now). 17 script tests green.

This was not a keyboard bug. The keyboard input model
(`:has(#kb-X:active)` → input-edge recogniser → pre-tick
`apply_input_edges`) is sound: a clean CLI `doom-loading` run
reaches in-game at tick 34,650,000 — exact parity with the
2026-05-02 `setvar_pulse` baseline. The prior "keyboard stuck at
title" finding was wrong; the symptom was this stride regression
(web) plus a CSS-DOS-side bench-run watch-wipe (logged CSS-DOS
side). Post-fix the web `doom-loading` bench reaches in-game at
tick 34,294,512 (`ok:true`, all six stages fire).

## 2026-05-18 — genericity↔perf cost isolated (cross-cutting)

Cross-link: CSS-DOS
[`docs/plans/2026-05-18-genericity-perf-cost-isolation.md`](../../CSS-DOS/docs/plans/2026-05-18-genericity-perf-cost-isolation.md)
and LOGBOOK 2026-05-18 FINDING of the same name.

Decomposed `feat/calcite-genericity` (`a89067a`/`3592bf0`, 30 files
over `ef44f20`) into a verified per-change perf table **and benched
it end-to-end** (prior log numbers treated as untrusted). One
`doom-all --headed` run on `3592bf0` vs the on-disk `ef44f20`/
BIF2-off baseline: **75.9 s / 448.5K t/s / doomLoad 64.8 s** vs
baseline **77–82 s / 416–443K / 65–70 s** — at or below the fastest
baseline run on every metric. **No regression — measured.** Static
analysis agrees: all new pattern modules
(`loop_descriptor`/`dispatch_specialise`/`identity_prune`) are called
only from `Evaluator::from_parsed` (compile-time) or behind
default-off `OnceLock` gates; `git grep` confirms zero call sites in
`execute`/`exec_ops`. `column_drawer_fast_forward` deletion removes
dead-by-default code. The `apply_input_edges` drop
(`a5e8eee`→`6d9e80a`) is **`feat/keyboard-pseudo-input`**, not this
branch. The only genericity change with unknown perf cost is the
`rep_fast_forward` generic applier — unknown because it was never
built.

---

## 2026-05-02 — script: setvar_pulse + cond:repeat sustain mode

Closes the first follow-up from the 2026-05-01 chunk D entry: the
script primitives didn't drive cabinet keyboard handlers (which need
make/break edge pairs) when used as a sustain-cond + setvar_pulse
spam loop. Three coupled fixes:

1. **New action `Action::SetVarPulse { name, value, hold_ticks }`.**
   Writes value now, schedules a write-of-0 release after `hold_ticks`
   ticks. The release is dispatched at the top of the next `poll()`
   that crosses the release tick. Pulses queued while a release is
   still pending — or that fires THIS poll — are skipped, so the
   engine sees a clean make/break/make/break alternation at twice
   the gating poll stride. CLI flag:
   `setvar_pulse=NAME,VALUE,HOLD_TICKS`.

2. **`cond:repeat` is sustain mode now.** It fires on every gated
   poll while the predicate holds, matching the doc that's been
   there since chunk D. The original rising-edge-only implementation
   was overfitted to a use case nobody's exercising and broke the
   spam pattern the new primitives are meant to replace.

3. **Registry tracks `pending_releases` + `released_this_poll`.** The
   action dispatch consults both to decide whether to skip a pulse.
   No change to the hot eval path for non-pulse actions.

Tests: 9 → 10 integration. New `sustain_cond_pulse_alternates_make_
and_break_at_2x_poll_stride` proves the cadence. The chunk D-era
`setvar_pulse_re_arm_extends_release` test was renamed to
`setvar_pulse_skips_when_release_pending` — the re-arm semantic was
theoretical; skip-while-pending is what cabinets need.

End-to-end: doom8088 CLI bench (CSS-DOS-side) reaches in_game in
145.8 s / 34.65 M ticks (Chunk A baseline 119 s / 35 M ticks). +22 %
wall covers the watch-poll overhead; engine work essentially
identical. See CSS-DOS LOGBOOK 2026-05-02.

Cardinal-rule check: still zero upstream knowledge in calcite-core.
The fact that "keyboard" is the var being pulsed lives in the bench
profile (CSS-DOS-side). The action knows only "set var X to value
Y, write 0 to X after N ticks."

## 2026-05-01 — Repo cleanup: Chunk D — script-primitive layer landed

Per `../CSS-DOS/docs/audit-summary-and-plan.md` Chunk D. Three legacy
DSLs (calcite-cli's `--cond`, `--poll-stride`, `--script-event`) are
collapsed into one generic measurement-primitive substrate in
calcite-core. Same syntax surface on calcite-cli (`--watch`) and
calcite-wasm (`engine.register_watch`).

**What's new in calcite-core**

- `script::WatchRegistry` + `script_eval::poll`. Hosts register
  watches; per-tick (or per chunk) call `poll` with the current state
  and tick. Events accumulate; host drains.
- Primitives: `Stride { every }`, `Burst { every, count }`,
  `At { tick }`, `Edge { addr }`, `Cond { tests, repeat }`,
  `Halt { addr }`. Cheap (Stride/Burst/At/Halt) vs expensive
  (Edge/Cond) split is structural — expensive watches name a `gate`
  watch and only evaluate on ticks where the gate fired (two-phase
  poll).
- Predicates: `ByteEq`, `ByteNe`, `BytePatternAt { base, stride,
  max_window, needle }`. The third replaces the old
  `vram_text:NEEDLE` helper — the upstream-knowledge constants
  (text VRAM at 0xB8000, stride 2) live in CSS-DOS-side profiles
  now, NOT in calcite. Generic across any cabinet whose layout puts
  bytes at a known stride.
- Actions: `Emit`, `DumpMemRange { addr, len, path_template }`,
  `Snapshot { path_template }`, `SetVar { name, value }`, `Halt`.
  Path templates support `{tick}` and `{name}` substitution.
- `DumpSink` chooses `File` (native CLIs; bytes written to disk) vs
  `Memory` (wasm; bytes ride out on `MeasurementEvent.dumps`).
- Text-format parser in `script_spec::parse_watch` shared by both
  calcite-cli and calcite-wasm.

**calcite-cli changes**

- New flags: `--watch NAME:KIND:SPEC[:gate=NAME][:sample=VAR1,VAR2][:then=ACTIONS]`
  (repeatable) and `--measure-out=PATH` (JSON Lines stream).
- Removed: `--cond`, `--poll-stride`, `--script-event`,
  `--script-file`. **No back-compat alias**, per the audit plan.
  ~280 lines of legacy parser + scheduler glue deleted.
- Migration map for the only remaining consumer
  (`tests/harness/bench-doom-stages-cli.mjs` in CSS-DOS, which gets
  rewired in Chunk E):
  - `--poll-stride=N` → `--watch poll:stride:every=N`
  - `--cond=ingame:ADDR=VAL:then=halt` →
    `--watch ingame:cond:ADDR=VAL:gate=poll:then=halt`
  - `--script-event=TICK:tap:VALUE` →
    `--watch <name>:at:tick=TICK:then=setvar=keyboard,VALUE`
    (paired release event registered separately if needed; the
    cabinet's edge detector handles the make/break itself)

**calcite-wasm surface**

- `register_watch(spec)`, `watch_count()`, `clear_watches()`,
  `run_batch_watched(count, chunk_ticks)`, `drain_measurements()`,
  `watch_halt_requested()`. `reset()` also clears the registry.
- `drain_measurements()` returns JSON. Dump bytes are base64-encoded
  so the payload survives the JS string boundary; bench JS decodes
  if it cares. Tiny built-in base64 encoder, no new dep.

**Tests**

- 7 unit tests in `script_eval` (registry smoke, stride/burst/At
  cadence, edge priming, byte-pattern matching, template
  substitution).
- 7 integration tests in `tests/script_primitives.rs` exercising
  every primitive end-to-end through real Evaluator + State on
  trivial 1-property non-x86 cabinets.
- 6 parser tests in `script_spec` (stride, burst, cond+pattern,
  dump-with-template-path, setvar, sample-vars).

**Verification**

- `cargo test --workspace`: 161 passed, 4 pre-existing failures
  (`compile_full_program`, `compile_dispatch_table`,
  `compile_value_forwarding`, `eval::tick_applies_assignments` —
  all panic in `rep_fast_forward` with "no-opcode" on cabinets that
  don't expose `--opcode`; documented in CSS-DOS LOGBOOK 2026-05-01,
  out of scope for Chunk D). All new tests pass.
- `wasm-pack build crates/calcite-wasm --target web --release`
  succeeds; ~33s including wasm-opt.

**What this enables (Chunk E preview)**

The same watch spec drives doom-stages bench (web AND native) without
the harness having to know about poll-stride / script-event /
condition syntax. CSS-DOS-side bench profiles compose the generic
primitives with their own constants for upstream layer (e.g.
`pattern@0xb8000:2:4000=DR-DOS` rather than `vram_text:DR-DOS` —
calcite never sees the meaning of 0xB8000). The cardinal rule holds:
calcite knows only structural CSS-shape predicates over guest memory;
upstream knowledge lives in the host that registers them.

Files touched: `crates/calcite-core/src/{lib,script,script_eval,
script_spec}.rs`, `crates/calcite-core/tests/script_primitives.rs`,
`crates/calcite-cli/src/main.rs`,
`crates/calcite-wasm/src/lib.rs`. Branch
`cleanup-2026-05-01`, not pushed.

---

## 2026-05-01 — LoadPackedByte: euclid → bitwise byte extract

3 exec arms (production, profiled, traced) of `Op::LoadPackedByte`
replaced `cell.rem_euclid(256) / cell.div_euclid(256).rem_euclid(256)`
with the documented form `(cell >> (off * 8)) & 0xFF`. Equivalent for
all i32 cells (verified against negative two's-complement); skips a
branch in libcore's signed-euclid path. Doc on Op::LoadPackedByte
already specified this form; the executor was the outlier.

Web bench (vs FxHashMap baseline, same cabinet, headed):

|                          | fxhash only | + bitwise | Δ      |
|--------------------------|------------:|----------:|-------:|
| loading→ingame           |  66,524 ms  | 62,740 ms |  −5.7% |
| runMsToInGame            |  78,506 ms  | 74,307 ms |  −5.4% |
| gameplay ticks/sec       |     426,299 |   430,106 |  +0.9% |
| gameplay simulatedFps    |        43.5 |      43.8 |  +0.7% |
| gameplay vramFps         |        2.39 |      2.38 |   noise|
| gameplay cycles/sec      |   5,930,982 | 5,968,883 |  +0.6% |

Asymmetric — LoadPackedByte fires hard during level-load (window-byte
reads from disk into RAM), but is only ~3% of dispatched ops in
steady-state gameplay (per 2026-04-29 op-profile). So we shave the
load curve and barely move gameplay. Real but small.

Cardinal rule check: just bitwise byte extraction following the op's
existing documented semantics. Generic across any pack value.

## 2026-04-30 — FxHashMap swap: +25% ingame fps, −24% web level-load

Followup to today's flamegraph. Replaced `std::HashMap` (SipHash) with
`rustc_hash::FxHashMap` in the runtime hot-path tables only:

- `state.extended` — `HashMap<i32, i32>` for the >0xF0000 fallback in
  `read_mem` (9% in flamegraph).
- `DispatchChainTable.entries` — `HashMap<i32, u32>` for `Op::DispatchChain`.
- `CompiledDispatchTable.entries` — `HashMap<i64, (Vec<Op>, Slot)>` for
  `Op::Dispatch` (the 4% `hash_one` frame's call site).
- `CompiledBroadcastWrite.address_map` — `HashMap<i64, i32>` for
  per-tick broadcast write fan-out.
- `CompiledSpillover.entries` — `HashMap<i64, (Vec<Op>, Slot)>`.

Compile-time string-keyed maps in `CompilerCtx` and `dispatch_tables`
left as std::HashMap — touched once at load, swap would ripple across
crates for no benefit.

**Same cabinet, same cycles, same ticks; pure per-tick eval speedup.**

| Bench                              | baseline   | fxhash     | Δ        |
|------------------------------------|-----------:|-----------:|---------:|
| CLI loading→ingame (29.5M ticks)   |  72,000 ms |  61,562 ms |  −14.5%  |
| Web loading→ingame (29.8M ticks)   |  88,200 ms |  66,524 ms |  −24.5%  |
| Web runMsToInGame                  | ~125,000 ms|  78,506 ms |  −37%    |
| Gameplay ticks/sec (60s LEFT)      |    333,000 |    426,299 |   +28%   |
| Gameplay simulatedFps              |       34.7 |       43.5 |   +25%   |
| Gameplay vramFps                   |        1.6 |        2.4 |   +50%   |
| Gameplay cycles/sec                |  4,970,000 |  5,930,982 |   +19%   |

ticksToInGame: 35,000,000 → 34,650,589 (−1%, stage-detect race; not
real). cyclesToInGame: 397M (CLI) — identical, confirms zero work
elision.

**Why this works.** The flamegraph showed SipHash + hash_one + extended
HashMap calls totalling 17% of worker CPU. SipHash is constant-time
hardened and runs ~30 cycles per i32 key; FxHash is ~3 cycles
(multiply + xor). On runtime maps that get hit per-tick (`Op::Dispatch`
inner loop, `read_mem` BIOS-region fallback, broadcast-write
address_map fan-out), the difference is the dominant cost in those
samples.

Why bigger win on web than CLI: native Chrome's V8 wasm interpreter
makes hash function calls relatively more expensive vs the loop body
than native code's loop-unrolled hot path. Same delta in absolute
seconds, larger as a percentage of the slower web baseline.

Smoke gate: 4 PASS / 3 pre-existing FAIL — same set as before
(dos-smoke, zork1, montezuma fail with `ticks=0` due to harness
build-budget issue documented 2026-04-29, unrelated to this change).
calcite-core unit tests: 148 PASS / 4 pre-existing FAIL (rep_fast_forward
no-opcode panics).

Calcite commit: includes `rustc-hash = "2"` dep + 6 type swaps in
`compile.rs` + 1 in `state.rs`. ~12-line diff, 0 algorithmic change.

The cliff is breakable. Top of next flamegraph will be a different
shape — re-profile if/when chasing the next 10%.

## 2026-04-30 — Web flamegraph: exec_ops dominates, hashing is 17%

New tool: `tests/harness/flamegraph-doom.mjs` drives Playwright + raw CDP
to capture V8 cpuprofile (worker + main thread) and chrome trace JSON for
two phases: LOAD (snapshot-restore from `stage_loading`, run to GS_LEVEL)
and INGAME (snapshot-restore from `stage_ingame`, hold LEFT 60s).
`resolve-cpuprofile.mjs` parses the wasm `name` section and rewrites the
profile in place so DevTools shows real Rust names.

To get names, calcite must be built without wasm-opt's name-section
strip: `wasm-pack build crates/calcite-wasm --target web --profiling
--no-opt`. Profiling build is ~5% slower but the % breakdown matches
release.

**Worker self-time (LOAD 173s / INGAME 60s, near-identical shapes):**

```
                                             LOAD   INGAME
calcite_core::compile::exec_ops             76.07%  75.40%
calcite_core::state::State::read_mem         9.10%   9.35%
core::hash::sip::Hasher::write               4.07%   4.11%
core::hash::BuildHasher::hash_one            3.85%   3.88%
calcite_core::compile::execute               2.94%   3.11%
(idle)                                       1.62%   1.69%
calcite_core::compile::rep_fast_forward      0.49%   0.57%
... everything else < 0.5% individually
```

**Main thread:** 99% idle. Bridge is not the problem; render is not the
problem; it's all wasm.

**Headlines:**
- LOAD: 173s wall, 33M ticks, 200K t/s steady state
- INGAME: 60s wall, 10M ticks, 171K t/s
  (matches LOGBOOK 2026-04-29 gameplay bench at 333K ÷ ~2 for the slower
  profiling build — % shape unchanged.)

**Reading.** `exec_ops` is the per-op dispatch loop. It's 76% — that's
the headline. But the other 17% (`read_mem` + SipHash + hash_one) is
almost entirely **HashMap lookup overhead**: `read_mem` hits a
`HashMap<linear_addr, byte>` for sparse/MMIO writes, and `hash_one` is
called from `Dispatch` Op evaluation (the per-register dispatch table is
a HashMap). SipHash is the default `std::collections::HashMap` hasher.

That's the answer the user asked for. Two real leads, both generic
(no upstream-layer knowledge needed):

1. **Replace HashMap with FxHash or AHash** in the hot lookups.
   SipHash is ~5x slower than FxHash and 17% of total CPU is
   hash-related. Even a 3x speedup here = ~10% wall.
2. **`read_mem`'s HashMap is sparse-overlay over the dense ROM/RAM**.
   At 9% of CPU there's likely a path that takes the slow `HashMap.get`
   even when the address is in the dense regions. Worth a flame-graph
   zoom on what's calling it.

The 76% in `exec_ops` is the main interpreter. To break that down further
would need finer sub-function profiling (LLVM-level), or a
function-pointer-table dispatch backend (calcite Phase 3 closure backend
prototype, 2026-04-28 logbook entry, was 1.19× slower with only 10/50
ops specialised — the work isn't proven dead, just paused).

Artifacts:
- `tmp/flamegraph/load/{worker,main}.cpuprofile` — load in DevTools
  → Performance for the actual flame chart.
- `tmp/flamegraph/load/trace.json` — perfetto/about:tracing.
- Same under `tmp/flamegraph/ingame/`.
- `tmp/flamegraph/{load,ingame}/summary.json` — top-N tables.

Stop chasing peepholes. Real next swing: kill the SipHash overhead.

## 2026-04-30 — read_mem borrow-overhead fix: dead lead, reverted

Gated `read_mem`'s three `RefCell::try_borrow_mut` probes behind a
`Cell<bool>`. Theoretical save ≤0.25%; web doom8088 level-load 133–134s
both runs, no signal. Reverted.

## 2026-04-30 — BIfNEL2 fusion: dead lead, off by default

`Op::BranchIfNotEqLit2` collapses adjacent diff-slot AND-guard BIfNEL
pairs (1330/1395 in doom8088). Fired 794×, dropped runtime BIfNEL→BIfNEL
adjacency 12.35% → 9.41%. Web bench in noise floor (ON avg 2.5% slower
across 2 runs, sign flips). Saved dispatch absorbed by `pc += 2;
continue;`. Disabled behind `CALCITE_BIF2_FUSE=1`. Calcite `ac0e7bb`.

Stop chasing 1-3% peepholes. 405K → 200K throughput is a 2× cliff;
flame-graph the hot path before next attempt.

## 2026-04-29 — runtime op-adjacency profile (post-fusion truth)

Built `--op-profile=PATH` in calcite-cli (calcite `pattern/op_profile.rs`).
Records (prev_kind, curr_kind) counts for every op dispatched, including
inside dispatch entries / function bodies / broadcast-write value_ops.
Thread-local matrix, ~10ns/op when enabled, ~1ns when disabled. Doom8088
restored from `stage_loading.snap`, 200K-tick window, 169M ops dispatched.

**Top kinds (% of dispatched, runtime not static):**
```
LoadSlot              27.34%
BranchIfNotEqLit      20.08%
LoadState             11.05%
LoadLit                8.38%
DispatchChain          4.24%   ← chains *are* hot, despite collapsing 208 of them
Add                    4.03%
LoadPackedByte         3.26%
MulLit                 2.97%
Dispatch               2.81%
AddLit                 2.69%
```

LoadSlot+BIfNEL+LoadState+LoadLit = **66%** of all dispatched ops. The
earlier static-bytecode 27%/25% numbers were directionally right.

**Top adjacencies:**
```
BIfNEL  -> BIfNEL                12.35%   ← biggest spike
LoadSlot -> LoadSlot              9.63%
LoadSlot -> BIfNEL                5.23%
LoadState -> LoadSlot             3.31%
LoadSlot -> LoadPackedByte        3.26%   ← packed-byte load setup
LoadPackedByte -> LoadSlot        3.26%
LoadSlot -> DispatchChain         3.26%
LoadLit  -> LoadSlot              3.22%
LoadState -> LoadState            2.72%   ← back-to-back state reads
LoadSlot -> Jump                  2.14%
```

**Verdict.** `BIfNEL → BIfNEL` at 12% is the only striking spike. This
shape is what `dispatch_chains` is built to collapse, yet it survives —
either chains below threshold (< 3), testing different slots, or
adjacent across control-flow rather than in the static op array. Worth
investigating: **why does the dispatch_chains pass leave so many bare
BIfNEL→BIfNEL pairs adjacent at runtime?** That's the real next lead.

`LoadSlot → BIfNEL` at 5% looked like the previous fuser's target, but
that fuser fired 0× because it required same-slot
(`LoadSlot(dst) → BIfNEL(a=dst)`); the 5% here is overwhelmingly
different-slot — dst is being loaded for a *later* instruction, not the
adjacent branch. Confirmed dead lead.

`LoadState → LoadState` at 2.7% (back-to-back state reads) is a
candidate fuse-target — but only 2.7%, and CSS-shape detection needs
to be careful (the two reads may target unrelated addresses).

## 2026-04-29 — REP FFD: leave alone

`CALCITE_REP_DIAG=1` boot-to-ingame: 213K fires / 1.64M iters elided, no
missing-variant bails (no REPNE, no segment-override). The "REPNE/REPE
SCASB+CMPSB missing" open item is stale — removed.

## 2026-04-29 — calcite: DiskWindow → WindowedByteArray rename

Cardinal-rule fix (calcite `cff0902`). Recogniser was named after upstream
concept (rom-disk) instead of CSS shape (windowed byte array indexed by
key cell + stride). Pure rename, no behavior change.

## 2026-04-29 — load+compare+branch widening: dead lead, reverted

Built `fuse_loadslot_branch` mirroring the state-source fuser. **Fired 0
times** on doom8088. `LoadSlot(dst, src) → BranchIfNotEqLit(a=dst)` doesn't
exist as adjacent ops post-`fuse_cmp_branch` (77K fires) and
`dispatch_chains` (208 chains collapsed). Op-profile's "27% LoadSlot + 25%
BranchIfNotEqLit" is misleading — those exist program-wide but aren't
adjacent (i, i+1) by the time peepholes run. Reverted. Real widening lead
is residual unfused chains *upstream* of those passes, not more downstream
peepholes.

## 2026-04-29 — fusion FFD: funnel data + verdict (dead end on this window)

thread_local `FusionDiag` (no atomics on hot path). Boot-to-ingame
doom8088 native, 35M ticks:

```
              fusion off    fusion on
total wall    136.74 s      140.85 s    (+3.0% slower)
ticks/sec     255,968       248,495
cycles        397,458,534   397,603,025 (+144,491)
```

Funnel (fusion ON):
```
pass_b0  (0x88 at IP)       48,715   0.139 %
pass_b1  (0xF0 at IP+1)      5,298   0.0151%   ← 89% filtered
pass_flags                   5,153   0.0147%
pass_rom (full 21-byte)        159   0.0005%   ← fires
body_iters_applied           1,708             ← avg 10.7/fire
```

Verdict: detector fires 159× / 35M ticks. Max theoretical save = 1708 /
35M = **0.0049%**. Earlier "1.4% wall" was noise; this run shows opposite
sign. Cycle delta ~144K matches 1708 fires × ~50 cycles + noise → work
*is* elided correctly, just not enough for wall-time.

Why detection cost matters at low fire rate: fast-out runs every tick.
~35M `read_mem` (byte 0) + 50K (byte 1) per run; `read_mem` does
`RefCell::try_borrow_mut` on `read_log` (~5-10ns each) = ~350ms overhead /
137s = the observed 3% slowdown.

This bench is boot+level-load, not gameplay. Earlier
bench-doom-gameplay also showed -3%, but with hash-gated paint and only
1.6 visible-fps the column drawer's CPU share may be smaller than static
analysis (21×16 reps × 30 occurrences) implied.

**Direction.** Stop polishing polling shape. Move detection compile-time:
byte_period finds 30 ROM occurrences of the 21-byte body — mark linear
addresses at compile time, insert a single guarded op keyed on
`--ip == known_site`, collapsing detection to one slot-compare/tick. If
that doesn't pay either, fusion belongs in another cabinet's perf budget.

## 2026-04-29 — fusion FFD: framing + diag redesign

Two upstream issues blocking the investigation:

**1. Runtime feature gates are env-vars + cfg stubs, not real config.**
`CALCITE_FUSION_FASTFWD`, `CALCITE_REP_FASTFWD`, `CALCITE_FUSION_DIAG`
read via `std::env::var`, latched in `OnceLock<bool>`, with
`#[cfg(target_arch = "wasm32")]` stubs hardcoding per-target default.
Web isn't toggleable from JS; latch prevents per-cabinet/per-test
control. Right shape: `RunOptions` struct on `CompiledProgram` (or
threaded into `execute`), populated by both calcite-cli and calcite-wasm
callers. Not refactored — flagged. (Web fusion was tested by hardcoding
the wasm stub to true; net-loss finding holds across both targets.)

**2. Diag counters distorted measurements.** First-cut used
`AtomicUsize::fetch_add` per tick at each funnel stage — at 10M ticks/s
that's 50-100ms/s pure `lock xadd` overhead, showing up *as* the fusion
overhead it was measuring. Replaced with `&mut FusionDiag` threaded
through `execute → column_drawer_fast_forward`. Same problem in
`rep_fast_forward`'s diag (atomics) — fix later.

**Three places fusion detection could live:**
- (a) End-of-tick polling (current). O(ticks). 10ms/s detection floor
  even with perfect 1-byte fast-out. Pays on every cabinet.
- (b) Hot-IP gating. O(ticks-while-CS-in-hot-segment). Generic version
  is "compile-time detect which CS values execute most ROM bytes, gate
  fusion on those," not "0x55 is hot."
- (c) Compile-time ROM scan + op-stream rewrite — the real fix.
  `byte_period` already finds matches (4065 regions on doom8088). Either
  rewrite the dispatch entry for that IP to invoke `Op::FusedColumnDrawer`,
  or insert `Op::FusedSiteHook` guarded by
  `LoadStateAndBranchIfEqLit(IP, KNOWN_FUSION_IP)`. Cardinal-rule clean:
  the generic primitive is "fuse any periodic ROM region of N bytes × K
  reps."

(c) is the JIT-correct pattern: detection cost paid once at load,
runtime = one slot-read + immediate-compare/tick.

## 2026-04-29 — fusion disabled by default (net loss, investigation pending)

Initial fusion fast-forward hook (`column_drawer_fast_forward`,
end-of-tick parallel to `rep_fast_forward`) showed +1.4% wall on
level-load. `bench-doom-gameplay.mjs` flipped sign:

| Window               | fusion off  | fusion on (with byte-0/1 fast-out) |
|----------------------|------------:|-----------------------------------:|
| ticks/sec            | 333K        | 319K (-3%)                         |
| simulatedFps         | 34.7        | 35.4 (+2%)                         |
| vramFps              | 1.6         | ~1.7 (noise)                       |
| cycles/sec           | 4.97M       | 4.83M                              |

(Without fast-out, -34% throughput from per-tick 21-byte ROM scan.)

Fusion *does* fire (cycle delta confirms), but per-tick detection cost
across 10M non-firing ticks/s exceeds savings. 20M `state.read_mem`/s
just to detect.

**Disabled by default** (`CALCITE_FUSION_FASTFWD=1` to enable). To
re-enable needs: profile gameplay fire rate; gate on coarser hot-IP
signal (e.g. CS=0x55); move detection from end-of-tick to hot-IP
callback; tune cycle-charge (currently 50/iter).

Simulator + lowerer (88.6% body compose, 94% op shrink) are correct;
runtime hook needs smarter trigger.

## 2026-04-29 — fusion-sim: 88.6% body compose on doom column-drawer

Pushed body-composition probe from 52% → 88.6% FULL via real
dispatch-table support and per-byte decoder pinning.

**Probe** (`crates/calcite-cli/src/bin/probe_fusion_compose.rs`):
- Per-body-byte decoder pin table — for each opcode-byte, asserts
  `--prefixLen`, `--opcode`, `--mod`, `--reg`, `--rm`, `--q0`/`--q1` at
  fire time.
- Body-invariant slot pins each fire: `--hasREP=0`, `--_repActive=0`,
  `--_repContinue=0`, `--_repZF=0`, all segment-override flags = 0.
- Skip non-fire bytes (modrm + immediates) — phantom dispatches were
  bailing.
- Bail-reason histogram (`CALCITE_DUMP_BAIL_OPS`).

**fusion_sim** (`crates/calcite-core/src/pattern/fusion_sim.rs`):
- `Op::DispatchChain` Const-keyed: walk `chain_tables[chain_id]` with
  Const value, jump body PC or `miss_target`. Eliminated 8 dispatch*
  bails (bytes 2 0xd0, 15 0x81).
- `Op::Dispatch` Const-keyed: recursively simulate HashMap entry's ops,
  write `result_slot` into `dst`. Threads `dispatch_tables` through
  `simulate_ops_full_ext`.
- `Op::DispatchFlatArray` non-Const → new `SymExpr::FlatArrayLookup`.

**Probe results** (44 fire tables, 21-byte body):
```
                  FULL   partial   bail
baseline           23     16        5      52.3%
+ pin per-byte     27     17        0      61.4%
+ skip non-fire    30     14        0      68.2%
+ DispatchChain    36      8        0      81.8%
+ Dispatch         38      6        0      86.4%
+ flag invariants  39      5        0      88.6%
+ FlatArrayLookup  39      5        0      88.6% (no change)
```

Last 5 partials are deep flag-side: first FlatArrayLookup result flows
into `BranchIfNotEqLit` needing Const but getting symbolic — needs
symbolic branch outcomes (partial compilation through if-trees), out of
scope.

Sample composed expressions (post-body state vs entry-state slots):
```
table 40: LowerBytes(BitOr16(Slot(--rmVal16), Add(Slot(--immByte), Mul(BitExtract(Slot(--immByte), Const(7)), Const(...))))
table 42: LowerBytes(Add(Floor(Div(Slot(--rmVal16), Const(2))), Mul(BitExtract(Slot(--rmVal16), Const(0)), Const(...))))
table 50: Shr(LowerBytes(Add(Floor(Div(Slot(--rmVal16), Max([Const(1), Slot(--_pow2CL)]))), ...
```

Tests: 13 fusion_sim pass. wasm32 clean.

**SymExpr → Op lowering** (same session). `SymExpr::lower_to_ops` emits
flat `Vec<Op>`. Lit-folded fast paths (`AddLit`/`SubLit`/`MulLit`/
`ShlLit`/`ShrLit`/`AndLit`/`ModLit`) when one operand is Const. 3
round-trip tests confirm `simulate(ops) → expr → lower(expr) →
simulate` matches (16/16 fusion_sim green).

End-to-end shrink (39 FULL tables → fused op sequences):
```
total original ops: 2174
total fused ops:    131
shrink:             94.0%
```

Per-table range: 99.8% (420 → 1 op, flag tables collapse to Const) down
to -50% on some pixel-write expressions (no CSE in naive lowering).

**Memory-write capture extended**: `simulate_with_effects_ext` threads
chain_tables + dispatch_tables. StoreMem/StoreState inside
DispatchChain/Dispatch entries captured to `SimResult.writes`.

**Not done**: runtime CS:IP fusion-site detector, `Op::FusedBody`
variant, runner integration, correctness verification. Real compiler
work, est. 3-5 sessions.

**Smoke gate observation** (pre-existing, not regression): zork1,
montezuma, dos-smoke fail `tests/harness/run.mjs smoke` with runTicks=0
because compile through `calcite-debugger` takes ~8s vs 15s wall budget.
Independent of this session. 4 fast-compiling cabinets (hello-text,
cga4-stripes, cga5-mono, cga6-hires) pass. Fix: raise budget to 30s, or
runner uses calcite-cli (~3.8s compile).

**Runtime fast-forward landed**: end-of-tick hook in `compile.rs`
detects column-drawer body in ROM at current CS:IP, bulk-applies net
effect derived from x86 opcode definitions (two memory reads
palette+colormap, AX broadcast, two stosw, DI advance + 0xEC, DX advance
+ BP). Up to 16 stacked iterations per fire. Gated by
`CALCITE_FUSION_FASTFWD` (later disabled by default — see above).

Level-load measurement (29.5M ticks):
```
fusion OFF: 135.837s / 323,102,046 cycles
fusion ON:  133.948s / 323,246,537 cycles
Δ: 1.4% wall faster, +144,491 cycles (0.04%)
ticksToInGame identical.
```

(Later invalidated by funnel data — see top entry.)

Open follow-ups: gameplay-frame bench; auto fusion-site catalogue from
byte_period; cycle-charge tuning; regression bisection.

## 2026-04-29 — calcite-v2-rewrite Phase 1 lands

Parallel stream, branch `calcite-v2-rewrite`. Clean rewrite from
`ParsedProgram` (parser output) instead of `Vec<Op>` (v1 bytecode), so
DAG + rewriters aren't downstream of v1 pattern decisions.

**Phase 1**: v2 DAG walker matches Chrome on primitive conformance.
Backend enum (`Bytecode | DagV2`) on `Evaluator`; default Bytecode,
opt-in via `set_backend(Backend::DagV2)`.

**Phase 0.5 conformance**: v2 41 PASS / 5 SKIP / 3 XFAIL — identical to
v1. Same 3 documented gaps (div-by-zero serialisation, ignored-selector,
var-undefined-no-fallback invalidity).

**Walker**: terminals topo-sorted at DAG-build by `LoadVar Current`
deps. Per-tick value cache (state-var slots `Vec<Option<i32>>`, memory
sparse `HashMap`). `LoadVar Current` reads cache then committed state;
`LoadVar Prev` reads committed directly. Buffer-copy assignments
(`--__0/__1/__2`) skipped — prefix-stripped slot model already exposes
prior tick as `LoadVar Prev`.

**Phase 1 stubs**: `FuncCall` delegates to v1's `eval_function_call` by
rebuilding `Expr::Literal` args. `IndirectStore` (broadcast write) stub;
conformance suite doesn't exercise broadcasts.

**Two v1 fixes ported in** (real CSS-spec compliance):
- `compile.rs`: gate `rep_fast_forward` on new
  `CompiledProgram::has_rep_machinery` flag (true iff program declares
  `--opcode`). Fixes pre-existing main-branch panic on every cabinet
  without `--opcode`.
- `CalcOp::Mod` fixed in 4 places (compile const-fold, exec_ops Op::Mod,
  Op::ModLit, eval.rs interpreter) → CSS-spec floor-mod
  (`mod(-7, 3) == 2`), not Rust `%`. Caught by `calc_mod_negative`.

`cargo test -p calcite-core`: 196 pass (5 pre-existing rep-fast-forward
fails unrelated). `wasm-pack`: clean.

Next gate: Phase 2 ≥30% DAG node-count reduction on Doom — needs Doom
in worktree + broadcast/dispatch recognisers consuming
`prebuilt_broadcast_writes`. Phase 1 wraps; not merging — owner
reconciles streams.

## 2026-04-28 — calcite Phase 3 prototype: closure backend

Option (c) per mission doc. Each block lowers to `Vec<Box<dyn Fn>>` +
pre-resolved `TerminatorPlan`. No match-on-Op on hot path. Specialised
closures for ~10 common ops; rest fall through to exec_ops on one-op
slice.

162 tests green: backend_equivalence (bytecode/dag/closure 200 ticks
bit-identical), primitive_conformance under all three. wasm32 clean.
web/demo.css throughput: bytecode 261k t/s, dag 210k, closure 220k —
1.19× slower than bytecode, matches spec for prototype with only 10/~50
ops specialised. (c) ceiling 3-5× with full specialisation.

Bugs found writing closure backend:
- `Op::AndLit` val is mask bits, not bit-width. Phase 2's BitFieldMatch
  + LitMerge::And had wrong semantics; fixed.
- `ShrLit`/`ShlLit` are signed i32, not unsigned. Closure was masking
  u32; fixed.

Phase 3 main (option (a) hand-emitted wasm) deferred — weeks of work,
prototype validates the lowering shape codegen would build on. Revisit
once Doom cabinet is in worktree.

## 2026-04-28 — calcite Phase 2: recogniser substrate

14-shape idiom catalogue (derived from `kiln/emit-css.mjs`), `Pattern`
trait + driver in `dag/normalise.rs`, 9 generic recognisers in
`dag/patterns.rs` (LitMerge, BitField, Hold, RepeatedLoad). Annotations
parallel to ops → bit-identical by construction. 161 calcite-core tests
green; Phase 1 gates pass; wasm32 clean. Annotation density on
`web/demo.css` 0.1% — expected (v1 already collapses dominant shapes).
Real metric is Doom; revisit when in worktree.

## 2026-04-28 — Load-time fusion: byte_period + fusion_sim

Bottom two layers of load-time fusion pipeline.

**Layer 1: byte_period detector** (calcite, generic). Walks rom-disk,
finds periodic regions (period P, K reps). doom8088: 610ms over 1.83MB,
4065 regions. Headline: 21×16 at offset 86306 (column-drawer kernel),
21×14 sibling at 86661 (variant). 30 total occurrences of the 21-byte
body. Driver: `probe-byte-periods`.

**Layer 2: fusion_sim symbolic interpreter** (calcite, generic). Walks
compiled Op trees, threads slot reads as `SymExpr::Slot` free vars,
composes arith/bitwise symbolically. Distinguishes calcite's `And`/
`AndLit` (lowerBytes truncation) from `BitAnd16` (true bitwise). Bails
on branches, memory side-effects, unsupported variants. Driver:
`probe-fusion-sim`.

**Concrete win** in table 21 (232-entry per-register dispatch,
`result_slot=386862` = `--ip`): IP-advance composed for **12 of 15**
body bytes:
- `0x88` → `Add(Const(2), Slot(37))` — 2-byte instr base + offset
- `0x81` → `Add(Add(Const(2), Slot(37)), Const(2))` — 4-byte
- `0xea` → `Add(Slot(27), Mul(Slot(28), Const(256)))` — far jump

3 bail: `0xe8` (Div), `0xab` STOSW (Branch on `--df`), `0xcb` (nested
LoadMem). `0xab` needs branch eval under known-flag assumptions
(CLD before body → `--df=0`).

Files: `crates/calcite-core/src/pattern/byte_period.rs`,
`pattern/fusion_sim.rs`, `crates/calcite-cli/src/bin/probe_byte_periods.rs`,
`probe_fusion_sim.rs`. 19 unit tests pass (10 + 9). Smoke not re-run
(diagnostic-only, no execution-path changes).

### 2026-04-28 (followup) — Body-composition probe

`probe-fusion-compose`: simulates every dispatch table's entries across
the 21-byte body in sequence (slot env threaded), reports per-table
FULL/partial/BAIL.

Extended fusion_sim Ops: `Bit`, `Div`, `Round`, `Min`, `Max`, `Abs`,
`Sign`, `Clamp`, `CmpEq`, `DispatchFlatArray` (const-key). Added
`Assumptions` for resolving `LoadState`/`LoadStateAndBranchIfNotEqLit`
against compile-time-known flags (e.g. `--df=0` after CLD).

doom8088 column-drawer body (44 fire tables):
- Initial: 14/44 FULL (32%)
- +CFG/assumptions: 14/44 (no change)
- +Bit/Div/Round/Min/Max/Abs/Sign/Clamp/CmpEq: 23/44 (52%)
- +DispatchFlatArray (const-key): 23/44 (no change — keys symbolic)

Remaining bails: `Branch (non-const)` byte 4 (0x89 `mov r/m, r` with
non-const reg comparison) or `DispatchFlatArray (non-const key)` byte 18
(0x00 imm). To push past 52%: track register-shaped slots symbolically
(partial register lattice), or SymExpr nodes for symbolic array indexing.

**Decision** (later superseded by 88.6% session above): paused. Code
stays — correct, well-tested diagnostic infra.

## 2026-04-28 — Replicated-body recogniser: built, dead lead

Generic recogniser folding unrolled straight-line regions into
`Op::ReplicatedBody`. Period detector + per-Op-variant operand stride
classifier + eval arms in 3 runners (production/profiled/traced) +
pipeline wiring after `compact_slots`. 34 unit tests, smoke green (7
carts).

**doom8088: zero regions folded.** `CALCITE_DBG_REPL=1`: largest
straight-line region in 405K-op main array is **32 ops**; 11 regions
across 24,596 reach 16-op threshold; period-detector finds no period.

Why: asm-level "16× unrolled XLAT body" lives in `i_vv13ha.asm` etc. as
16 back-to-back 6-op pixel kernels, but **Kiln compiles each x86 instr
into its own CSS dispatch entry**. Repetition is at runtime (dispatch
loop fires opcodes 1..6 sixteen times), not in static op stream.
Detecting it would need a dispatch-trace cycle analyser — different
problem, and risks "calcite knows about emitter-shaped opcodes" cardinal
violation.

Lesson: *measure static shape calcite sees before designing recogniser
around asm shape*.

Code stays in main: correct, ~0ms compile cost when nothing matches,
may fire on future cabinets with flat unrolled bodies.

Files: `crates/calcite-core/src/pattern/replicated_body.rs` (~750 LoC
incl. tests), `compile.rs` (Op::ReplicatedBody variant, eval arms in 3
runners, `recognise_replicated_bodies` pass).

---

## 2026-05-01: Repo cleanup: Chunk B — CSS-DOS-shaped code removed

Per `../CSS-DOS/docs/audit-summary-and-plan.md` Chunk B. Calcite is a
generic CSS evaluator; CSS-DOS-the-platform's cabinet sources, dev
server, and BIOS-test harness should not live here.

Deleted:
- `site/`, `programs/`, `output/`, `serve.mjs`, `serve.py` — CSS-DOS-
  shaped infrastructure (cabinet sources, dev server, pre-built outputs).
- `target/release/calcite-debugger.exe.old` — debris left by
  `kill-and-rebuild.bat`.
- `bench-splash.bat`, `run-bios-test.bat`, `run-oldbios-test.bat`,
  `run-splash.bat`, `run-web.bat`, `run-js.bat` — all driven by
  CSS-DOS-the-platform concerns.
- `tools/fulldiff.mjs`, `tools/ref-dos.mjs` — both BROKEN (import
  the deleted `transpiler/` directory). Replacement is
  `tests/harness/pipeline.mjs fulldiff` in CSS-DOS.

Archived (moved to `tools/archive/`, history preserved):
`diagnose.mjs`, `codebug.mjs`, `boot-trace.mjs`, `calc-mem.mjs`,
`ref-emu.mjs`, `compare.mjs`, `serve-js8086.mjs`, `serve-web.mjs`,
`test-daemon-smoke.mjs` — all shell out to `../CSS-DOS/builder/build.mjs`,
which assumes a sibling repo we don't own. Reversible if anything
turns out to be load-bearing.

Modified:
- `crates/calcite-cli/src/menu.rs` — stripped the `node ../CSS-DOS/
  builder/build.mjs` shell-out. The interactive picker now only lists
  pre-built `.css` cabinets in the calcite root and `./cabinets/`.
- `CLAUDE.md` — removed references to deleted things and the
  "Pre-built `.css` cabinets in `output/`" framing.

Verification: `cargo test --workspace` and `wasm-pack build
crates/calcite-wasm --target web` both pass. Branch
`cleanup-2026-05-01`, not yet pushed.

---

## 2026-04-16: big round of peephole/specialization wins

Cumulative commits:
- 8e2ccd8 — fuse LoadState + BranchIfNotEqLit
- 32e8479 — skip per-tick state_vars.clone() in run_batch
- 1930b52 — skip per-tick slot zeroing in execute()
- a7b1625 — dense-array fast path for DispatchChain (had latent bug)
- cb4cbae — AddLit/SubLit/MulLit variants
- 2ccceb2 — AndLit/ShrLit/ShlLit/ModLit variants
- ae1ae51 — fix: flat_table targets weren't remapped in fuse_loadstate_branch

Measured rogue.css: ~6K → ~190K ticks/s (**~32×**).
Measured fib.css: ~7K → ~280K ticks/s (**~40×**).
Measured bootle.css: ~7K → ~190K ticks/s (**~26×**).
Measured splash-fill (bootle-ctest.css): ~17s → ~11s (1,828,538 conformant
ticks at ~170K t/s).

Biggest single win was the dense-array DispatchChain. Most other wins are
5–15% each.

**Lesson: verify halt/conformance on every perf commit.** The dense-array
bug was invisible from pure ticks/s numbers (they got better) but the
program was producing wrong results post-fusion. Caught only when I
noticed `Cycles: 0` in the final bench writeup. Add cycleCount/halt
sanity checks to the perf workflow.

Session notes (meta): most "regressions" of smaller changes were
thermal-throttle noise; once the machine cooled down, several were
actually neutral. Consistently alternating change/baseline runs
interleaved is the only reliable way to separate signal from noise
on this laptop. Three runs per side, take the mean.

### Things that didn't work (saved for reference)
- LoadSlot + BranchIfNotEqLit fusion: only 682 fusions found, neutral.
- Shrinking Op enum (Box<Vec<Slot>> for Min/Max): 32→24 bytes regressed bench.
- LTO=thin: neutral; LTO=fat: -30%.
- MIN_CHAIN_LEN=3: regression (3-chain lookup slower than linear compare).
- LoadStateVar specialization: regressed (match dispatch pressure outweighed).
- Local copy-propagation / dead LoadSlot elim: either neutral (no dead-code
  pass) or broke or-conditions test (too aggressive liveness heuristic).

---

## Current priority: Mode 13h blitting is painfully slow

Filling the 320×200 framebuffer once takes roughly a minute — you can
watch the pixels scroll into place. Even after the compound-AND fuser
landed (3× overall speedup, see below), the mode 13h splash in
`bootle-ctest.css` is nowhere near usable. We need something like a 100×
improvement on tight inner loops like this.

Working through candidates in [docs/optimisation-ideas.md](optimisation-ideas.md).
Stacking order: ~~native bitwise recognition~~ (done) → dead LoadLit
sinking → wider dispatch-chain recognition → change-gated ops → affine
self-loop fixed-point recognition (the big structural move) →
value-keyed region memoisation.

---

## 2026-04-15: Native 16-bit bitwise recognition (idea (a))

CSS-DOS `--and`, `--or`, `--xor`, `--not` are 32-local bit-decomposition
functions (16 `mod(round(down, var/2^k), 2)` ops per input, then a
16-term reconstruction sum). Each call compiled to ~100 ops.

Added a body-shape recogniser `classify_bitwise_decomposition` and four
native ops `BitAnd16` / `BitOr16` / `BitXor16` / `BitNot16`. The
recogniser never looks at function names — it matches on the bit-extract
shape of the locals and the per-bit combine pattern in the result sum
(`a*b` → AND, `min(1,a+b)` → OR, `min(1,a+b)-a*b` → XOR, `1-a` → NOT).

Result on bootle-ctest cold (300K ticks):

- Previous baseline: 134K ticks/s (17.1% of 8086), 2472 main-stream ops/tick.
- After: **259K ticks/s** (32.3% of 8086), **1342 main-stream ops/tick**.
- 1.9× speedup; 45% fewer ops per tick.
- Conformance: compile vs interpret diffs at ticks 500K/1M/2M are
  unchanged (6/10/3) vs baseline — those are a pre-existing divergence,
  not introduced by this change.

---

## 2026-04-15: Flat-array fast path for single-param literal dispatch

### Context

CSS-DOS's rom-disk feature emits `@function --readDiskByte(--idx)` with
one branch per disk byte — ~68K branches for a ~68 KB floppy, ~1.5M
for Doom8088 (future). The generic dispatch compile path at
`compile_dispatch_call` walks every entry and produces a per-entry
`Vec<Op>` sequence, which froze compile for tens of minutes and blew
through 48 GB of RAM on bootle+rom-disk.

### Change

Wired up `Op::DispatchFlatArray`, a pre-existing-but-inert op. The
builder (`try_build_flat_dispatch`) fires when a dispatch has:

- ≤1 parameter, non-empty,
- every entry is an integer literal representable as i32,
- fallback is an integer literal,
- `max_key - min_key + 1 ≤ 10_000_000` (caps worst-case array at ~40 MB).

When all guards pass, the whole table compiles to a single `Vec<i32>`
stored on the program, and the call site becomes a single op doing a
bounds-checked array index. Multi-parameter, non-literal, and sparse
dispatches fall through to the old path unchanged.

Also added a name-keyed cache: repeated call sites of the same function
share the same array. Critical because the rom-disk window has 512
dispatch sites (one per byte in the 0xD0000–0xD01FF window), each
calling `--readDiskByte`.

### Results

Rogue (unrelated to rom-disk, just the standard benchmark):
- Compile: unchanged (no literal single-param dispatches in plain rogue).

Bootle + rom-disk (457 MB CSS, 723K properties, 1.45M assignments,
65 functions including the 56794-entry `--readDiskByte`):
- Parse: 4.7s
- Compile: **29s → 16s** after the name-keyed cache fix; was previously
  frozen indefinitely with the 48 GB allocation on the 2-parameter
  form, and ~79s on the first 1-parameter form before the cache.
- 1 tick: 74 µs.
- Bootle boots end-to-end through the rom-disk path, verified live in
  the interactive CLI.

### Open follow-ups

- Profile the runtime cost of the array lookup under heavy INT 13h load
  (REP MOVSW through the window does 256 reads per sector). Should be
  negligible vs. the slow-path HashMap it replaces.
- Retest Zork+FROTZ (~284 KB disk → ~284K dispatch branches); within
  the i32 literal + 10M span guards, so should take the fast path.

### Other changes bundled with this

Unrelated to the fast path but shipped together:
- `calcite-cli` gained an interactive program picker (grid menu,
  arrow-key navigation) when invoked without `--input`.
- Parse and compile phases now render progress bars to stderr (can be
  disabled with `CALCITE_NO_PROGRESS=1`).
- `--ticks` is now optional; omitting it runs indefinitely in
  interactive mode.

---

## 2026-04-14: V4 baseline + bottleneck analysis

### Context

CSS-DOS v4 landed. It boots rogue, fib, bootle, etc. in ~300K ticks.
Current speed: **~4800 ticks/s on rogue** (0.9% of real 8086 at 4.77 MHz).
Goal: much faster.

### Profiling infrastructure added

Added `calcite-bench` (headless benchmark binary) with `--profile` for
granular per-phase and per-op-type breakdown. See
[docs/benchmarking.md](benchmarking.md) for usage.

### Top-level phase breakdown

Profiled rogue.css, 2000 ticks after 200 warmup:

| Phase | % of tick | Avg/tick |
|---|---|---|
| **Linear ops** | **91.8%** | 231us |
| Dispatch lookups | 5.1% | 13us |
| Change detect | 2.2% | 5us |
| Broadcast writes | 0.5% | 1us |
| Writeback | 0.2% | 0.4us |
| Everything else | <0.2% | — |

**All optimisation work should target the bytecode interpreter loop
(`exec_ops` in `compile.rs`).**

### Op frequency breakdown

The 102K ops/tick break down as:

| Op | Per tick | % of ops |
|---|---|---|
| **LoadLit** | 34,475 | 33.3% |
| **BranchIfZero** | 34,140 | 33.0% |
| **CmpEq** | 34,118 | 33.0% |
| LoadSlot | 196 | 0.2% |
| Mul | 114 | 0.1% |
| Add | 77 | 0.1% |
| everything else | <65 | <0.1% each |

**99.3% of all ops are three instructions in equal proportion:**
`LoadLit + CmpEq + BranchIfZero`.

### What this means

The CSS has ~34K if-chains per tick, each checking `if(style(--prop: N))`.
Per tick, exactly **one** of these 34K matches. The other 33,999 all fail
and branch over immediately.

This is the compiled form of CSS patterns like:

```css
--result: if(style(--opcode: 0)) { ... }
          if(style(--opcode: 1)) { ... }
          if(style(--opcode: 2)) { ... }
          /* ... 34K more ... */
```

Each `if(style(--prop: N))` compiles to:

```
LoadLit   slot[X] = N          # the constant to compare
CmpEq     slot[Y] = (slot[prop] == slot[X])
BranchIfZero slot[Y] → skip    # jump past body if no match
... body ops ...               # only reached for the one match
```

The pattern recogniser already converts `if(style())` chains into dispatch
tables (HashMap lookups) when it detects ≥4 branches on the **same property**.
But these 34K comparisons are **not being caught** — either because they test
different properties, or because the chain structure doesn't match the
recogniser's expectations.

### Optimisation directions (in order of expected impact)

1. **More aggressive dispatch table recognition.** If the pattern recogniser
   can catch these 34K if-chains, they collapse to one HashMap lookup per
   tick. That would eliminate >99% of ops. This is the 100x opportunity.

2. **Fused CmpEq+Branch op.** Since CmpEq is always followed by
   BranchIfZero on the same result slot, fuse them into
   `BranchIfNotEqual { a, b, target }`. Eliminates the intermediate slot
   write and halves the match-dispatch overhead for uncaught patterns.

3. **Skip-chain optimisation.** Recognise runs of `LoadLit + CmpEq +
   BranchIfZero` all testing the same slot and emit a dispatch at compile
   time.

### Branch statistics

- 34,140 branches per tick
- 99.9% taken (i.e., the condition is zero → jump)
- Only 47 branches per tick fall through (the one matching case + a few others)

This is a massive dead-code-skip pattern. Most of the bytecode stream
exists only for the rare tick where that particular property matches.

### Key numbers to track

When making changes, run:

```sh
cargo run --release --bin calcite-bench -- -i output/rogue.css -n 5000 --warmup 500
```

Baseline (2026-04-14, no profile overhead):
- **rogue.css**: ~4800 ticks/s, 0.9% of 8086
- **fib.css**: ~5300 ticks/s, 1.0% of 8086
- ~204us/tick (rogue), ~187us/tick (fib)

---

## 2026-04-14: Fused CmpEq+Branch (BranchIfNotEqLit)

### Investigation

Profiling showed 99.3% of ops were `LoadLit + CmpEq + BranchIfZero`
triplets. Investigation revealed:

1. **Dispatch table recognition is working** — ~179 tables with 34,850
   entries are created. Only ~100 small chains are missed (compound
   conditions and multi-property tests).

2. **The 34K branches per tick come from inside dispatch table entries**
   and the main bytecode stream's linear if-chains. Each opcode entry
   contains further if-chains for register selection, addressing modes,
   etc. These are too small or complex for dispatch table recognition.

3. **402K ops in the main stream, 1.13M ops in dispatch entries** — most
   are the same `LoadLit + CmpEq + BranchIfZero` triplet pattern.

### The fix: `BranchIfNotEqLit` fused op

Added a peephole pass (`fuse_cmp_branch`) that runs after compilation and
before slot compaction. It scans for the pattern:

```
LoadLit(dst=X, val=N) → CmpEq(dst=Y, a=P, b=X) → BranchIfZero(cond=Y, target=T)
```

and replaces it with a single:

```
BranchIfNotEqLit(a=P, val=N, target=T)
```

This eliminates two intermediate slot writes and two match-dispatch cycles
per branch test. The pass also fuses inside dispatch table entry ops and
broadcast write ops.

### Results

| Program | Before | After | Improvement |
|---|---|---|---|
| rogue.css | ~4800 ticks/s | **6054 ticks/s** | **+26%** |
| fib.css | ~5300 ticks/s | **7066 ticks/s** | **+33%** |
| bootle.css | — | **7214 ticks/s** | — |

Main-stream ops per tick dropped from **102K to 35K** (3x reduction).
96.4% of remaining ops are now `BranchIfNotEqLit`.

Correctness verified: bootle shows hearts, rogue boots to DOS.

### Remaining opportunities

The dispatch table entries still execute unfused ops (80 sub-ops per
dispatch on average). The fused op handles the main stream and entry
ops arrays, but dispatch entries that are executed via `exec_ops` (the
non-profiled path) also benefit from the fusion.

Next directions:
- The 35K `BranchIfNotEqLit` ops per tick are still the bottleneck.
  These are all testing the same few properties against different
  constants. A sorted-array binary search or perfect hash at compile
  time could collapse them further.
- Multi-property dispatch: chains testing `--_tf` and `--opcode`
  together could be split into nested dispatches.

---

## 2026-04-14: Unchecked slot indexing in exec_ops hot loop

### The fix

Switched all `slots[idx as usize]` accesses in `exec_ops` (compile.rs) to
`unsafe { *slots.get_unchecked(idx) }` / `get_unchecked_mut`, hidden behind
`sload!` / `sstore!` macros with `debug_assert!` for debug builds. Also
unchecked the `ops[pc]` load and the `dispatch_tables[table_id]` lookup.

Safety rests on invariants the compiler already guarantees: every slot
index in an op is `< slot_count`, every branch target is `<= ops.len()`,
every `table_id` is a valid index. Debug builds still check.

Also reordered the match so `BranchIfNotEqLit` (96% of ops) is the first
arm — helps the compiler lay out the jump table favorably.

### Results

| Program | Before (BranchIfNotEqLit only) | After (unchecked) | Improvement |
|---|---|---|---|
| rogue.css | ~6316 ticks/s | **~7100-7400 ticks/s** | **+13-17%** |
| fib.css | ~7066 ticks/s | ~7900 ticks/s | +12% |
| bootle.css | ~7214 ticks/s | ~7450 ticks/s | +3% |

Run-to-run noise on Windows is ~±5%. The rogue gain is reliably above +10%
across multiple runs.

Correctness: all 27 core tests pass; rogue boots to the same
`IP=890, 404767 cycles` state as before at tick 30000.

### Why it helps

The bytecode interpreter runs `match` + slot index on every op, 34K ops/tick.
Each `slots[idx as usize]` does a bounds check (compare, branch). LLVM
cannot elide these because slot indices come from the op struct, not a loop
induction variable. Skipping them is safe because the compiler statically
allocates slots and never emits an out-of-range index.

### What did NOT work

- **Hot-chain peel** inside the `BranchIfNotEqLit` arm (speculatively
  handle adjacent BranchIfNotEqLit ops in an inner loop without re-entering
  the match). Tested: slowed rogue from 7100 → 6500. The extra inner branch
  disrupted branch prediction and the compiler couldn't inline cleanly.
  Reverted.

### Session notes (meta)

- The session started with a stale diagnostic reporting "takes 4 arguments
  but 5 supplied" that pointed at an exec_ops call site which was actually
  correct. The build succeeded; the diagnostic was cached. Worth remembering
  to trust `cargo build` over rust-analyzer's live diagnostics when they
  disagree.

---

## 2026-04-14: Dispatch-chain recognition (DispatchChain op)

### The fix

Added a new op `DispatchChain { a, chain_id, miss_target }` and a
post-fusion pass `build_dispatch_chains` that detects runs of
`BranchIfNotEqLit { a: P, val: V_i, target: T_i }` where each miss target
points at the next branch-on-same-slot, and collapses them to a single
HashMap lookup.

Implementation sketch:
- For each `BranchIfNotEqLit` at position `i` on slot `P`, walk the chain
  by following each `target`. If the target points to another
  `BranchIfNotEqLit` on the same slot, extend the chain.
- If ≥ `MIN_CHAIN_LEN` (currently 6) branches accumulate, build a
  `DispatchChainTable: HashMap<i32, u32>` mapping each `V_i → body_pc`
  (body_pc = branch_pc + 1).
- Replace `ops[i]` with `DispatchChain`. Leave the rest of the chain in
  place as dead code — keeps any external jumps into the middle of the
  chain valid.

At runtime, `DispatchChain` reads `slot[a]`, looks up in the table, and
either jumps to the matching body PC or to `miss_target`. Eliminates up to
~30 `BranchIfNotEqLit` ops per chain per tick.

### Results

Steady-state benchmark on rogue (20K ticks, 5K warmup):

| Metric | Before | After |
|---|---|---|
| Ticks/s | ~7,100 | **272,194** |
| Cycles/s | 67 KHz | **4.35 MHz** |
| % of 4.77 MHz 8086 | 1.4% | **91.2%** |
| μs/tick | 140 | **3.7** |

Short 5K-tick bench (warmup 1K): 137K-151K ticks/s on rogue/fib/bootle
(all previously 7-8K).

Longer 30K-tick run from cold boot shows smaller apparent gains
(5559 ticks/s), because boot code contains many ops outside dispatch
chains and dominates that window. Steady state hot-loop performance is
where the big win materialises.

### Correctness

- All 27 core tests pass.
- Rogue at tick 30000: IP=890, 404,767 cycles — **identical** to baseline.
- Rogue at tick 60000: 832,808 cycles — consistent with baseline (tick
  counts are deterministic so any logic bug would diverge these).

### Design notes

- Dead-code approach (leave intermediate branches in place) was chosen
  over removal because branch targets from elsewhere in the program may
  point into the middle of a chain; tracking and patching those is extra
  complexity for no runtime gain (dead code isn't executed).
- `MIN_CHAIN_LEN = 6` is a rough guess — HashMap lookup is ~30ns, linear
  match-dispatch is ~5ns/op, so break-even is around 6. Could be tuned.
- Chain tables live in `CompiledProgram::chain_tables`, threaded through
  `exec_ops`, `exec_ops_profiled`, and `exec_ops_traced` as an extra arg.

### Session notes (meta)

- Adding a new `Op` variant touches 6+ `match` sites (op_slots_read,
  op_dst, map_op_slots, seed_from_parent, three exec_ops variants). rust-
  analyzer diagnostics for "non-exhaustive patterns" are reliable here —
  use them as a checklist. The function-arity diagnostics stayed stale
  longer than the pattern ones; trusting `cargo build` still pays off.

---

## 2026-04-14: Interactive CLI path was throttling the evaluator

### The discovery

After the two evaluator wins above, the bench reported 270K+ ticks/s but
running the same program via `run.bat` showed only ~5–12K ticks/s. The
user (correctly!) pushed back that they weren't seeing the speedup.

Root cause: `run.bat` invoked `calcite-cli` with `--halt halt`, and the
CLI's `needs_per_tick` check (`cli.verbose || halt_addr.is_some() ||
(interactive && screen_interval > 0) || !key_events.is_empty()`) forced
the per-tick loop. That loop did one `evaluator.tick(state)` plus one
`crossterm::event::poll(Duration::ZERO)` syscall per simulated tick. On
Windows, the event poll is ~1–5 μs per call — which used to be drowned
out by a 140 μs tick, but after the optimisations each tick is ~4 μs, so
the keyboard poll became the dominant cost.

### The fix

Two changes to run.bat / calcite-cli:

1. **run.bat**: Dropped `--halt halt` from the rogue/general program
   launch — unnecessary for interactive use and the main trigger for
   per-tick mode.
2. **calcite-cli**: Rewrote the interactive loop to run in configurable
   batches (`--interactive-batch`, default 50,000). Between batches, the
   CLI still polls keyboard, fires `--key-events`, re-renders the screen,
   and checks `--halt`. Between ticks within a batch, none of that
   happens — `run_batch` goes full-speed. Scripted key events force the
   batch to shrink so they fire at the correct tick; held-key BDA refill
   runs once per batch (BIOS INT 16h busy-spin still sees the key in
   time — 50K ticks ≈ 10.5 ms of sim time, well inside a human keypress).

Physical intuition: at 4.77 MHz the 8086 executes ~80K instructions per
60 fps frame, so a 50K-tick batch is ~10 ms of sim time — imperceptible
for input latency. Earlier default screen_interval=500 ran renders 160×
per sim frame for no benefit.

### Results

Same rogue.css, same 500,000 ticks, via `calcite-cli` (not bench):

| Config | Before | After |
|---|---|---|
| `--screen-interval 500` (default) | ~12,000 ticks/s | **245,000–287,000 ticks/s** |
| `--screen-interval 50000`         | ~12,000 ticks/s | **peaks 346,000 ticks/s (4.36 MHz = 91% of 8086)** |

Cycle count (6,290,226) identical to baseline — no correctness regression.

### Status-line readability

User asked to stop the speed readout from glitching between units (KHz ↔
MHz, different widths). Replaced the live status line with:

- **Fixed-width formatting**: always `X.XX MHz` (8-char wide) regardless
  of speed. No more "100 KHz" ↔ "1.0 MHz" width flips.
- **EMA smoothing** (α = 0.3) on ticks/s so noise doesn't flicker the
  display.
- **Refresh throttle**: the rendered status text only updates every 500 ms
  of wall time; the screen itself still repaints every `screen_interval`
  ticks, but the speed digits stay put between status refreshes.

### Session notes (meta)

- The user's debugging instinct was right before my analysis caught up. I
  built a theory (`event::poll` overhead dominates) and was about to
  write another optimisation when the user asked "isn't it just the
  batch mode thing?" Looking again: the CLI's `run_batch(cli.ticks)`
  path only activates when `needs_per_tick == false`, and any of
  `--halt`, screen_interval > 0, verbose, or scripted key events trips
  it. Answering user doubts honestly saves wall time — I did not need to
  write more code to reach the diagnosis.
- Feature-flagging rule of thumb for this project: if something "should"
  be fast per the bench but isn't in a real run, look first at whether
  the CLI wrapper is forcing a slow path.


## 2026-04-17 — Memoisation viability: Probes 1-4 + runtime period projector prototype

Spec: `docs/superpowers/specs/2026-04-17-memoisation-viability.md`.

### Probes summary

Four probes built as `probe-splash-{memo,trace,affine,period}` binaries
against bootle-ctest.css. First three (per-tick value-keyed memoisation;
LuaJIT-style trace specialisation; consecutive-tick affine store detection)
are **dead ends** — the data rules them out.

Probe 4 (loop-period autocorrelation over the fingerprint stream) **found
the real signal**: splash-fill is a 26-tick microcode iteration, 99.6% of
the splash phase, one affine memory write per iteration (`base +
iter * 1`, constant value = pixel colour). See spec for numbers.

### Runtime projector prototype

New module `crates/calcite-core/src/tick_period.rs` + bench binary
`probe-splash-project`. Pipeline:

1. **Cold phase**: collect 4096 samples (`(pre_tick_vars, first_mem_write)`).
2. **Calibration**: identify "cyclic" slots (≤ min(32, len/64) unique values);
   vote per cyclic slot for its best absolute-value period under
   autocorrelation. Quorum = ≥ half the voting slots agreeing.
3. **Affinity verification**: across `CONFIRM_ITERS+1` candidate iterations,
   verify each state var evolves as `base + k * per_iter_delta` and each
   offset's memory write evolves as `(base_addr + k * addr_stride,
   constant_value)`. Non-affine vars are only tolerated if their delta is 0.
4. **Projection**: at an iteration boundary, advance state vars scalarly
   and fill memory with a `memset` over `N` iterations of writes.
5. **Validation**: after projection, run one real iteration and check that
   post-iteration state matches `anchor + (iters_since_lock+1) * delta`
   absolutely. Miss → cooldown 64 ticks, re-enter Cold.
6. **Rollback**: driver snapshots `state.memory`, `state.state_vars`,
   `state.extended`, `state.string_properties`, `frame_counter` before
   every projection; on validation miss, restores all five.

### Current status — honest assessment

**Correctness: bit-identical to baseline.** Halt tick, memory hash, and
state_vars hash all match baseline (`1828538 / 94e2a9a5d967e282 /
a7d99bf7857452b2`) on bootle-ctest end-to-end.

Early attempts had silent state drift: memory hashed correctly (rollback
caught every miss) but halt tick drifted by 104 and state_vars hash
mismatched. Fixed by changing `validate_iteration` to compare **absolute**
state against `anchor + iters * delta`, not just the incremental `post ==
pre + delta`. Under an affine workload, one real iteration advances by
`delta` from ANY starting state — incremental check is trivially true
even when the projected `pre` is wrong. The absolute check catches it.

But the projector still doesn't pay off:
- **156 of 157 locks miss validation.** The detector locks after
  CONFIRM_ITERS=3 iterations of affine behaviour, but 4 iterations is not
  enough evidence that the next N will ALSO be affine. The spec's data
  backs this up: longest P=26 contiguous run is only 8292 ticks (318
  iterations), so locks happen mostly on shorter runs where projection
  quickly outruns the regime.
- **Net speedup: 1.02×, inside noise.**

This is a prototype, not a ship-ready optimisation. Remaining work to get
a real win:

1. **Stronger lock gate.** Require CONFIRM_ITERS ≥ 10 and a high match
   ratio in the full calibration window (not just the anchor region). In
   theory this moves the false-positive rate below the disruption rate,
   so locks stick.
2. **Reduce calibration cost.** O(n_vars × max_period × window) per
   attempt is ~5ms per 4096 ticks at current settings — on par with the
   tick cost itself. Only recompute when `state_vars` hash rings a bell
   (i.e., skip calibration while the workload looks unchanged).
3. **Smarter projection budget.** Current code doubles on success, resets
   to 4 on miss. In a workload where ~1 in 300 iterations is a disruption,
   the expected-value-optimal starting budget is much higher — but we need
   the lock gate to be reliable first.
4. **Consider whether the detector overhead can be made pay-per-use**: if
   Cold-mode observation costs > compiled-tick cost, we're net-negative.
   The probe's observation adds a state_vars.clone() + first-mem-write
   scan per tick — that's already a significant fraction of tick cost.

### Artifacts

- `crates/calcite-core/src/tick_period.rs` — detector + projector.
- `crates/calcite-cli/src/bin/probe_splash_{memo,trace,affine,period,project}.rs`
  — the four research probes + the end-to-end projector driver.
- `crates/calcite-cli/src/bin/probe_{full_vs_sub,cyclic_slots}.rs` —
  diagnostic probes for fingerprint-strategy selection (kept for future
  reference; they were how I discovered that delta-fingerprint gives 45%
  match rate while a subset-of-cyclic-slots absolute fingerprint gives
  100%).
- `probe.*.log` — raw probe outputs (not committed).

### What the prototype establishes

- The detection pipeline (fingerprint → voting → affine verify) does find
  P=26 on bootle-ctest in a 4096-sample window. With a wider window it
  would also find the harmonics (52, 78, 104, …).
- The rollback pathway keeps memory bit-identical across projection
  attempts even when the projection is wrong.
- The spec's 20–30× upper bound is **not** demonstrated by this code —
  the validation gap collapses every projection back to a rollback.

The easy validation fix landed (absolute-anchor comparison). The hard
remaining work is detector discipline: locks are firing far too
optimistically, so almost every projection gets rolled back. A longer
confirm window or a more stringent agreement threshold should move the
false-positive rate below the workload's natural disruption rate — at
which point the 20–30× upper bound from the spec becomes reachable in
principle. Until that lands, the detector is instrumentation, not
optimisation.


## 2026-04-18 — Splash fill: REP STOSB rewrite + runtime projector stabilised + signature-based detector WIP

Two parallel workstreams this session. One shipped (on CSS-DOS side); one
partially landed (runtime projector fixes); one is research-quality code
that needs a final correctness pass before it's useful.

### CSS-DOS: REP STOSB splash rewrite — shipped

Commit `2acc748` on `../CSS-DOS/master`. `bios/splash.c`'s per-pixel C
loop for the dark-gray fill replaced with an OpenWatcom `#pragma aux
vga_fill` wrapper emitting `rep stosb`. **Splash ticks: 1,828,538 →
194,918. 9.4× fewer CSS ticks** for the 64,000-byte fill. Output CSS
rebuilt via `generate-dos-c.mjs`.

### Runtime projector (`PeriodTracker`) — correctness fix + opts

Prior session left the projector in a broken state: correct detection but
`project()` didn't cap N by counter-zero-crossing, so it over-filled past
REP STOSB end by ~1,500 bytes and corrupted post-fill memory. Fixed:

1. **Zero-crossing cap in `project()`.** For any state slot with non-zero
   delta, bound N so the slot doesn't cross zero during projection. Pure
   observation of slot values, no x86 knowledge — cardinal-rule-safe.
   **Result: memory hash matches baseline bit-identically.**

2. **Opt: no per-tick heap allocations in `observe()`.** `Sample.vars`
   boxes pre-allocated at construction; hot path just `copy_from_slice`s
   into the next ring slot. Probe's `pre_vars` clone replaced with a
   reusable `Vec<i32>` scratch.

3. **Opt: `CALIB_LEN` 4096 → 256** with scaled `MIN_MATCHES` (512 → 32)
   and a cyclic-threshold clamp of `(len/8).clamp(8, 32)`. Gives 16× more
   calibration attempts per workload; still passes the in-module tests.

4. **Opt: `INITIAL_BUDGET` 4 → 64.** Zero-crossing cap is now the safety
   net against overfill, so we can start projecting more aggressively.

**Combined result: stable ~1.25–1.30× median splash speedup (best
~1.42×), memory hash bit-identical to baseline, no missed validations on
the hot path.**

### Why only 1.30× — honest bottleneck analysis

Expected 5–10×, got 1.30×. Measured decomposition:

- **67% of splash** is burnt in Cold-mode calibration before lock. The
  voting-based autocorrelation needs many samples to disambiguate a real
  period from dispatchy-slot coincidences. With MIN_PERIOD=1 (necessary
  for the REP STOSB workload after the rewrite), only one in ~32
  calibration buffers lands with an affine-verifiable window — all the
  others fail `non-affine var with nonzero delta`.
- **Remaining 33%** is projected. Bulk memset saves ~30% of that 33% =
  ~10% of total. Plus some validation overhead saved from the small
  opts.
- **Tracker observation overhead** is only ~5% per tick after opt 1, not
  the ~18% I'd initially guessed from a noisy run.

The wall: lock timing isn't deterministic — it depends on buffer
end-alignment landing inside a phase of the microcode cycle where 5
consecutive tick-to-tick deltas happen to be self-consistent. That only
happens ~1/32 of the time. Shrinking the buffer gives more attempts but
doesn't change per-attempt success rate.

### Structural critique → signature-based cycle detector (WIP)

The real fix is to stop inducing cycles statistically from state_vars
autocorrelation and instead use structural execution signatures: two
ticks that wrote to the same set of state slots with the same
relative-address mem-write pattern did mechanically the same work.
Cycle period falls out in O(ticks), not O(ticks²).

New module `crates/calcite-core/src/cycle_tracker.rs`. Per-tick
signature = hash of (slot-change-set, relative-mem-write-offsets).
Last-seen-at map gives a period candidate in O(1); `CONFIRM_CYCLES=3`
cycles of matching confirms. Handles harmonics via a third-cycle
affine-consistency check at lock time.

**Detection works**: on the real workload (bootle-ctest), the tracker
locks on the period-4 REP STOSB cycle at tick **3,263** instead of tick
131,072. 40× reduction in time-to-lock, correctly identifies
addr_stride=+2/cycle and writes/cycle=2. Unit tests pass.

**Projection is broken**: after the harmonic-rejection gate, the detector
falls through to locking on a DIFFERENT pattern later in the trace (tick
130,948) and my hand-rolled `project()` writes wrong bytes. Memory hash
diverges from baseline. Phase alignment between captured anchor and
current state_vars is the source of the bug — I tried several fixes but
didn't converge in-session.

### Next step (unfinished)

The right shape of the fix: keep `CycleTracker`'s detection primitive
(it solves the real "observe for 131K ticks" problem); throw away my
hand-rolled `project()`; wire CycleTracker's output (period,
addr_stride, write_offsets, per_cycle_delta, anchor_vars) directly into
`PeriodTracker::Mode::Locked` to use the proven projection code. That
gives fast lock + correct project. Expected: the 5–10× that didn't land
this session.

### Honest scope comparison with the original brief

Original brief (from prior session handover) was to pattern-match the
REP STOSB shape at **compile time** and lower it to a new
`Op::MemoryFill` bytecode. I did not do that. I iterated on the
**runtime** detector instead — a different architectural choice with a
different risk profile (less invasive to `compile.rs`, but lower ceiling
than symbolic compile-time analysis). The 9.4× from Task 1 (CSS-DOS
side) is real; the 10×-ceiling of Task 2 is not here.

### Artifacts (this session)

- `crates/calcite-core/src/tick_period.rs` — zero-crossing cap +
  opts 1/2/4. Stable, correct, bench-worthy.
- `crates/calcite-core/src/cycle_tracker.rs` — signature-based detector.
  Detection works, projection broken. Marked as experimental.
- `crates/calcite-cli/src/bin/probe_write_sig.rs`,
  `probe_cycle_detect.rs`, `probe_cycle_project.rs` — diagnostic
  + bench probes for the new detector.
- `../CSS-DOS/bios/splash.c` (commit `2acc748`) — REP STOSB rewrite,
  shipped.
