//! calcite-debugger — MCP debug server for CSS execution.
//!
//! Parses a CSS file once, then exposes step / inspect / compare operations
//! as MCP tools over stdio. Designed for LLM-driven debugging.
//!
//! ## Transport
//!
//! stdio only. **stdout is reserved for MCP protocol frames** — every byte of
//! logging goes to stderr (via `eprintln!` and `env_logger` configured to
//! `Target::Stderr`).
//!
//! ## Tool surface
//!
//! See `DebugSession` impl below; each `#[tool]`-annotated method is exposed.
//! Run with `--list-tools` for a runtime dump (handy when wiring an MCP client).

use calcite_core::{Evaluator, State};
use clap::Parser;
use rmcp::{
    handler::server::{
        router::tool::ToolRouter,
        wrapper::{Json, Parameters},
    },
    model::{ErrorData, ServerCapabilities, ServerInfo},
    schemars,
    tool, tool_handler, tool_router,
    transport::stdio,
    ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "calcite-debugger", version, about = "MCP debug server for calcite")]
struct Cli {
    /// Path to the CSS file to debug. Optional — if omitted, the server starts
    /// with no program loaded and you call the `open` tool to choose one.
    /// When given, `--session <name>` is also required so the pre-loaded
    /// program has a session name.
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Session name to install the pre-loaded program under. Required when
    /// `-i/--input` is given. Ignored otherwise.
    #[arg(short = 's', long)]
    session: Option<String>,

    /// Ticks between full-state base checkpoints. Deltas are recorded every
    /// tick regardless. Default 50,000 matches the PR 2 brief.
    #[arg(long, default_value = "50000")]
    base_interval: u32,

    /// Listen on a TCP address (e.g. `127.0.0.1:3334`) instead of stdio.
    /// Daemon mode — the server keeps all in-memory session state alive
    /// across client reconnects. Each accepted connection gets its own
    /// handler task; the underlying session map is shared, so clients
    /// using the same session name collaborate, and distinct names stay
    /// isolated. Without this flag, the server speaks MCP over stdio
    /// (standard per-client-process mode). See docs/daemon-mode.md.
    #[arg(long, value_name = "HOST:PORT")]
    listen: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool parameter & result types
//
// All types derive serde + schemars::JsonSchema so rmcp can build the
// JSON-RPC tool/list schema automatically.
// ---------------------------------------------------------------------------

#[derive(Deserialize, Serialize, schemars::JsonSchema, Default)]
struct EmptyParams {}

/// For tools that need no params besides the session name.
#[derive(Deserialize, schemars::JsonSchema)]
struct SessionOnlyParams {
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct TickParams {
    /// Number of ticks to advance. Defaults to 1.
    #[serde(default = "default_one")]
    count: u32,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_one() -> u32 {
    1
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SeekParams {
    /// Target tick number. Forward or backward — uses nearest snapshot then replays.
    tick: u32,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct MemoryParams {
    /// Linear address (e.g. 0x400 for BDA, 0xB8000 for text VGA).
    addr: i32,
    /// Number of bytes to read. Defaults to 256.
    #[serde(default = "default_mem_len")]
    len: usize,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_mem_len() -> usize {
    256
}

/// Params for `diff_packed_memory`. Two sessions are stepped forward in
/// lockstep; at each tick a byte-level view of memory derived from the
/// appropriate storage (plain per-byte state vars in `ref_session`,
/// packed `mc{N}` cells in `test_session`) is compared. The tool returns
/// the first (tick, linear_addr) where the byte views disagree, or
/// reports convergence if `max_ticks` is exhausted without a mismatch.
#[derive(Deserialize, schemars::JsonSchema)]
struct DiffPackedMemoryParams {
    /// Reference session name — expected to use per-byte `--m{N}` storage (PACK_SIZE=1).
    ref_session: String,
    /// Test session name — expected to use packed `--mc{N}` storage (PACK_SIZE=2).
    test_session: String,
    /// Maximum ticks to step before giving up. Defaults to 10,000.
    #[serde(default = "default_diff_max")]
    max_ticks: u32,
    /// Inclusive linear start address. Defaults to 0.
    #[serde(default)]
    addr_start: Option<i32>,
    /// Exclusive linear end address. Defaults to 0xA0000 (640 KB conventional).
    #[serde(default)]
    addr_end: Option<i32>,
    /// Pack size of the test session's cells (bytes per cell). Defaults to 2.
    #[serde(default = "default_diff_pack")]
    pack: u8,
}
fn default_diff_max() -> u32 { 10_000 }
fn default_diff_pack() -> u8 { 2 }

#[derive(Serialize, schemars::JsonSchema)]
struct DiffPackedMemoryResult {
    /// True if both sessions agreed byte-for-byte through `ticks_stepped`.
    converged: bool,
    /// Ticks actually stepped. Equals `max_ticks` on convergence, or the tick
    /// where divergence was detected (both sessions are advanced to this tick).
    ticks_stepped: u32,
    /// Linear address of the first divergence, or None if converged.
    first_diff_addr: Option<i32>,
    /// Reference byte at `first_diff_addr`, or None if converged.
    ref_byte: Option<u8>,
    /// Test byte at `first_diff_addr`, or None if converged.
    test_byte: Option<u8>,
    /// CPU state (CS/IP, all six write slots) from the ref session at the
    /// divergence tick. Enough context to determine whether the two
    /// sessions intended the same memory write or diverged earlier in
    /// decode. None when converged.
    ref_cpu: Option<DiffCpuContext>,
    /// Same for the test session.
    test_cpu: Option<DiffCpuContext>,
    /// Human-readable summary.
    summary: String,
}

/// CPU context captured at the divergence tick — just enough to tell
/// whether the CPU in both sessions was trying to do the same write.
#[derive(Serialize, schemars::JsonSchema)]
struct DiffCpuContext {
    cs: i32,
    ip: i32,
    /// `(addr, val)` for every write slot whose addr is non-negative
    /// (the CPU's encoding uses -1 for "unused"). In slot order 0..5.
    writes: Vec<DiffWriteSlot>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct DiffWriteSlot {
    slot: u8,
    addr: i32,
    val: i32,
    live: i32,
}

/// Params for `inspect_packed_cell_table`. Looks up a cell index in the
/// compiled packed-cell address table and reads back the current state
/// value (and the interpreter's view, for cross-check).
#[derive(Deserialize, schemars::JsonSchema)]
struct InspectPackedCellParams {
    /// Session name. Required.
    session: String,
    /// Cell index (N in `mcN`) to inspect.
    cell_index: u32,
    /// Which packed cell table to read from. Defaults to 0 (the first / usually only).
    #[serde(default)]
    table_id: Option<u32>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct InspectPackedCellResult {
    /// Cell index that was inspected.
    cell_index: u32,
    /// Number of entries in the chosen packed cell table. `cell_index >= table_len`
    /// means LoadPackedByte would fall back to cell_addr=0 and return 0.
    table_len: usize,
    /// The state-var address stored in `packed_cell_tables[table_id][cell_index]`.
    /// Should be negative (state var) for a valid cell. 0 means "no entry" — any
    /// LoadPackedByte for a byte inside this cell will return 0.
    cell_addr: i32,
    /// `state.read_mem(cell_addr)`. If `cell_addr` is 0 this is also 0. Otherwise
    /// the current packed cell value (two bytes, low in bits 0..7, high in 8..15).
    cell_value_via_cell_addr: i32,
    /// The state-var index that `mcN` is actually assigned to, via the address map.
    /// Negative (state-var convention). `None` if `mcN` is not a recognised state var.
    expected_state_var_addr: Option<i32>,
    /// State-var value read through the expected address, independent of the
    /// packed cell table. If `cell_value_via_cell_addr != this` the packed
    /// cell table entry is wrong (mismatched address → LoadPackedByte reads
    /// a different cell than the CSS intended).
    expected_state_var_value: Option<i32>,
    /// If there's a literal-exception entry in `packed_exception_tables[table_id]`
    /// for the low byte at cell_index*pack, its value. Takes precedence over the
    /// packed extract in LoadPackedByte.
    exception_low: Option<i32>,
    /// Same for the high byte at cell_index*pack + 1.
    exception_high: Option<i32>,
    /// Pack size (bytes per cell) for this table.
    pack: u8,
    /// Total number of packed cell tables in the compiled program.
    total_tables: usize,
    /// Whether `mcN` exists in state.state_var_index (bare name, no dashes).
    state_var_index_has_bare: bool,
    /// state_var index for `mcN` if present.
    state_var_index_value: Option<usize>,
    /// Lookup result for `property_to_address("mcN")` — without the `--` prefix.
    /// This is what `get_or_build_packed_cell_table` actually calls. If this
    /// is `None` while `expected_state_var_addr` is `Some`, there's a
    /// dash-stripping bug in property_to_address.
    property_to_address_without_dashes: Option<i32>,
    /// Summary string explaining the result.
    summary: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ScreenParams {
    /// Linear address of video memory. Defaults to detected video region or 0xB8000.
    #[serde(default)]
    addr: Option<i32>,
    /// Columns. Defaults to 80.
    #[serde(default)]
    width: Option<usize>,
    /// Rows. Defaults to 25.
    #[serde(default)]
    height: Option<usize>,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CompareReferenceParams {
    /// Reference trace: list of {tick, registers...} entries to diff against.
    reference: Vec<RefTickJson>,
    /// Stop at the first divergence. Defaults to true.
    #[serde(default = "default_true")]
    stop_at_first: bool,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_true() -> bool {
    true
}

#[derive(Deserialize, Clone, schemars::JsonSchema)]
struct RefTickJson {
    tick: u32,
    /// Register values keyed by name (e.g. {"AX": 1234, "IP": 256}).
    /// Use a separate `registers` object — flatten doesn't play well with schemars.
    registers: HashMap<String, i64>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CompareStateParams {
    /// Register values to diff against, keyed by name ("AX", "IP", ...).
    #[serde(default)]
    registers: Option<HashMap<String, i64>>,
    /// Memory ranges to diff: list of {addr, len, bytes} entries.
    #[serde(default)]
    memory: Option<Vec<CompareMemEntry>>,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CompareMemEntry {
    addr: i32,
    len: usize,
    /// Expected bytes (from the reference side) as a u8 array.
    bytes: Vec<u8>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SendKeyParams {
    /// Packed key value: (scancode << 8) | ascii. Use 0 to clear / release.
    value: i32,
    /// Where to write the key:
    /// `bda` (default) pushes into the BIOS Data Area ring buffer at 0x41E,
    /// `keyboard` sets the `--keyboard` CSS state var (used by the v3 microcode
    /// path that converts edges into IRQ 1).
    #[serde(default = "default_key_target")]
    target: String,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_key_target() -> String {
    "bda".into()
}

#[derive(Deserialize, schemars::JsonSchema)]
struct TracePropertyParams {
    /// CSS property name to trace (e.g. "--memAddr").
    property: String,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct WatchpointParams {
    /// Linear memory address to watch.
    addr: i32,
    /// Maximum ticks to run before giving up. Defaults to 100,000.
    #[serde(default = "default_watch_max")]
    max_ticks: u32,
    /// If set, seek to this tick first.
    #[serde(default)]
    from_tick: Option<u32>,
    /// Stop when the byte equals this value (instead of "any change").
    #[serde(default)]
    expected: Option<i32>,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_watch_max() -> u32 {
    100_000
}

/// Discriminated union — pass exactly one variant.
/// Matches the original HTTP `RunUntilCondition` shape.
///
/// Deserialization accepts either the structured object form
/// (`{"property_equals": {"name": "--haltCode", "value": 192}}`) OR a
/// JSON-encoded string containing the same shape. Some MCP clients
/// stringify nested objects in tool parameters; the string form keeps
/// those clients working without forcing them to flatten the schema.
#[derive(Clone, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RunUntilCondition {
    /// Stop when CS:IP matches.
    CsIp { cs: i32, ip: i32 },
    /// Stop when CS matches.
    Cs(i32),
    /// Stop when CS matches and min ≤ IP ≤ max.
    IpRange { cs: i32, min: i32, max: i32 },
    /// Stop when next opcode at CS:IP is 0xCD (any INT).
    Int(bool),
    /// Stop when CS:IP points to `CD <N>`.
    IntNum(i32),
    /// Stop when a CSS property equals a value.
    PropertyEquals { name: String, value: i64 },
    /// Stop when a CSS property changes from its starting value.
    PropertyChanges(String),
    /// Stop when memory byte equals a value.
    MemByteEquals { addr: i32, value: i32 },
    /// Stop when any byte in `[start, end)` is nonzero. Useful for detecting
    /// "this region got populated" — e.g. fire's framebuffer at 0xA0000 or
    /// the VGA DAC palette shadow at 0x100000.
    MemRangeNonzero { start: i32, end: i32 },
}

// Manual Deserialize: accept either the native object shape OR a JSON string
// that re-deserializes to the same shape. Both rmcp's strict schema clients
// and clients that stringify nested objects end up working.
impl<'de> Deserialize<'de> for RunUntilCondition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // The real (strict) externally-tagged form, under a different name
        // so we can delegate to its auto-derived impl.
        #[derive(Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Inner {
            CsIp { cs: i32, ip: i32 },
            Cs(i32),
            IpRange { cs: i32, min: i32, max: i32 },
            Int(bool),
            IntNum(i32),
            PropertyEquals { name: String, value: i64 },
            PropertyChanges(String),
            MemByteEquals { addr: i32, value: i32 },
            MemRangeNonzero { start: i32, end: i32 },
        }
        impl From<Inner> for RunUntilCondition {
            fn from(i: Inner) -> Self {
                match i {
                    Inner::CsIp { cs, ip } => RunUntilCondition::CsIp { cs, ip },
                    Inner::Cs(v) => RunUntilCondition::Cs(v),
                    Inner::IpRange { cs, min, max } => {
                        RunUntilCondition::IpRange { cs, min, max }
                    }
                    Inner::Int(v) => RunUntilCondition::Int(v),
                    Inner::IntNum(v) => RunUntilCondition::IntNum(v),
                    Inner::PropertyEquals { name, value } => {
                        RunUntilCondition::PropertyEquals { name, value }
                    }
                    Inner::PropertyChanges(v) => RunUntilCondition::PropertyChanges(v),
                    Inner::MemByteEquals { addr, value } => {
                        RunUntilCondition::MemByteEquals { addr, value }
                    }
                    Inner::MemRangeNonzero { start, end } => {
                        RunUntilCondition::MemRangeNonzero { start, end }
                    }
                }
            }
        }

        // Buffer into a serde_json::Value first so we can inspect whether
        // the client sent a string or an object.
        let v = serde_json::Value::deserialize(deserializer)?;
        let inner: Inner = match v {
            serde_json::Value::String(s) => serde_json::from_str(&s)
                .map_err(|e| serde::de::Error::custom(format!(
                    "condition was a string but did not parse as a RunUntilCondition: {e}"
                )))?,
            other => serde_json::from_value(other)
                .map_err(|e| serde::de::Error::custom(e.to_string()))?,
        };
        Ok(inner.into())
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RunUntilParams {
    condition: RunUntilCondition,
    /// Maximum ticks before giving up. Defaults to 1,000,000.
    #[serde(default = "default_run_until_max")]
    max_ticks: u32,
    /// If set, seek to this tick first.
    #[serde(default)]
    from_tick: Option<u32>,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_run_until_max() -> u32 {
    1_000_000
}

#[derive(Deserialize, schemars::JsonSchema)]
struct JobIdParams {
    job_id: u64,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct RunUntilCancelResult {
    job_id: u64,
    cancelled: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct DumpOpsParams {
    /// Op range start (inclusive).
    #[serde(default)]
    start: Option<usize>,
    /// Op range end (exclusive).
    #[serde(default)]
    end: Option<usize>,
    /// Property name — dump every op that contributes to this property.
    /// Mutually exclusive with start/end.
    #[serde(default)]
    property: Option<String>,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SnapshotParams {
    /// `create` to checkpoint at the current tick, `list` to enumerate existing snapshots.
    /// Defaults to `list`.
    #[serde(default = "default_snapshot_action")]
    action: String,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_snapshot_action() -> String {
    "list".into()
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SummaryParams {
    /// `start` to reset and begin recording; `stop` to pause; `get` to
    /// return the current set of blocks; `clear` to discard the log.
    /// `start` is the usual entry point.
    #[serde(default = "default_summary_action")]
    action: String,
    /// Max events to keep before recording halts (default 500k).
    /// A REP loop is one event, so 500k is plenty for a 1M-tick run.
    #[serde(default)]
    max_events: Option<usize>,
    /// Also record memory writes? Off by default — turning it on roughly
    /// doubles per-tick overhead because we scan 8 state slots each tick.
    #[serde(default)]
    record_writes: Option<bool>,
    /// IP-window threshold for block breaks (default 256 bytes).
    #[serde(default)]
    ip_window: Option<i32>,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}
fn default_summary_action() -> String {
    "get".into()
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SummariseParams {
    /// Number of ticks to record. Hard-capped at 100000 — the point of this
    /// endpoint is tactical recording over a small window, not bulk tracing.
    /// For longer runs use `run_until` (no recording) to navigate, then
    /// `summarise` over the small interesting window.
    max_ticks: u32,
    /// IP-window threshold for block breaks (default 256 bytes). Smaller =
    /// more, finer blocks; larger = fewer, coarser blocks.
    #[serde(default)]
    ip_window: Option<i32>,
    /// Session name. Required.
    session: String,
}

const SUMMARISE_HARD_CAP: u32 = 100_000;

#[derive(Deserialize, schemars::JsonSchema)]
struct EntryStateCheckParams {
    /// Session name. Required.
    session: String,
    /// CS value to wait for. If omitted, runs until CS is outside the BIOS
    /// (0xF000) and outside the very low IVT/BDA region (CS < 0x0100). For a
    /// .COM loaded via SHELL=, that lands in the program's PSP+0x10 segment.
    #[serde(default)]
    target_cs: Option<i32>,
    /// Maximum ticks to run before giving up. Defaults to 30,000,000 — enough
    /// to clear BIOS splash + DOS boot + CONFIG.SYS processing for a small .COM.
    #[serde(default)]
    max_ticks: Option<u32>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct IvtEntry {
    /// Interrupt number (e.g. 0x10 for video).
    int_num: i32,
    /// Handler segment from IVT slot int_num*4+2.
    seg: i32,
    /// Handler offset from IVT slot int_num*4+0.
    off: i32,
    /// Linear address = seg*16+off.
    linear: i32,
    /// First 8 bytes of code at linear (hex). All-zeros = handler not installed.
    code_hex: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct EntryStateCheckResult {
    /// True if the wait condition was hit; false on timeout.
    found: bool,
    /// Tick we landed on (or last tick reached on timeout).
    tick: u32,
    /// Reason for stopping: "matched", "timeout", "halted".
    reason: String,
    /// All registers + flags at the landing tick.
    registers: BTreeMap<String, i32>,
    /// IVT entries for the interrupts most programs actually use.
    /// Empty `code_hex` (all zeros) is the smoking gun for "BIOS didn't install
    /// this handler" — the program will execute through a null vector and crash.
    ivt: Vec<IvtEntry>,
    /// BDA fields the program might read. Each entry is (label, value).
    bda: BTreeMap<String, i32>,
    /// Stack contents around SS:SP. 32 bytes starting at SS:SP. Useful for
    /// confirming the return address looks like a real CS:IP, that the
    /// program-environment + cmdline-offset words DOS pushes are sane, etc.
    stack_hex: String,
    /// First 32 bytes at CS:IP — confirms the program code is actually loaded
    /// at the address we think it is, and matches what NASM produced.
    code_at_pc_hex: String,
    /// First 32 bytes at CS:0x100 — for .COM files this is the entry point.
    /// If CS:IP != CS:0x100 it tells us we landed somewhere unexpected.
    code_at_com_entry_hex: String,
    /// PSP fields if this looks like a .COM (CS-segment:0). cmdline length+text.
    /// `null` if the byte at CS:0x80 doesn't look like a plausible cmdline length.
    psp_cmdline: Option<String>,
    /// Plain-English notes about anything that looks wrong.
    /// e.g. "INT 10h handler is null", "DS != CS for a .COM (DOS sets DS=PSP)".
    warnings: Vec<String>,
}

// --- Result types -----------------------------------------------------------

#[derive(Serialize, schemars::JsonSchema)]
struct InfoResult {
    /// True if any session is loaded. Check `sessions` for the full list.
    loaded: bool,
    /// Back-compat: the "default" session's CSS path (if any). Prefer `sessions`.
    css_file: Option<String>,
    /// Back-compat: the "default" session's current tick. Prefer `sessions`.
    current_tick: u32,
    /// Back-compat fields — all refer to the "default" session.
    properties_count: usize,
    functions_count: usize,
    assignments_count: usize,
    snapshots: Vec<u32>,
    /// The current base_interval setting (shared across all loaded programs).
    base_interval: u32,
    /// All loaded sessions, keyed by name. Each entry is a snapshot of that
    /// session's metadata. Independent sessions let multiple agents debug
    /// different cabinets (or the same cabinet at different ticks) without
    /// stepping on each other's state.
    sessions: BTreeMap<String, SessionInfo>,
}

#[derive(Serialize, Clone, schemars::JsonSchema)]
struct SessionInfo {
    css_file: String,
    current_tick: u32,
    properties_count: usize,
    functions_count: usize,
    assignments_count: usize,
    snapshots: Vec<u32>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct OpenParams {
    /// Path to the CSS file to load. Replaces any currently-loaded program
    /// in this session; previous state / snapshots / deltas for this
    /// session are discarded. Other sessions are unaffected.
    path: String,
    /// Session name. Required — pass the same name to every tool call that should share state (open, tick, seek, etc.). If the server restarts (rebuild, crash), call `open` again with the same name to rehydrate.
    session: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct OpenResult {
    css_file: String,
    properties_count: usize,
    functions_count: usize,
    assignments_count: usize,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CloseSessionParams {
    /// Session name to close. Drops all state for this session: snapshots,
    /// deltas, the loaded program. Cancels any in-flight run_until job.
    /// Other sessions are unaffected. Calling close on an unknown name is
    /// not an error — returns `existed: false`.
    session: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct CloseSessionResult {
    session: String,
    existed: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
struct StateResult {
    tick: u32,
    registers: BTreeMap<String, i32>,
    properties: HashMap<String, i64>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct TickResult {
    tick: u32,
    ticks_executed: u32,
    /// Per-state-var changes accumulated across all ticks executed this call.
    changes: Vec<(String, String)>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct MemoryResult {
    addr: i32,
    len: usize,
    /// Hex dump (space-separated bytes).
    hex: String,
    bytes: Vec<u8>,
    /// Little-endian u16 view, for convenience.
    words: Vec<u16>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ScreenResult {
    addr: i32,
    width: usize,
    height: usize,
    text: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct DivergenceInfo {
    tick: u32,
    register: String,
    expected: i64,
    actual: i64,
}

#[derive(Serialize, schemars::JsonSchema)]
struct CompareReferenceResult {
    divergences: Vec<DivergenceInfo>,
    ticks_compared: u32,
}

#[derive(Serialize, schemars::JsonSchema)]
struct DiffEntry {
    property: String,
    compiled: i64,
    interpreted: i64,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ComparePathsResult {
    tick: u32,
    register_diffs: Vec<DiffEntry>,
    property_diffs: Vec<DiffEntry>,
    memory_diffs: Vec<DiffEntry>,
    total_diffs: usize,
}

#[derive(Serialize, schemars::JsonSchema)]
struct MemDiffEntry {
    addr: i32,
    expected: u8,
    actual: u8,
}

#[derive(Serialize, schemars::JsonSchema)]
struct CompareStateResult {
    tick: u32,
    register_diffs: Vec<DiffEntry>,
    memory_diffs: Vec<MemDiffEntry>,
    total_diffs: usize,
}

#[derive(Serialize, schemars::JsonSchema)]
struct TracePropertyResult {
    tick: u32,
    property: String,
    compiled_value: i32,
    interpreted_value: Option<i32>,
    trace_entries: usize,
    /// Per-op trace entries as JSON (fields: pc, op, dst_slot, dst_value, inputs, branch_taken, depth).
    trace: serde_json::Value,
}

#[derive(Serialize, schemars::JsonSchema)]
struct SendKeyResult {
    tick: u32,
    value: i32,
    target: String,
    /// True if the keyboard state var existed (only meaningful for target=keyboard).
    ok: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
struct WatchpointResult {
    hit: bool,
    tick: u32,
    addr: i32,
    /// Present if hit.
    old_value: Option<i32>,
    /// Present if hit.
    new_value: Option<i32>,
    /// Present if not hit (the unchanged value).
    value: Option<i32>,
    /// Snapshot of `--memAddr` register at hit time.
    mem_addr: Option<i64>,
    /// Snapshot of `--memVal` register at hit time.
    mem_val: Option<i64>,
    /// 24 bytes of context centered on `addr`. Empty if not hit.
    context_hex: String,
    /// Registers at hit time. Empty if not hit.
    registers: BTreeMap<String, i32>,
}

#[derive(Serialize, Clone, schemars::JsonSchema)]
struct RunUntilResult {
    /// The condition matched within the run.
    hit: bool,
    /// The run finished — either because it hit, exhausted max_ticks, was
    /// cancelled, or errored. When `done` is false, poll with run_until_poll
    /// using the `job_id`. When `done` is true with `hit=false`, the run
    /// exhausted max_ticks (or was cancelled).
    done: bool,
    /// Present when `done` is false — pass to run_until_poll / run_until_cancel.
    job_id: Option<u64>,
    tick: u32,
    ticks_run: u32,
    cs: Option<i32>,
    ip: Option<i32>,
    opcode: Option<i32>,
    next_byte: Option<i32>,
    ah: Option<i32>,
    /// Free-form description of which condition matched. Present only if hit.
    matched: Option<serde_json::Value>,
    registers: BTreeMap<String, i32>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct DumpOpsResult {
    /// Each entry is one decoded op as a string.
    ops: Vec<String>,
    /// Echoed back so the caller knows which mode was used.
    property: Option<String>,
    start: Option<usize>,
    end: Option<usize>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct SlotMapResult {
    slots: Vec<SlotEntry>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct SlotEntry {
    slot: u32,
    name: String,
    /// None if the slot is out of range.
    value: Option<i32>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct SnapshotResult {
    action: String,
    /// All snapshot ticks currently held.
    snapshots: Vec<u32>,
    /// If action=create, the tick that was just snapshotted (or already existed).
    created: Option<u32>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct SummaryBlock {
    start_tick: u32,
    end_tick: u32,
    cs: i32,
    ip_min: i32,
    ip_max: i32,
    total_ticks: u32,
    event_count: u32,
    /// If ≥60% of the block sat at one PC: `[cs, ip, hits]`. Else omitted.
    dominant_pc: Option<Vec<i32>>,
    /// Interrupts observed: map of opcode/vector (as string) → count.
    /// `"-1"` = block started already mid-IRQ. `"205"` = software INT (0xCD).
    interrupts: BTreeMap<String, u32>,
    /// Coarse memory write regions: each entry is `[lo, hi, count]`.
    /// `lo`/`hi` are 256-byte bucket starts; `hi+0xFF` is the inclusive end.
    write_regions: Vec<Vec<i32>>,
    /// Pre-rendered one-liner.
    line: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct SummaryResult {
    action: String,
    /// True once a recorder is attached and `action` was a control verb
    /// (start/stop/clear). Always true for `get`.
    active: bool,
    /// True if the recorder hit its `max_events` cap.
    truncated: bool,
    /// Number of raw events in the log (post REP-collapse).
    event_count: u32,
    /// Blocks produced by the segmenter (only populated for `get`).
    blocks: Vec<SummaryBlock>,
    /// Full prose rendering, one block per line.
    prose: String,
}

// ---------------------------------------------------------------------------
// DebugSession — the actual debugger state, behind a Mutex inside the handler.
// Identical mechanics to the HTTP version; only the transport changed.
// ---------------------------------------------------------------------------

/// Delta-snapshot model:
///
/// - `bases`: full State clones at tick 0 and every `base_interval` ticks.
///   Memory cost: O(state_size) per base. With the default 50,000-tick
///   interval, a multi-MB State means ~MB per 50K ticks — acceptable for
///   long-running sessions and much less than the HTTP version's 1K-tick
///   clones.
/// - `deltas`: one per tick that has actually been executed, indexed by
///   frame_counter_before. Each delta stores old+new for every changed
///   channel — so we can apply forward (seek into the future) OR backward
///   (reverse seek).
///
/// Seeking from tick A to target T:
///  1. Find the nearest base b with b.tick ≤ T. If the current tick already
///     lies in [b.tick, T] and we're going forward, no restore needed.
///  2. Otherwise restore from the base (O(state_clone)).
///  3. Apply forward deltas from current tick up to T (O(T - current)).
///  4. Reverse seek (current > T) with no better base: revert deltas backward
///     from current down to T without touching the base.
struct DebugSession {
    evaluator: Evaluator,
    state: State,
    /// Full-state checkpoints. Always contains (0, initial_state).
    bases: Vec<(u32, State)>,
    /// One entry per executed tick, in order. `deltas[i].frame_counter_before`
    /// is the tick number when that delta was produced.
    deltas: Vec<calcite_core::StateDelta>,
    /// Ticks between base checkpoints. 0 = only keep the initial base.
    base_interval: u32,
    properties_count: usize,
    functions_count: usize,
    assignments_count: usize,
    css_file: String,
    property_names: Vec<String>,
    /// Optional per-tick event recorder. When `Some`, `step_one` feeds it
    /// after each tick. Blocks/prose come from `calcite_core::summary`.
    summary: Option<calcite_core::summary::EventLogger>,
}

impl DebugSession {
    fn current_tick(&self) -> u32 {
        self.state.frame_counter
    }

    fn registers_map(&self) -> BTreeMap<String, i32> {
        let mut map = BTreeMap::new();
        for (i, name) in self.state.state_var_names.iter().enumerate() {
            if i < self.state.state_vars.len() {
                map.insert(name.clone(), self.state.state_vars[i]);
            }
        }
        map
    }

    fn properties_map(&self) -> HashMap<String, i64> {
        let mut props = HashMap::new();
        for name in &self.property_names {
            if let Some(val) = self.evaluator.get_slot_value(name) {
                props.insert(name.clone(), val as i64);
            }
        }
        props
    }

    /// Execute a single tick: run the evaluator, record the delta, and drop a
    /// base checkpoint if we've crossed a base_interval boundary.
    ///
    /// This is the ONLY write path for `deltas` — every other tick-advancing
    /// operation (tick_n, seek forward, watchpoint, run_until) goes through
    /// here. Keeps the invariant "deltas covers every executed tick" trivially
    /// true.
    fn step_one(&mut self) -> calcite_core::eval::TickResult {
        let tick_before = self.current_tick();
        let (result, delta) = self.evaluator.tick_with_delta(&mut self.state);
        // deltas is indexed by frame_counter_before = position in the vec
        // relative to the first-ever executed tick. We simply append.
        // Re-execution after a seek can overwrite at [tick_before as usize]
        // if we already hold a delta there (they must be equivalent, but keep
        // the freshly-produced one defensively).
        if (tick_before as usize) < self.deltas.len() {
            self.deltas[tick_before as usize] = delta;
        } else {
            self.deltas.push(delta);
        }
        // Feed the summary recorder if enabled. Done here so every tick path
        // (tick_n, watchpoint, run_until) is covered — everything funnels
        // through step_one.
        if let Some(ref mut log) = self.summary {
            log.record_tick(&self.state);
        }
        // Base checkpoint on crossing boundaries. Tick 0 is always in `bases`.
        if self.base_interval > 0 {
            let t = self.current_tick();
            if t > 0 && t % self.base_interval == 0
                && !self.bases.iter().any(|(bt, _)| *bt == t)
            {
                self.bases.push((t, self.state.clone()));
            }
        }
        result
    }

    /// Bulk-advance variant: same as `step_one` but skips the per-tick delta
    /// computation. Used by `run_until_chunk`, where the per-tick change log
    /// is never read — only the final landing tick matters. Skipping the
    /// delta avoids cloning state.memory + state.extended every tick, which
    /// dominated runtime on large cabinets (370 MB+ CSS, 50 MB+ memory image
    /// → ~50 MB clone per tick → 250x slowdown vs. calcite-cli).
    ///
    /// Trade-off: after a `run_until`, the deltas vec has a gap from the
    /// pre-run tick to the landing tick. `seek` backward into that gap will
    /// fall through to base-snapshot replay rather than delta-revert. The
    /// summary recorder is also skipped (it is only meaningful for
    /// step-by-step inspection anyway).
    fn step_one_no_log(&mut self) {
        self.evaluator.tick(&mut self.state);
        // Feed the summary recorder if one is attached. Cheap when None.
        // Used by `summarise` to capture a tactical window of execution
        // without paying the per-tick delta cost.
        if let Some(ref mut log) = self.summary {
            log.record_tick(&self.state);
        }
        if self.base_interval > 0 {
            let t = self.current_tick();
            if t > 0 && t % self.base_interval == 0
                && !self.bases.iter().any(|(bt, _)| *bt == t)
            {
                self.bases.push((t, self.state.clone()));
            }
        }
    }

    fn tick_n(&mut self, count: u32) -> TickResult {
        let start = self.current_tick();
        let mut all_changes = Vec::new();
        for _ in 0..count {
            let r = self.step_one();
            all_changes.extend(r.changes);
        }
        TickResult {
            tick: self.current_tick(),
            ticks_executed: self.current_tick() - start,
            changes: all_changes,
        }
    }

    /// Seek to `target_tick`. Forward: replay deltas or re-execute. Backward:
    /// revert deltas if we have them, else restore from nearest base and
    /// replay forward.
    fn seek(&mut self, target_tick: u32) {
        let current = self.current_tick();
        if current == target_tick {
            return;
        }

        // --- Backward path ---
        if target_tick < current {
            // If we have every delta between target and current, walk back.
            // deltas[i] captures the transition from tick i -> tick i+1, so
            // reverting deltas[target..current] takes us from current to target.
            if (target_tick as usize) < self.deltas.len()
                && (current as usize) <= self.deltas.len()
            {
                // Revert in reverse order (current-1, current-2, ..., target).
                for i in (target_tick as usize..current as usize).rev() {
                    self.state.revert_delta(&self.deltas[i]);
                }
                self.refresh_slots();
                return;
            }
            // Otherwise restore from nearest base ≤ target and replay forward.
            let (base_tick, base_state) = self.nearest_base_le(target_tick);
            self.state = base_state;
            assert_eq!(self.state.frame_counter, base_tick);
            // Forward replay via re-execution. These ticks already have deltas
            // recorded; step_one will overwrite them at the same indices.
            while self.current_tick() < target_tick {
                self.step_one();
            }
            return;
        }

        // --- Forward path ---
        // Prefer delta replay if we have ALL the deltas we'd need, because
        // that avoids re-running the evaluator.
        let have_all_forward = (current as usize..target_tick as usize)
            .all(|i| i < self.deltas.len());
        if have_all_forward {
            for i in current as usize..target_tick as usize {
                // Safety: cloning the delta would work but is wasteful. Pull
                // the reference, apply, move on — state is mutated in place.
                // Re-borrow needed because self.state and self.deltas are both
                // fields of self.
                let delta_ptr = &self.deltas[i] as *const calcite_core::StateDelta;
                // SAFETY: we hold &mut self; deltas is not mutated in this
                // loop (apply_delta only touches self.state).
                unsafe {
                    self.state.apply_delta(&*delta_ptr);
                }
            }
            self.refresh_slots();
            return;
        }

        // Otherwise re-execute. If we'd need to start from a closer base
        // than where we are (e.g., we're way past target and there's no
        // path forward), jump to nearest base.
        if current > target_tick {
            // Unreachable given the backward-path early return above, but
            // kept defensively.
            let (_, base_state) = self.nearest_base_le(target_tick);
            self.state = base_state;
        }
        while self.current_tick() < target_tick {
            self.step_one();
        }
    }

    /// Re-populate the Evaluator's internal slot cache so that
    /// `get_slot_value` / `properties_map` reflect the current state.
    ///
    /// Needed after a delta-only seek (apply_delta / revert_delta) because
    /// those paths don't call `execute()`, which is what normally leaves
    /// `self.slots` populated. Without this, `properties` returned to the
    /// MCP client would be stale from the last executed tick.
    ///
    /// Runs the compiled path on a scratch clone. State-var deltas recorded
    /// by this scratch run are discarded (we only keep the updated slot
    /// cache on the Evaluator); frame_counter doesn't advance on the real
    /// state because scratch is thrown away.
    fn refresh_slots(&mut self) {
        let mut scratch = self.state.clone();
        // `tick` calls `compile::execute` which populates `self.slots`.
        // Pre-tick hooks on `scratch` are fine — they mutate scratch.memory
        // which is discarded.
        self.evaluator.tick(&mut scratch);
    }

    /// Nearest base with tick ≤ target. Always succeeds because (0, initial)
    /// is always in `bases`. Returns an owned State clone.
    fn nearest_base_le(&self, target: u32) -> (u32, State) {
        let mut best: Option<(u32, &State)> = None;
        for (t, s) in &self.bases {
            if *t <= target && (best.is_none() || *t > best.unwrap().0) {
                best = Some((*t, s));
            }
        }
        let (t, s) = best.expect("bases always contains (0, initial_state)");
        (t, s.clone())
    }

    fn read_memory(&self, addr: i32, len: usize) -> MemoryResult {
        let mut bytes = Vec::with_capacity(len);
        for i in 0..len {
            bytes.push(self.state.read_mem(addr + i as i32) as u8);
        }
        let hex = bytes
            .iter()
            .map(|b| format!("{:02X}", b))
            .collect::<Vec<_>>()
            .join(" ");
        let words: Vec<u16> = bytes
            .chunks(2)
            .map(|c| {
                let lo = c[0] as u16;
                let hi = if c.len() > 1 { c[1] as u16 } else { 0 };
                lo | (hi << 8)
            })
            .collect();
        MemoryResult { addr, len, hex, bytes, words }
    }

    fn render_screen(&self, addr: i32, width: usize, height: usize) -> ScreenResult {
        let text = self.state.render_screen(addr as usize, width, height);
        ScreenResult { addr, width, height, text }
    }

    fn watchpoint(&mut self, addr: i32, max_ticks: u32, expected: Option<i32>) -> WatchpointResult {
        let initial = self.state.read_mem(addr);
        let start = self.current_tick();
        let t0 = std::time::Instant::now();

        for _ in 0..max_ticks {
            self.step_one();
            let current = self.state.read_mem(addr);
            let matched = match expected {
                Some(exp) => current == exp,
                None => current != initial,
            };
            if matched {
                let elapsed = t0.elapsed();
                let mem_addr_val = self.state.get_var("memAddr").map(|v| v as i64);
                let mem_val_val = self.state.get_var("memVal").map(|v| v as i64);
                let ctx = self.read_memory(addr - 8, 24);
                eprintln!(
                    "[watchpoint] addr=0x{:x} changed at tick {} ({:.3}s, {} ticks)",
                    addr,
                    self.current_tick(),
                    elapsed.as_secs_f64(),
                    self.current_tick() - start
                );
                return WatchpointResult {
                    hit: true,
                    tick: self.current_tick(),
                    addr,
                    old_value: Some(initial),
                    new_value: Some(current),
                    value: None,
                    mem_addr: mem_addr_val,
                    mem_val: mem_val_val,
                    context_hex: ctx.hex,
                    registers: self.registers_map(),
                };
            }
        }

        let elapsed = t0.elapsed();
        eprintln!(
            "[watchpoint] addr=0x{:x} no change after {} ticks ({:.3}s)",
            addr,
            max_ticks,
            elapsed.as_secs_f64()
        );
        WatchpointResult {
            hit: false,
            tick: self.current_tick(),
            addr,
            old_value: None,
            new_value: None,
            value: Some(self.state.read_mem(addr)),
            mem_addr: None,
            mem_val: None,
            context_hex: String::new(),
            registers: BTreeMap::new(),
        }
    }

    /// Run up to `chunk_ticks` iterations, checking `cond` after each one.
    /// Returns `Some(result)` if a match (or something terminal) was found,
    /// `None` if the chunk completed without matching. Used by the background
    /// run_until driver, which repeatedly calls this with a small chunk size
    /// and releases the session lock between chunks so other MCP tools (poll,
    /// cancel, get_state) can interleave.
    ///
    /// `initial_prop_value` and `start_tick` are carried across chunks by the
    /// caller so "ticks_run" and PropertyChanges' baseline are stable.
    fn run_until_chunk(
        &mut self,
        cond: &RunUntilCondition,
        initial_prop_value: Option<i64>,
        chunk_ticks: u32,
        start_tick: u32,
    ) -> Option<RunUntilResult> {
        for _ in 0..chunk_ticks {
            self.step_one_no_log();

            let cs = self.state.get_var("CS").unwrap_or(0) & 0xFFFF;
            let ip = self.state.get_var("IP").unwrap_or(0) & 0xFFFF;
            let fetch_addr = cs * 16 + ip;
            let op_byte = self.state.read_mem(fetch_addr) & 0xFF;
            let next_byte = self.state.read_mem(fetch_addr + 1) & 0xFF;

            let matched: Option<serde_json::Value> = match cond {
                RunUntilCondition::CsIp { cs: c, ip: i } => {
                    (cs == *c && ip == *i).then(|| serde_json::json!("cs_ip"))
                }
                RunUntilCondition::Cs(c) => (cs == *c).then(|| serde_json::json!("cs")),
                RunUntilCondition::IpRange { cs: c, min, max } => {
                    (cs == *c && ip >= *min && ip <= *max).then(|| serde_json::json!("ip_range"))
                }
                RunUntilCondition::Int(_) => {
                    (op_byte == 0xCD).then(|| serde_json::json!({"int": next_byte}))
                }
                RunUntilCondition::IntNum(n) => (op_byte == 0xCD && next_byte == *n)
                    .then(|| serde_json::json!({"int": next_byte})),
                RunUntilCondition::PropertyEquals { name, value } => {
                    match self.evaluator.get_slot_value(name) {
                        Some(v) if v as i64 == *value => {
                            Some(serde_json::json!({"prop": name, "value": v as i64}))
                        }
                        _ => None,
                    }
                }
                RunUntilCondition::PropertyChanges(name) => {
                    let cur = self.evaluator.get_slot_value(name).map(|v| v as i64);
                    if cur.is_some() && cur != initial_prop_value {
                        Some(serde_json::json!({"prop": name, "from": initial_prop_value, "to": cur}))
                    } else {
                        None
                    }
                }
                RunUntilCondition::MemByteEquals { addr, value } => {
                    ((self.state.read_mem(*addr) & 0xFF) == (*value & 0xFF))
                        .then(|| serde_json::json!({"addr": addr, "value": value}))
                }
                RunUntilCondition::MemRangeNonzero { start, end } => {
                    let mut hit = None;
                    for a in *start..*end {
                        let v = self.state.read_mem(a) & 0xFF;
                        if v != 0 {
                            hit = Some(serde_json::json!({"addr": a, "value": v}));
                            break;
                        }
                    }
                    hit
                }
            };

            if let Some(desc) = matched {
                let ax = self.state.get_var("AX").unwrap_or(0) & 0xFFFF;
                let ah = (ax >> 8) & 0xFF;
                let ticks_run = self.current_tick() - start_tick;
                return Some(RunUntilResult {
                    hit: true,
                    done: true,
                    job_id: None,
                    tick: self.current_tick(),
                    ticks_run,
                    cs: Some(cs),
                    ip: Some(ip),
                    opcode: Some(op_byte),
                    next_byte: Some(next_byte),
                    ah: Some(ah),
                    matched: Some(desc),
                    registers: self.registers_map(),
                });
            }
        }
        None
    }

    fn initial_prop_value(&self, cond: &RunUntilCondition) -> Option<i64> {
        match cond {
            RunUntilCondition::PropertyChanges(name) => {
                self.evaluator.get_slot_value(name).map(|v| v as i64)
            }
            _ => None,
        }
    }

    fn compare_paths(&mut self) -> ComparePathsResult {
        let tick = self.current_tick();
        let saved_state = self.state.clone();

        let mut compiled_state = saved_state.clone();
        self.evaluator.tick(&mut compiled_state);
        let compiled_slots = self.evaluator.get_all_slot_values();

        let mut interpreted_state = saved_state.clone();
        self.evaluator.tick_interpreted(&mut interpreted_state);
        let interpreted_props = self.evaluator.get_all_interpreted_values();

        // Don't advance the real state.
        self.state = saved_state;

        let mut register_diffs = Vec::new();
        let reg_names: Vec<String> = compiled_state.state_var_names.clone();
        for (i, name) in reg_names.iter().enumerate() {
            let c = compiled_state.state_vars.get(i).copied().unwrap_or(0);
            let interp = interpreted_state.state_vars.get(i).copied().unwrap_or(0);
            if c != interp {
                register_diffs.push(DiffEntry {
                    property: name.clone(),
                    compiled: c as i64,
                    interpreted: interp as i64,
                });
            }
        }

        let mut property_diffs = Vec::new();
        let mut all_prop_names: std::collections::BTreeSet<String> =
            compiled_slots.keys().cloned().collect();
        all_prop_names.extend(interpreted_props.keys().cloned());
        for name in &all_prop_names {
            let c = compiled_slots.get(name).copied().unwrap_or(0);
            let interp = interpreted_props.get(name).copied().unwrap_or(0);
            if c != interp {
                property_diffs.push(DiffEntry {
                    property: name.clone(),
                    compiled: c as i64,
                    interpreted: interp as i64,
                });
            }
        }

        let mut memory_diffs = Vec::new();
        let mem_len = compiled_state.memory.len().min(interpreted_state.memory.len());
        for i in 0..mem_len {
            let c = compiled_state.memory[i];
            let interp = interpreted_state.memory[i];
            if c != interp {
                memory_diffs.push(DiffEntry {
                    property: format!("m{}", i),
                    compiled: c as i64,
                    interpreted: interp as i64,
                });
            }
        }

        let total_diffs = register_diffs.len() + property_diffs.len() + memory_diffs.len();
        ComparePathsResult {
            tick,
            register_diffs,
            property_diffs,
            memory_diffs,
            total_diffs,
        }
    }

    fn trace_property(&mut self, property: &str) -> TracePropertyResult {
        let tick = self.current_tick();

        let mut interp_state = self.state.clone();
        self.evaluator.tick_interpreted(&mut interp_state);
        let interp_props = self.evaluator.get_all_interpreted_values();
        let interpreted_value = interp_props.get(property).copied();

        let (compiled_value, trace) = self
            .evaluator
            .trace_property(&self.state, property)
            .unwrap_or((0, Vec::new()));

        TracePropertyResult {
            tick,
            property: property.to_string(),
            compiled_value,
            interpreted_value,
            trace_entries: trace.len(),
            trace: serde_json::to_value(&trace).unwrap_or(serde_json::Value::Null),
        }
    }
}

// ---------------------------------------------------------------------------
// MCP handler — wraps DebugSession in a Mutex so all tools take `&self`.
//
// Mutating tool methods lock the mutex internally. All evaluator work is
// CPU-bound and synchronous; we don't yield across awaits while holding it.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct DebuggerHandler {
    /// Named sessions. The "default" slot is what a tool call without a
    /// `session` param talks to. Clients that want to debug two cabinets at
    /// once (or two agents sharing one MCP server) can `open` into distinct
    /// names and pass `session: "foo"` on every call. Each entry is wrapped
    /// in its own Mutex so two sessions can run independently without
    /// contending for a single lock.
    sessions: Arc<Mutex<HashMap<String, Arc<Mutex<DebugSession>>>>>,
    /// CLI-provided base_interval, used when a new program is loaded via `open`.
    base_interval: u32,
    /// Background run_until jobs, keyed by session name. One job per session.
    run_jobs: Arc<Mutex<HashMap<String, RunJob>>>,
    /// Last-activity timestamp per session (unix millis). Bumped by `get_session`
    /// on every tool-call lookup, and seeded when `install_session` adds a new
    /// entry. Read by the operator REPL's `list` command. Never blocks anyone:
    /// AtomicU64 update is wait-free and the parallel map is touched only
    /// during session insert / lookup / kill.
    last_activity: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    tool_router: ToolRouter<DebuggerHandler>,
}

/// Unix-millis as u64. Saturates on overflow (won't happen for ~584 million years).
fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Format a unix-millis timestamp as `HH:MM:SS` for the event log.
fn format_hms(ms: u64) -> String {
    let secs = ms / 1000;
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

/// In `--listen` (daemon) mode, stdout is the operator console — events go
/// there. In stdio MCP mode, stdout is reserved for MCP frames, so events
/// must go to stderr instead. `main` flips this once at startup.
static EVENTS_TO_STDOUT: AtomicBool = AtomicBool::new(false);

/// Print one event-log line. The format is intentionally plain so an operator
/// can grep / scroll the supervisor window without TUI fuss.
fn log_event(kind: &str, msg: &str) {
    let line = format!("{}  [{kind}] {msg}", format_hms(now_millis()));
    if EVENTS_TO_STDOUT.load(Ordering::Relaxed) {
        println!("{line}");
    } else {
        eprintln!("{line}");
    }
}

/// Shared state between the run_until spawned thread and the MCP handler.
/// The thread owns forward progress; the handler reads it from poll/cancel.
struct RunJob {
    id: u64,
    cancel: Arc<AtomicBool>,
    /// Condition matched / max_ticks exhausted / cancel ⇒ thread writes the
    /// final RunUntilResult here and flips `done`.
    result: Arc<Mutex<Option<RunUntilResult>>>,
    done: Arc<AtomicBool>,
    /// Thread handle. Kept so a drop of the outer job waits for the thread
    /// (which should have noticed `cancel` and exited promptly).
    handle: Option<thread::JoinHandle<()>>,
}

static JOB_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Snapshot CS/IP and every live memory write slot on a state. Used by
/// `diff_packed_memory` to attach CPU context to a divergence report.
///
/// The 6 write slots mirror the CSS-DOS convention: per-tick `--memAddrN`
/// / `--memValN` / `--_slotNLive` triples. Live==0 means the slot was
/// not requested this tick; we include it anyway so the caller can see
/// "no slot fired" vs "slot fired but with zeros".
fn snapshot_cpu_context(state: &calcite_core::State) -> DiffCpuContext {
    let cs = state.get_var("CS").unwrap_or(0);
    let ip = state.get_var("IP").unwrap_or(0);
    let mut writes = Vec::with_capacity(6);
    for slot in 0u8..6 {
        let addr = state.get_var(&format!("memAddr{slot}")).unwrap_or(-1);
        let val = state.get_var(&format!("memVal{slot}")).unwrap_or(0);
        // Slot-live is an underscore-prefixed helper (`--_slotNLive`).
        // `to_bare_name` strips `--__1` etc. but not `_` — the name in
        // state_var_index is the leading-underscore form.
        let live = state.get_var(&format!("_slot{slot}Live")).unwrap_or(0);
        writes.push(DiffWriteSlot { slot, addr, val, live });
    }
    DiffCpuContext { cs, ip, writes }
}

fn invalid_params(msg: impl Into<String>) -> ErrorData {
    ErrorData::invalid_params(msg.into(), None)
}

fn no_program() -> ErrorData {
    ErrorData::invalid_params(
        std::borrow::Cow::Borrowed(
            "no program loaded — call the `open` tool with a CSS file path first",
        ),
        None,
    )
}

/// Read the current tick from a shared session without holding the lock
/// across the whole caller. Used for diagnostics in run_until cancel /
/// session-unload paths.
fn current_tick_locked(session: &Arc<Mutex<DebugSession>>) -> u32 {
    session.lock().unwrap().current_tick()
}

/// Parse + compile a CSS file into a fresh DebugSession. Extracted so `main`
/// and the `open` tool share the same loader path.
fn load_css(path: &std::path::Path, base_interval: u32) -> Result<DebugSession, String> {
    eprintln!("Loading {}...", path.display());
    let css = std::fs::read_to_string(path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;

    let t0 = std::time::Instant::now();
    let parsed =
        calcite_core::parser::parse_css(&css).map_err(|e| format!("parse error: {e}"))?;
    eprintln!(
        "Parsed: {} @property, {} @function, {} assignments ({:.2}s)",
        parsed.properties.len(),
        parsed.functions.len(),
        parsed.assignments.len(),
        t0.elapsed().as_secs_f64(),
    );

    let mut state = State::default();
    state.load_properties(&parsed.properties);
    let (properties_count, functions_count, assignments_count) = (
        parsed.properties.len(),
        parsed.functions.len(),
        parsed.assignments.len(),
    );

    let t1 = std::time::Instant::now();
    let (evaluator, _properties) = Evaluator::from_parsed_owned(parsed);
    eprintln!("Compiled in {:.2}s", t1.elapsed().as_secs_f64());

    // BIOS ROM bytes are emitted as literal branches inside --readMem, not as
    // --mN @property declarations. Copy them into state memory so read_memory
    // / get_state tooling can see ROM contents. Without this, the debugger
    // reports F0000+ as all zeros even though the CPU reads correct bytes.
    if let Some(table) = evaluator.dispatch_tables.get("--readMem") {
        state.populate_memory_from_readmem(table);
    }

    // Packed-cell cabinets (PACK_SIZE > 1) store guest memory in `mcN`
    // state-var cells, not in `state.memory[]`. `State::read_mem` has a
    // cell-lookup code path, but it's gated on `state.packed_cell_table`
    // being populated — the compiled program has the data, but nothing
    // was copying it into State. Use the shared helper so every consumer
    // (debugger, wasm, CLI, tests) stays in sync.
    evaluator.wire_state_for_packed_memory(&mut state);

    let property_names: Vec<String> = evaluator
        .assignments
        .iter()
        .map(|a| a.property.clone())
        .collect();

    Ok(DebugSession {
        evaluator,
        state: state.clone(),
        bases: vec![(0u32, state)],
        deltas: Vec::new(),
        base_interval,
        properties_count,
        functions_count,
        assignments_count,
        css_file: path.display().to_string(),
        property_names,
        summary: None,
    })
}

impl DebuggerHandler {
    fn new(
        session_name: Option<&str>,
        session: Option<DebugSession>,
        base_interval: u32,
    ) -> Self {
        let mut sessions = HashMap::new();
        if let (Some(name), Some(s)) = (session_name, session) {
            sessions.insert(name.to_string(), Arc::new(Mutex::new(s)));
        }
        Self {
            sessions: Arc::new(Mutex::new(sessions)),
            base_interval,
            run_jobs: Arc::new(Mutex::new(HashMap::new())),
            last_activity: Arc::new(Mutex::new(HashMap::new())),
            tool_router: Self::tool_router(),
        }
    }

    /// Resolve a session name to its mutex, or return NO_PROGRAM_LOADED if
    /// the slot is empty. Bumps last-activity for the operator REPL's `list`.
    fn get_session(&self, name: &str) -> Result<Arc<Mutex<DebugSession>>, ErrorData> {
        let arc = {
            let sessions = self.sessions.lock().unwrap();
            sessions.get(name).cloned()
        };
        match arc {
            Some(s) => {
                if let Some(stamp) = self.last_activity.lock().unwrap().get(name) {
                    stamp.store(now_millis(), Ordering::Relaxed);
                }
                Ok(s)
            }
            None => Err(ErrorData::invalid_params(
                format!("no program loaded in session '{name}' — call the `open` tool with a CSS file path first"),
                None,
            )),
        }
    }

    fn install_session(&self, name: &str, s: DebugSession) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(name.to_string(), Arc::new(Mutex::new(s)));
        let mut activity = self.last_activity.lock().unwrap();
        activity.insert(name.to_string(), Arc::new(AtomicU64::new(now_millis())));
    }

    /// Drop a session and any associated run_until job. Returns `true` if the
    /// session existed and was removed, `false` otherwise. Used by the
    /// `close_session` MCP tool and the operator REPL's `kill` command.
    fn drop_session(&self, name: &str) -> bool {
        // Cancel any outstanding run job first so its worker thread notices
        // and exits cleanly (it holds an Arc to the session, so the session
        // mutex won't actually drop until the thread releases it).
        {
            let mut jobs = self.run_jobs.lock().unwrap();
            if let Some(job) = jobs.remove(name) {
                if !job.done.load(Ordering::Acquire) {
                    job.cancel.store(true, Ordering::Release);
                }
            }
        }
        let removed = self.sessions.lock().unwrap().remove(name).is_some();
        self.last_activity.lock().unwrap().remove(name);
        removed
    }

    /// One-line summary of every loaded session, as `(name, summary)` pairs.
    /// Cheap — does not lock individual session mutexes for long.
    fn list_sessions(&self) -> Vec<(String, String)> {
        let entries: Vec<(String, Arc<Mutex<DebugSession>>, Option<u64>)> = {
            let sessions = self.sessions.lock().unwrap();
            let activity = self.last_activity.lock().unwrap();
            sessions
                .iter()
                .map(|(k, v)| {
                    let stamp = activity.get(k).map(|a| a.load(Ordering::Relaxed));
                    (k.clone(), Arc::clone(v), stamp)
                })
                .collect()
        };
        let now = now_millis();
        let mut out = Vec::with_capacity(entries.len());
        for (name, arc, stamp) in entries {
            // Try to lock briefly. If contended (a long-running tool holds
            // it), skip the per-session detail rather than block the REPL.
            let detail = match arc.try_lock() {
                Ok(g) => format!(
                    "tick={:<8} bases={:<3} css={}",
                    g.current_tick(),
                    g.bases.len(),
                    short_path(&g.css_file),
                ),
                Err(_) => "(busy: holding session lock — long-running call in progress)".to_string(),
            };
            let idle = stamp
                .map(|t| {
                    let secs = now.saturating_sub(t) / 1000;
                    if secs < 60 {
                        format!("{secs}s")
                    } else if secs < 3600 {
                        format!("{}m{}s", secs / 60, secs % 60)
                    } else {
                        format!("{}h{}m", secs / 3600, (secs / 60) % 60)
                    }
                })
                .unwrap_or_else(|| "-".to_string());
            out.push((name, format!("idle={:<8} {}", idle, detail)));
        }
        out
    }

    /// Detailed multi-line view of one session for the operator REPL's `info`.
    fn session_info_lines(&self, name: &str) -> Option<Vec<String>> {
        let arc = self.sessions.lock().unwrap().get(name).cloned()?;
        let stamp = self
            .last_activity
            .lock()
            .unwrap()
            .get(name)
            .map(|a| a.load(Ordering::Relaxed));
        let now = now_millis();
        let mut lines = vec![format!("session: {name}")];
        match arc.try_lock() {
            Ok(g) => {
                lines.push(format!("  css:           {}", g.css_file));
                lines.push(format!("  current_tick:  {}", g.current_tick()));
                lines.push(format!("  snapshots:     {} ({})", g.bases.len(),
                    g.bases.iter().map(|(t, _)| t.to_string()).collect::<Vec<_>>().join(", ")));
                lines.push(format!("  properties:    {}", g.properties_count));
                lines.push(format!("  functions:     {}", g.functions_count));
                lines.push(format!("  assignments:   {}", g.assignments_count));
            }
            Err(_) => {
                lines.push("  (busy: long-running tool call holding session lock)".into());
            }
        }
        if let Some(t) = stamp {
            let secs = now.saturating_sub(t) / 1000;
            lines.push(format!("  last_activity: {secs}s ago"));
        }
        Some(lines)
    }
}

/// Trim leading directory components for compact `list` output, keeping the
/// last two path segments so different cabinets in the same folder remain
/// distinguishable. Best-effort — falls back to the full string.
fn short_path(p: &str) -> String {
    let parts: Vec<&str> = p.rsplit(['/', '\\']).take(2).collect();
    if parts.is_empty() {
        p.to_string()
    } else {
        parts.into_iter().rev().collect::<Vec<_>>().join("/")
    }
}

#[tool_router]
impl DebuggerHandler {
    #[tool(
        description = "Session metadata. Also works when no program is loaded — returns {loaded: false} so the client knows to call `open` first."
    )]
    fn info(&self, _params: Parameters<EmptyParams>) -> Result<Json<InfoResult>, ErrorData> {
        let base_interval = self.base_interval;

        // Clone the Arc list first so we don't hold the outer lock while
        // locking each session (which could deadlock if another op has the
        // sessions map + a session lock in the opposite order).
        let entries: Vec<(String, Arc<Mutex<DebugSession>>)> = {
            let sessions = self.sessions.lock().unwrap();
            sessions.iter().map(|(k, v)| (k.clone(), Arc::clone(v))).collect()
        };

        let mut session_infos = BTreeMap::new();
        for (name, arc) in &entries {
            let g = arc.lock().unwrap();
            session_infos.insert(name.clone(), SessionInfo {
                css_file: g.css_file.clone(),
                current_tick: g.current_tick(),
                properties_count: g.properties_count,
                functions_count: g.functions_count,
                assignments_count: g.assignments_count,
                snapshots: g.bases.iter().map(|(t, _)| *t).collect(),
            });
        }

        // Sessions are all listed in `sessions`; the top-level back-compat
        // fields are left empty now that there is no implicit "default"
        // session to mirror.
        Ok(Json(InfoResult {
            loaded: !session_infos.is_empty(),
            css_file: None,
            current_tick: 0,
            properties_count: 0,
            functions_count: 0,
            assignments_count: 0,
            snapshots: vec![],
            base_interval,
            sessions: session_infos,
        }))
    }

    #[tool(
        description = "Load a CSS file into the debugger. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Replaces any currently-loaded program. Required before any other tool (except `info`) can run if the server was started without -i."
    )]
    fn open(
        &self,
        Parameters(p): Parameters<OpenParams>,
    ) -> Result<Json<OpenResult>, ErrorData> {
        let path = std::path::PathBuf::from(&p.path);
        let new_session = load_css(&path, self.base_interval).map_err(invalid_params)?;
        let result = OpenResult {
            css_file: new_session.css_file.clone(),
            properties_count: new_session.properties_count,
            functions_count: new_session.functions_count,
            assignments_count: new_session.assignments_count,
        };
        // If there's a still-running run_until job on this session, cancel it
        // before replacing the session out from under it. The worker thread
        // will notice, write a cancellation result, and exit.
        let session_key = p.session.clone();
        {
            let jobs = self.run_jobs.lock().unwrap();
            if let Some(existing) = jobs.get(&session_key) {
                if !existing.done.load(Ordering::Acquire) {
                    existing.cancel.store(true, Ordering::Release);
                }
            }
        }
        self.install_session(&session_key, new_session);
        log_event(
            "open",
            &format!(
                "session={session_key}  css={}  ({} props, {} assignments)",
                short_path(&result.css_file),
                result.properties_count,
                result.assignments_count,
            ),
        );
        Ok(Json(result))
    }

    #[tool(
        description = "Drop a session and free its memory. Cancels any in-flight run_until job, removes snapshots and deltas. Other sessions are unaffected. Calling on an unknown session is not an error — returns existed=false."
    )]
    fn close_session(
        &self,
        Parameters(p): Parameters<CloseSessionParams>,
    ) -> Result<Json<CloseSessionResult>, ErrorData> {
        let existed = self.drop_session(&p.session);
        if existed {
            log_event("close", &format!("session={}", p.session));
        }
        Ok(Json(CloseSessionResult {
            session: p.session,
            existed,
        }))
    }

    #[tool(
        description = "Current registers (state vars) and computed property values at the current tick. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc."
    )]
    fn get_state(
        &self,
        Parameters(p): Parameters<SessionOnlyParams>,
    ) -> Result<Json<StateResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let guard = sess.lock().unwrap();
        let s = &*guard;
        Ok(Json(StateResult {
            tick: s.current_tick(),
            registers: s.registers_map(),
            properties: s.properties_map(),
        }))
    }

    #[tool(
        description = "Advance the simulation by N ticks, returning the per-tick state-var change log. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Intended for step-by-step debugging (count=1) or small bursts (up to a few hundred). For bulk advancement to a specific tick, use `seek` — it replays from the nearest snapshot and returns only the final state. For \"run until something happens\", use `run_until`."
    )]
    fn tick(&self, Parameters(p): Parameters<TickParams>) -> Result<Json<TickResult>, ErrorData> {
        // Guard rail: `tick` returns every state-var change across every tick
        // executed. At ~5 changes per tick that's thousands of tuples for even
        // a modest count, which blows past the MCP client's result-size cap
        // and is rarely what the caller actually wants. If they really do
        // want N-thousand ticks of change log, splitting the request makes
        // that explicit. Pointed error so callers can route themselves:
        // - bulk advance → `seek`
        // - run until some condition → `run_until`
        const TICK_CHANGELOG_LIMIT: u32 = 500;
        if p.count > TICK_CHANGELOG_LIMIT {
            return Err(invalid_params(format!(
                "tick(count={}): {} ticks is too many for the per-tick change log — the response would likely exceed the MCP result-size limit. \
                 Use `seek(tick: {})` to jump to a specific tick without the change log, or `run_until(...)` to run until a condition matches. \
                 If you genuinely need the change log for a big range, issue multiple `tick` calls each with count <= {}.",
                p.count,
                p.count,
                self.get_session(&p.session)
                    .and_then(|sess| Ok(sess.lock().unwrap().current_tick() + p.count))
                    .unwrap_or(p.count),
                TICK_CHANGELOG_LIMIT,
            )));
        }
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        let t0 = std::time::Instant::now();
        let resp = s.tick_n(p.count);
        let elapsed = t0.elapsed();
        eprintln!(
            "[tick] {} -> {} ({} ticks, {:.3}s)",
            resp.tick - resp.ticks_executed,
            resp.tick,
            resp.ticks_executed,
            elapsed.as_secs_f64()
        );
        Ok(Json(resp))
    }

    #[tool(
        description = "Seek to a specific tick. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Restores the nearest snapshot at or before the target, then replays forward. Going backward also works (seeks via snapshot then replays)."
    )]
    fn seek(&self, Parameters(p): Parameters<SeekParams>) -> Result<Json<StateResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        let from = s.current_tick();
        let t0 = std::time::Instant::now();
        s.seek(p.tick);
        let elapsed = t0.elapsed();
        eprintln!(
            "[seek] {} -> {} ({:.3}s)",
            from,
            s.current_tick(),
            elapsed.as_secs_f64()
        );
        Ok(Json(StateResult {
            tick: s.current_tick(),
            registers: s.registers_map(),
            properties: s.properties_map(),
        }))
    }

    #[tool(description = "Read a range of bytes from 8086 memory. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Returns hex, raw bytes, and LE-u16 view.")]
    fn read_memory(
        &self,
        Parameters(p): Parameters<MemoryParams>,
    ) -> Result<Json<MemoryResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let guard = sess.lock().unwrap();
        let s = &*guard;
        Ok(Json(s.read_memory(p.addr, p.len)))
    }

    #[tool(
        description = "Render a region of video memory as text (each cell is one character byte; attribute bytes ignored). Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Defaults to detected video region or 0xB8000 80x25."
    )]
    fn render_screen(
        &self,
        Parameters(p): Parameters<ScreenParams>,
    ) -> Result<Json<ScreenResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let guard = sess.lock().unwrap();
        let s = &*guard;
        let (addr, w, h) = (
            p.addr.unwrap_or(0xB8000),
            p.width.unwrap_or(80),
            p.height.unwrap_or(25),
        );
        Ok(Json(s.render_screen(addr, w, h)))
    }

    #[tool(
        description = "Compare execution against a reference trace. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Each ref entry is {tick, registers: {NAME: value, ...}}. Stops at first divergence by default."
    )]
    fn compare_reference(
        &self,
        Parameters(p): Parameters<CompareReferenceParams>,
    ) -> Result<Json<CompareReferenceResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        s.seek(0);
        let mut divergences = Vec::new();
        let ticks_to_compare = p.reference.len() as u32;
        for ref_tick in &p.reference {
            while s.current_tick() < ref_tick.tick {
                s.tick_n(1);
            }
            if s.current_tick() == ref_tick.tick {
                s.tick_n(1);
            }
            for (name, expected) in &ref_tick.registers {
                let actual = s.state.get_var(name).unwrap_or(0) as i64;
                if actual != *expected {
                    divergences.push(DivergenceInfo {
                        tick: ref_tick.tick,
                        register: name.clone(),
                        expected: *expected,
                        actual,
                    });
                }
            }
            if p.stop_at_first && !divergences.is_empty() {
                break;
            }
        }
        Ok(Json(CompareReferenceResult {
            divergences,
            ticks_compared: ticks_to_compare,
        }))
    }

    #[tool(
        description = "Run one tick via both the compiled and interpreted paths and diff the results. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Does not advance the session. Use to find compiler bugs."
    )]
    fn compare_paths(
        &self,
        Parameters(p): Parameters<SessionOnlyParams>,
    ) -> Result<Json<ComparePathsResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        Ok(Json(s.compare_paths()))
    }

    #[tool(description = "Diff the current state against a supplied register/memory snapshot. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc.")]
    fn compare_state(
        &self,
        Parameters(p): Parameters<CompareStateParams>,
    ) -> Result<Json<CompareStateResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let guard = sess.lock().unwrap();
        let s = &*guard;
        let mut register_diffs = Vec::new();
        let mut memory_diffs = Vec::new();

        if let Some(regs) = p.registers {
            for (name, expected) in regs {
                let actual = s.state.get_var(&name).unwrap_or(0) as i64;
                if actual != expected {
                    register_diffs.push(DiffEntry {
                        property: name.clone(),
                        compiled: actual,
                        interpreted: expected,
                    });
                }
            }
        }

        if let Some(ranges) = p.memory {
            for range in ranges {
                for i in 0..range.len {
                    if i >= range.bytes.len() {
                        break;
                    }
                    let addr = range.addr + i as i32;
                    let actual = s.state.read_mem(addr) as u8;
                    let expected = range.bytes[i];
                    if actual != expected {
                        memory_diffs.push(MemDiffEntry { addr, expected, actual });
                    }
                }
            }
        }

        let total_diffs = register_diffs.len() + memory_diffs.len();
        Ok(Json(CompareStateResult {
            tick: s.current_tick(),
            register_diffs,
            memory_diffs,
            total_diffs,
        }))
    }

    #[tool(
        description = "Step two sessions in lockstep, comparing a byte-level view of memory derived from their respective storage conventions (per-byte `--m{N}` vars in `ref_session`, packed `--mc{N}` cells in `test_session`). Returns the first (tick, linear_addr) where the byte views diverge, or reports convergence if `max_ticks` exhausts. Designed specifically for diagnosing PACK_SIZE=1 vs PACK_SIZE=2 correctness regressions — the two cabinets run the same program, so any byte-level divergence is a packed-memory bug."
    )]
    fn diff_packed_memory(
        &self,
        Parameters(p): Parameters<DiffPackedMemoryParams>,
    ) -> Result<Json<DiffPackedMemoryResult>, ErrorData> {
        let ref_arc = self.get_session(&p.ref_session)?;
        let test_arc = self.get_session(&p.test_session)?;
        if Arc::ptr_eq(&ref_arc, &test_arc) {
            return Err(invalid_params(
                "ref_session and test_session must be distinct".to_string(),
            ));
        }
        // Deadlock guard: always lock in (ref, test) order. Since distinct
        // sessions are distinct Mutexes there's no cross-locking hazard with
        // other tools, but defining a consistent order is still good hygiene.
        let mut ref_guard = ref_arc.lock().unwrap();
        let mut test_guard = test_arc.lock().unwrap();

        let addr_start = p.addr_start.unwrap_or(0);
        let addr_end = p.addr_end.unwrap_or(0xA_0000);
        let pack = p.pack.max(1) as i32;
        if addr_end <= addr_start {
            return Err(invalid_params(
                format!("addr_end ({addr_end}) must be > addr_start ({addr_start})"),
            ));
        }

        // Pre-resolve test-session cell state var slots for every cell that
        // overlaps [addr_start, addr_end). Skipping name lookups in the inner
        // loop turns the per-tick cost from a HashMap probe per byte into a
        // direct array index per byte.
        let first_cell = (addr_start as i64 / pack as i64) as usize;
        let last_cell_inclusive = ((addr_end as i64 - 1) / pack as i64) as usize;
        let mut cell_slot: Vec<Option<usize>> = Vec::with_capacity(last_cell_inclusive + 1 - first_cell);
        for cell_n in first_cell..=last_cell_inclusive {
            let name = format!("mc{cell_n}");
            cell_slot.push(test_guard.state.var_slot(&name));
        }

        // Helper that extracts the test-session byte at linear `addr`.
        // Uses the precomputed `cell_slot` for the enclosing cell.
        let read_test_byte = |test: &DebugSession, addr: i32| -> u8 {
            let addr_usize = addr as i64;
            let cell_n = (addr_usize / pack as i64) as usize;
            let rel = cell_n - first_cell;
            let slot = cell_slot.get(rel).copied().flatten();
            match slot {
                Some(i) => {
                    let cell = test.state.state_vars[i];
                    let off = (addr_usize % pack as i64) as u32;
                    ((cell >> (8 * off)) & 0xFF) as u8
                }
                None => {
                    // No packed cell declared for this address. Fall back to
                    // the plain memory array so addresses outside the packed
                    // set (if any — shouldn't happen for well-formed carts)
                    // still get a value rather than a bogus zero.
                    (test.state.read_mem(addr) & 0xFF) as u8
                }
            }
        };

        let max_ticks = p.max_ticks;
        for _ in 0..max_ticks {
            ref_guard.step_one_no_log();
            test_guard.step_one_no_log();
            // Check tick-aligned after both step. This means on divergence
            // both sessions have executed the SAME number of ticks, making
            // the "first tick of divergence" report symmetric.
            for addr in addr_start..addr_end {
                let ref_byte = (ref_guard.state.read_mem(addr) & 0xFF) as u8;
                let test_byte = read_test_byte(&test_guard, addr);
                if ref_byte != test_byte {
                    let tick = ref_guard.current_tick();
                    let ref_cpu = snapshot_cpu_context(&ref_guard.state);
                    let test_cpu = snapshot_cpu_context(&test_guard.state);
                    return Ok(Json(DiffPackedMemoryResult {
                        converged: false,
                        ticks_stepped: tick,
                        first_diff_addr: Some(addr),
                        ref_byte: Some(ref_byte),
                        test_byte: Some(test_byte),
                        ref_cpu: Some(ref_cpu),
                        test_cpu: Some(test_cpu),
                        summary: format!(
                            "diverged at tick {tick}, addr 0x{addr:X}: ref=0x{ref_byte:02X} test=0x{test_byte:02X}"
                        ),
                    }));
                }
            }
        }
        Ok(Json(DiffPackedMemoryResult {
            converged: true,
            ticks_stepped: max_ticks,
            first_diff_addr: None,
            ref_byte: None,
            test_byte: None,
            ref_cpu: None,
            test_cpu: None,
            summary: format!("converged through {max_ticks} ticks over [0x{addr_start:X}, 0x{addr_end:X})"),
        }))
    }

    // (snapshot_cpu_context is defined below as a free fn — it doesn't need
    // self, and keeping it free-standing avoids extra method plumbing for a
    // helper used only by diff_packed_memory.)

    #[tool(
        description = "Inspect an entry in the compiled packed-cell address table for a PACK_SIZE>1 cabinet. Returns the state-var address stored in `packed_cell_tables[table_id][cell_index]`, the current cell value read through that address, and the value read directly from the state-var by name for cross-check. If `cell_addr == 0` or `cell_value_via_cell_addr != expected_state_var_value`, the packed cell table has a bad entry and LoadPackedByte will return wrong values for any byte in this cell. Also reports any literal-exception entries that would override the packed extract."
    )]
    fn inspect_packed_cell(
        &self,
        Parameters(p): Parameters<InspectPackedCellParams>,
    ) -> Result<Json<InspectPackedCellResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let guard = sess.lock().unwrap();
        let s = &*guard;
        let compiled = s.evaluator.compiled();
        let total_tables = compiled.packed_cell_tables.len();
        let tid = p.table_id.unwrap_or(0) as usize;
        if tid >= total_tables {
            return Err(invalid_params(format!(
                "table_id {tid} out of range (compiled program has {total_tables} packed cell tables)"
            )));
        }
        let table = &compiled.packed_cell_tables[tid];
        let table_len = table.len();
        let cell_index = p.cell_index as usize;
        let cell_addr = if cell_index < table_len { table[cell_index] } else { 0 };
        let cell_value_via_cell_addr = if cell_addr == 0 { 0 } else { s.state.read_mem(cell_addr) };

        // Cross-check: resolve `mcN` via the address map directly. This is the
        // address the cell table SHOULD contain (modulo 0 for "cell not present").
        let cell_name = format!("--mc{}", p.cell_index);
        let expected_state_var_addr = calcite_core::eval::property_to_address(&cell_name);
        let expected_state_var_value = expected_state_var_addr.map(|addr| s.state.read_mem(addr));

        // Cross-check: what does property_to_address see WITHOUT the leading `--`?
        // This is what `get_or_build_packed_cell_table` actually calls.
        let bare_name = format!("mc{}", p.cell_index);
        let property_to_address_without_dashes =
            calcite_core::eval::property_to_address(&bare_name);

        // Cross-check: bare name lookup via state.get_var.
        let state_var_index_value = s.state.get_var(&bare_name).map(|_| 0usize);
        let _ = state_var_index_value; // placeholder (exact index is not exposed)
        let state_var_index_has_bare = s.state.get_var(&bare_name).is_some();
        let state_var_index_value: Option<usize> = if state_var_index_has_bare {
            // Look up by name in state_var_names for a linear scan — only cheap
            // for diagnostics; fine here.
            s.state
                .state_var_names
                .iter()
                .position(|n| n == &bare_name)
        } else {
            None
        };

        // Pack size — read from the first broadcast-write port if available.
        // Packed writes and packed reads must agree on pack size.
        let pack = compiled.packed_broadcast_writes.first().map(|p| p.pack).unwrap_or(2);

        // Literal exceptions at this cell's byte addresses.
        let exc = &compiled.packed_exception_tables[tid];
        let low_key = (cell_index * pack as usize) as i32;
        let high_key = low_key + 1;
        let exception_low = exc.get(low_key);
        let exception_high = exc.get(high_key);

        let summary = if cell_addr == 0 {
            format!(
                "cell_index={} has cell_addr=0 in table {} (len={}). LoadPackedByte will return 0 for bytes {} and {}. Expected state-var addr: {:?}",
                p.cell_index, tid, table_len, low_key, high_key, expected_state_var_addr
            )
        } else if expected_state_var_addr != Some(cell_addr) {
            format!(
                "cell_addr MISMATCH: table has {}, but expected {:?} (cell_name={}). LoadPackedByte is reading the WRONG state var for cells that hash to this index.",
                cell_addr, expected_state_var_addr, cell_name
            )
        } else {
            format!(
                "cell OK: addr={}, value={} (low={}, high={}). Expected addr matches.",
                cell_addr,
                cell_value_via_cell_addr,
                cell_value_via_cell_addr & 0xFF,
                (cell_value_via_cell_addr >> 8) & 0xFF
            )
        };

        Ok(Json(InspectPackedCellResult {
            cell_index: p.cell_index,
            table_len,
            cell_addr,
            cell_value_via_cell_addr,
            expected_state_var_addr,
            expected_state_var_value,
            exception_low,
            exception_high,
            pack,
            total_tables,
            state_var_index_has_bare,
            state_var_index_value,
            property_to_address_without_dashes,
            summary,
        }))
    }

    #[tool(
        description = "Send a key into the machine. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. target=bda (default) pushes into the BIOS Data Area ring buffer; target=keyboard sets the --keyboard CSS state variable."
    )]
    fn send_key(
        &self,
        Parameters(p): Parameters<SendKeyParams>,
    ) -> Result<Json<SendKeyResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        let target = p.target.to_lowercase();
        let ok = match target.as_str() {
            "bda" => {
                s.state.bda_push_key(p.value);
                true
            }
            "keyboard" => s.state.set_var("keyboard", p.value),
            other => {
                return Err(invalid_params(format!(
                    "unknown target '{other}'; expected 'bda' or 'keyboard'"
                )))
            }
        };
        Ok(Json(SendKeyResult {
            tick: s.current_tick(),
            value: p.value,
            target,
            ok,
        }))
    }

    #[tool(
        description = "Trace every compiled op that contributes to a property at the current tick. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Returns the compiled value, interpreted value (for cross-check), and op-by-op trace."
    )]
    fn trace_property(
        &self,
        Parameters(p): Parameters<TracePropertyParams>,
    ) -> Result<Json<TracePropertyResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        Ok(Json(s.trace_property(&p.property)))
    }

    #[tool(
        description = "Run forward until a memory byte changes (or matches an expected value). Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Optionally seek to from_tick first. Stops at max_ticks."
    )]
    fn watchpoint(
        &self,
        Parameters(p): Parameters<WatchpointParams>,
    ) -> Result<Json<WatchpointResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        if let Some(t) = p.from_tick {
            s.seek(t);
        }
        Ok(Json(s.watchpoint(p.addr, p.max_ticks, p.expected)))
    }

    #[tool(
        description = "Run forward until a condition matches. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Conditions: cs_ip, cs, ip_range, int (any INT), int_num (specific INT), property_equals, property_changes, mem_byte_equals. \n\nStructural long-run handling: the run executes on a background thread, checking cancel between chunks. If the condition hits (or max_ticks exhausts) within ~8 seconds wall-clock, the result is returned inline with `done: true`. Otherwise this call returns `done: false, job_id: N` and the run keeps going — poll with `run_until_poll` or stop it with `run_until_cancel`."
    )]
    fn run_until(
        &self,
        Parameters(p): Parameters<RunUntilParams>,
    ) -> Result<Json<RunUntilResult>, ErrorData> {
        let session_key = p.session.clone();

        // Refuse if another job is already in flight for this session. This
        // keeps session-lock interleaving simple: at most one worker thread
        // mutates a given session at a time. Different sessions are
        // independent — two agents with distinct session names can run
        // simultaneously.
        {
            let mut jobs = self.run_jobs.lock().unwrap();
            if let Some(existing) = jobs.get(&session_key) {
                if !existing.done.load(Ordering::Acquire) {
                    return Err(invalid_params(format!(
                        "another run_until job is still in progress for session '{session_key}' (job_id={}). Poll or cancel it first.",
                        existing.id
                    )));
                }
                // Previous job finished. Drop it so we can start a new one.
                // Join its thread if present (should be immediate — it's done).
                if let Some(mut old) = jobs.remove(&session_key) {
                    if let Some(h) = old.handle.take() { let _ = h.join(); }
                }
            }
        }

        let session_arc = self.get_session(&p.session)?;

        // Apply from_tick up front, while we still have exclusive access.
        if let Some(t) = p.from_tick {
            session_arc.lock().unwrap().seek(t);
        }

        let cond = p.condition;
        let max_ticks = p.max_ticks;
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let result_slot: Arc<Mutex<Option<RunUntilResult>>> = Arc::new(Mutex::new(None));
        let id = JOB_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Snapshot start_tick and initial_prop_value now under the lock so the
        // worker can do its initial setup without racing with a caller that
        // seeks between our setup and the worker's first chunk.
        let (start_tick, initial_prop) = {
            let guard = session_arc.lock().unwrap();
            (guard.current_tick(), guard.initial_prop_value(&cond))
        };

        let worker_cancel = Arc::clone(&cancel);
        let worker_done = Arc::clone(&done);
        let worker_result = Arc::clone(&result_slot);
        let worker_session = Arc::clone(&session_arc);
        let worker_cond = cond.clone();
        let worker_key = session_key.clone();
        let t0 = Instant::now();

        let handle = thread::spawn(move || {
            // Chunk size is a trade-off: larger = less lock churn, smaller =
            // faster cancel response and more room for other tools. 2000 ticks
            // is ~5-20ms of work at typical speeds.
            const CHUNK: u32 = 2000;
            let mut total_run: u32 = 0;

            loop {
                if worker_cancel.load(Ordering::Acquire) {
                    let mut slot = worker_result.lock().unwrap();
                    *slot = Some(RunUntilResult {
                        hit: false, done: true, job_id: None,
                        tick: current_tick_locked(&worker_session),
                        ticks_run: total_run,
                        cs: None, ip: None, opcode: None, next_byte: None, ah: None,
                        matched: Some(serde_json::json!("cancelled")),
                        registers: BTreeMap::new(),
                    });
                    worker_done.store(true, Ordering::Release);
                    eprintln!("[run-until] job {id} (session '{worker_key}') cancelled after {total_run} ticks ({:.3}s)", t0.elapsed().as_secs_f64());
                    return;
                }

                let remaining = max_ticks.saturating_sub(total_run);
                if remaining == 0 {
                    let tick = current_tick_locked(&worker_session);
                    let mut slot = worker_result.lock().unwrap();
                    *slot = Some(RunUntilResult {
                        hit: false, done: true, job_id: None,
                        tick, ticks_run: total_run,
                        cs: None, ip: None, opcode: None, next_byte: None, ah: None,
                        matched: None, registers: BTreeMap::new(),
                    });
                    worker_done.store(true, Ordering::Release);
                    eprintln!("[run-until] job {id} (session '{worker_key}') no match after {max_ticks} ticks ({:.3}s)", t0.elapsed().as_secs_f64());
                    return;
                }
                let chunk = remaining.min(CHUNK);

                // Hold the session lock only for the chunk itself, then drop
                // it between iterations so poll/cancel/other tools can land.
                let hit = {
                    let mut guard = worker_session.lock().unwrap();
                    guard.run_until_chunk(&worker_cond, initial_prop, chunk, start_tick)
                };

                if let Some(res) = hit {
                    eprintln!("[run-until] job {id} (session '{worker_key}') hit at tick {} (+{}) ({:.3}s)",
                        res.tick, res.ticks_run, t0.elapsed().as_secs_f64());
                    let mut slot = worker_result.lock().unwrap();
                    *slot = Some(res);
                    worker_done.store(true, Ordering::Release);
                    return;
                }

                total_run += chunk;
                thread::yield_now();
            }
        });

        {
            let mut jobs = self.run_jobs.lock().unwrap();
            jobs.insert(session_key.clone(), RunJob {
                id,
                cancel: Arc::clone(&cancel),
                result: Arc::clone(&result_slot),
                done: Arc::clone(&done),
                handle: Some(handle),
            });
        }

        // Wait up to INLINE_BUDGET for the job to finish. If it does, return
        // the result inline. If not, hand the caller a job_id to poll with.
        const INLINE_BUDGET_MS: u64 = 8000;
        const POLL_INTERVAL_MS: u64 = 20;
        let deadline = Instant::now() + Duration::from_millis(INLINE_BUDGET_MS);
        while Instant::now() < deadline {
            if done.load(Ordering::Acquire) {
                let slot = result_slot.lock().unwrap();
                return Ok(Json(slot.clone().expect("done but no result")));
            }
            thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
        }

        Ok(Json(RunUntilResult {
            hit: false, done: false, job_id: Some(id),
            tick: 0, ticks_run: 0,
            cs: None, ip: None, opcode: None, next_byte: None, ah: None,
            matched: None, registers: BTreeMap::new(),
        }))
    }

    #[tool(
        description = "Poll a long-running run_until job. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Returns {done: false} if still running, or the final result with {done: true} if it finished (hit the condition, exhausted max_ticks, or was cancelled)."
    )]
    fn run_until_poll(
        &self,
        Parameters(p): Parameters<JobIdParams>,
    ) -> Result<Json<RunUntilResult>, ErrorData> {
        let session_key = p.session.as_str();
        let guard = self.run_jobs.lock().unwrap();
        let job = guard.get(session_key)
            .filter(|j| j.id == p.job_id)
            .ok_or_else(|| invalid_params(format!(
                "no run_until job with id {} on session '{session_key}'", p.job_id
            )))?;
        if job.done.load(Ordering::Acquire) {
            let slot = job.result.lock().unwrap();
            return Ok(Json(slot.clone().expect("done but no result")));
        }
        Ok(Json(RunUntilResult {
            hit: false, done: false, job_id: Some(job.id),
            tick: 0, ticks_run: 0,
            cs: None, ip: None, opcode: None, next_byte: None, ah: None,
            matched: None, registers: BTreeMap::new(),
        }))
    }

    #[tool(
        description = "Cancel a long-running run_until job. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Sets a flag that the worker thread checks between tick chunks (≤ ~5ms). After cancellation the run is 'done' with hit=false and matched='cancelled'."
    )]
    fn run_until_cancel(
        &self,
        Parameters(p): Parameters<JobIdParams>,
    ) -> Result<Json<RunUntilCancelResult>, ErrorData> {
        let session_key = p.session.as_str();
        let guard = self.run_jobs.lock().unwrap();
        let job = guard.get(session_key)
            .filter(|j| j.id == p.job_id)
            .ok_or_else(|| invalid_params(format!(
                "no run_until job with id {} on session '{session_key}'", p.job_id
            )))?;
        job.cancel.store(true, Ordering::Release);
        Ok(Json(RunUntilCancelResult { job_id: job.id, cancelled: true }))
    }

    #[tool(
        description = "Dump compiled ops. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Either by index range (start, end) or by property name. Mutually exclusive."
    )]
    fn dump_ops(
        &self,
        Parameters(p): Parameters<DumpOpsParams>,
    ) -> Result<Json<DumpOpsResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let guard = sess.lock().unwrap();
        let s = &*guard;
        match (p.property.as_deref(), p.start, p.end) {
            (Some(prop), None, None) => match s.evaluator.get_ops_for_property(prop) {
                Some(lines) => Ok(Json(DumpOpsResult {
                    ops: lines,
                    property: Some(prop.to_string()),
                    start: None,
                    end: None,
                })),
                None => Err(invalid_params(format!("property '{prop}' not found"))),
            },
            (None, Some(start), Some(end)) => Ok(Json(DumpOpsResult {
                ops: s.evaluator.dump_ops_range(start, end),
                property: None,
                start: Some(start),
                end: Some(end),
            })),
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => Err(invalid_params(
                "pass either property OR (start, end), not both",
            )),
            _ => Err(invalid_params("pass property, or both start and end")),
        }
    }

    #[tool(description = "List every compiled slot with its name and current value. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc.")]
    fn slot_map(
        &self,
        Parameters(p): Parameters<SessionOnlyParams>,
    ) -> Result<Json<SlotMapResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let guard = sess.lock().unwrap();
        let s = &*guard;
        let entries: Vec<SlotEntry> = s
            .evaluator
            .get_slot_map()
            .iter()
            .map(|(slot, name)| SlotEntry {
                slot: *slot,
                name: name.clone(),
                value: s.evaluator.get_slot_by_index(*slot),
            })
            .collect();
        Ok(Json(SlotMapResult { slots: entries }))
    }

    #[tool(
        description = "Snapshot (base) management. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. action='create' forces a full-state checkpoint at the current tick (in addition to the automatic ones at base_interval boundaries); action='list' enumerates existing bases. Deltas are tracked automatically every tick."
    )]
    fn snapshots(
        &self,
        Parameters(p): Parameters<SnapshotParams>,
    ) -> Result<Json<SnapshotResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        let action = p.action.to_lowercase();
        let created = match action.as_str() {
            "create" => {
                let tick = s.current_tick();
                if !s.bases.iter().any(|(t, _)| *t == tick) {
                    let cloned = s.state.clone();
                    s.bases.push((tick, cloned));
                }
                Some(tick)
            }
            "list" => None,
            other => {
                return Err(invalid_params(format!(
                    "unknown action '{other}'; expected 'create' or 'list'"
                )))
            }
        };
        Ok(Json(SnapshotResult {
            action,
            snapshots: s.bases.iter().map(|(t, _)| *t).collect(),
            created,
        }))
    }

    #[tool(
        description = "Execution summary: per-tick CS/IP/opcode log (REP-collapsed) broken into coherent blocks. Requires a `session` name — reuse the same name across `open`, `tick`, `seek`, etc. Usage: action='start' to begin recording (call before stepping), then step with tick/seek/run_until, then action='get' to read blocks. Each block describes one coherent phase (tick range, CS, IP range, interrupts, memory write regions, dominant-PC stall detection). Much more useful than tick-by-tick inspection when you're trying to understand what happened across thousands of ticks."
    )]
    fn execution_summary(
        &self,
        Parameters(p): Parameters<SummaryParams>,
    ) -> Result<Json<SummaryResult>, ErrorData> {
        use calcite_core::summary::{EventLogger, SegmentConfig, SummaryConfig, render_block, segment};
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;
        let action = p.action.to_lowercase();
        match action.as_str() {
            "start" => {
                let config = SummaryConfig {
                    max_events: p.max_events.unwrap_or(500_000),
                    record_writes: p.record_writes.unwrap_or(true),
                };
                s.summary = Some(EventLogger::new(config));
                Ok(Json(SummaryResult {
                    action,
                    active: true,
                    truncated: false,
                    event_count: 0,
                    blocks: Vec::new(),
                    prose: String::new(),
                }))
            }
            "stop" => {
                let active = s.summary.is_some();
                let (event_count, truncated) = s.summary.as_ref()
                    .map(|l| (l.events.len() as u32, l.truncated))
                    .unwrap_or((0, false));
                // Keep the log — caller still needs `get`. Just mark inactive
                // conceptually by dropping it after they fetch results.
                // For simplicity we leave it attached; `clear` discards.
                Ok(Json(SummaryResult {
                    action,
                    active,
                    truncated,
                    event_count,
                    blocks: Vec::new(),
                    prose: String::new(),
                }))
            }
            "clear" => {
                s.summary = None;
                Ok(Json(SummaryResult {
                    action,
                    active: false,
                    truncated: false,
                    event_count: 0,
                    blocks: Vec::new(),
                    prose: String::new(),
                }))
            }
            "get" => {
                let Some(ref log) = s.summary else {
                    return Err(invalid_params(
                        "no summary recording is active — call action='start' first",
                    ));
                };
                let seg_config = SegmentConfig {
                    ip_window: p.ip_window.unwrap_or(256),
                };
                let blocks = segment(&log.events, &seg_config);
                let prose = blocks
                    .iter()
                    .map(render_block)
                    .collect::<Vec<_>>()
                    .join("\n");
                let out_blocks: Vec<SummaryBlock> = blocks
                    .iter()
                    .map(|b| SummaryBlock {
                        start_tick: b.start_tick,
                        end_tick: b.end_tick,
                        cs: b.cs,
                        ip_min: b.ip_min,
                        ip_max: b.ip_max,
                        total_ticks: b.total_ticks,
                        event_count: b.event_count,
                        dominant_pc: b.dominant_pc.map(|(cs, ip, hits)| {
                            vec![cs, ip, hits as i32]
                        }),
                        interrupts: b.interrupts.iter()
                            .map(|(k, v)| (k.to_string(), *v))
                            .collect(),
                        write_regions: b.write_regions.iter()
                            .map(|(lo, hi, n)| vec![*lo, *hi, *n as i32])
                            .collect(),
                        line: render_block(b),
                    })
                    .collect();
                Ok(Json(SummaryResult {
                    action,
                    active: true,
                    truncated: log.truncated,
                    event_count: log.events.len() as u32,
                    blocks: out_blocks,
                    prose,
                }))
            }
            other => Err(invalid_params(format!(
                "unknown action '{other}'; expected 'start', 'stop', 'get', or 'clear'"
            ))),
        }
    }

    #[tool(
        description = "Run forward up to `max_ticks` (hard-capped at 100000) with full event recording, then return the segmented blocks. One-shot tactical view — no start/stop/get state to manage. Use it after navigating cheaply to a known landmark with `run_until`, to see what's actually happening across the next small window. Each block summarises a coherent execution phase (CS / IP range / interrupts / memory writes) so you can ask 'what happened between point A and point B' without wading through millions of ticks."
    )]
    fn summarise(
        &self,
        Parameters(p): Parameters<SummariseParams>,
    ) -> Result<Json<SummaryResult>, ErrorData> {
        use calcite_core::summary::{EventLogger, SegmentConfig, SummaryConfig, render_block, segment};
        if p.max_ticks > SUMMARISE_HARD_CAP {
            return Err(invalid_params(format!(
                "max_ticks={} exceeds the {SUMMARISE_HARD_CAP} hard cap. \
                 summarise is for tactical recording over small windows. \
                 For longer runs, use run_until (no recording) to navigate, \
                 then summarise over a smaller interesting window.",
                p.max_ticks
            )));
        }
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;

        // Stash any prior summary recorder so this call doesn't disturb it.
        let stash = s.summary.take();
        s.summary = Some(EventLogger::new(SummaryConfig {
            // Cap events at one-per-tick worst case; REP-collapse usually
            // gives much fewer. Leave headroom so truncation is rare.
            max_events: (p.max_ticks as usize).max(1) * 2,
            record_writes: true,
        }));

        let start_tick = s.current_tick();
        for _ in 0..p.max_ticks {
            s.step_one_no_log();
            if s.state.get_var("halt").unwrap_or(0) != 0 {
                break;
            }
        }
        let end_tick = s.current_tick();

        let log = s.summary.take().expect("recorder we just installed");
        // Restore any prior recorder so a separate `execution_summary`
        // session in progress isn't clobbered by this tactical call.
        s.summary = stash;

        let seg_config = SegmentConfig {
            ip_window: p.ip_window.unwrap_or(256),
        };
        let blocks = segment(&log.events, &seg_config);
        let prose_lines: Vec<String> = blocks.iter().map(render_block).collect();
        let prose = prose_lines.join("\n");
        let header = format!(
            "[summarise] ticks {}..{} ({} executed, {} events, {} blocks{})",
            start_tick,
            end_tick,
            end_tick - start_tick,
            log.events.len(),
            blocks.len(),
            if log.truncated { ", TRUNCATED" } else { "" },
        );
        let prose = format!("{header}\n{prose}");
        let out_blocks: Vec<SummaryBlock> = blocks
            .iter()
            .map(|b| SummaryBlock {
                start_tick: b.start_tick,
                end_tick: b.end_tick,
                cs: b.cs,
                ip_min: b.ip_min,
                ip_max: b.ip_max,
                total_ticks: b.total_ticks,
                event_count: b.event_count,
                dominant_pc: b.dominant_pc.map(|(cs, ip, hits)| vec![cs, ip, hits as i32]),
                interrupts: b.interrupts.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                write_regions: b.write_regions.iter().map(|(lo, hi, n)| vec![*lo, *hi, *n as i32]).collect(),
                line: render_block(b),
            })
            .collect();
        Ok(Json(SummaryResult {
            action: "summarise".into(),
            active: false,
            truncated: log.truncated,
            event_count: log.events.len() as u32,
            blocks: out_blocks,
            prose,
        }))
    }

    #[tool(
        description = "Run forward until execution lands at a 'program entry' moment (CS leaves the BIOS / IVT region, or matches `target_cs`), then dump everything the program's first instruction will see: regs, IVT entries it might call, BDA fields, stack, code bytes at PC, and PSP. Catches 'BIOS handed the program a broken environment' bugs that lockstep-against-program can't see (because both sides inherit the same broken setup). Also flags obviously-wrong things in `warnings`."
    )]
    fn entry_state_check(
        &self,
        Parameters(p): Parameters<EntryStateCheckParams>,
    ) -> Result<Json<EntryStateCheckResult>, ErrorData> {
        let sess = self.get_session(&p.session)?;
        let mut guard = sess.lock().unwrap();
        let s = &mut *guard;

        let max_ticks = p.max_ticks.unwrap_or(30_000_000);

        // ---- Run until landing condition ----
        let mut reason = "timeout".to_string();
        let mut found = false;
        for _ in 0..max_ticks {
            let cs = s.state.get_var("CS").unwrap_or(0);
            let matched = match p.target_cs {
                Some(tc) => cs == tc,
                None => cs != 0xF000 && cs >= 0x0100,
            };
            if matched {
                reason = "matched".into();
                found = true;
                break;
            }
            if s.state.get_var("halt").unwrap_or(0) != 0 {
                reason = "halted".into();
                break;
            }
            s.step_one_no_log();
        }

        // ---- Snapshot everything ----
        let registers = s.registers_map();
        let cs = registers.get("CS").copied().unwrap_or(0);
        let ip = registers.get("IP").copied().unwrap_or(0);
        let ds = registers.get("DS").copied().unwrap_or(0);
        let ss = registers.get("SS").copied().unwrap_or(0);
        let sp = registers.get("SP").copied().unwrap_or(0);

        let cs_lin = (cs as i64 * 16) as i32;
        let ss_lin = (ss as i64 * 16) as i32;

        // ---- IVT scan for the interrupts programs actually use ----
        let mut warnings: Vec<String> = Vec::new();
        let mut ivt: Vec<IvtEntry> = Vec::new();
        for &n in &[0x08, 0x09, 0x10, 0x13, 0x16, 0x1A, 0x21] {
            let off = s.state.read_mem16(n * 4);
            let seg = s.state.read_mem16(n * 4 + 2);
            let linear = (seg as i64 * 16 + off as i64) as i32;
            let mut bytes = Vec::with_capacity(8);
            for i in 0..8 {
                bytes.push(s.state.read_mem(linear + i) as u8);
            }
            let code_hex: String = bytes.iter().map(|b| format!("{:02X}", b)).collect();
            if seg == 0 && off == 0 {
                warnings.push(format!("IVT[0x{:02X}] is NULL (seg=0 off=0)", n));
            } else if bytes.iter().all(|b| *b == 0) {
                warnings.push(format!(
                    "IVT[0x{:02X}] -> {:04X}:{:04X} but those bytes are all zero (no handler installed there)",
                    n, seg, off
                ));
            }
            ivt.push(IvtEntry {
                int_num: n,
                seg,
                off,
                linear,
                code_hex,
            });
        }

        // ---- BDA fields ----
        let mut bda = BTreeMap::new();
        bda.insert("equipment_word(0x410)".into(), s.state.read_mem16(0x410));
        bda.insert("memory_size_kb(0x413)".into(), s.state.read_mem16(0x413));
        bda.insert("kbd_buf_head(0x41A)".into(), s.state.read_mem16(0x41A));
        bda.insert("kbd_buf_tail(0x41C)".into(), s.state.read_mem16(0x41C));
        bda.insert("video_mode(0x449)".into(), s.state.read_mem(0x449));
        bda.insert("video_cols(0x44A)".into(), s.state.read_mem16(0x44A));
        bda.insert("active_page(0x462)".into(), s.state.read_mem(0x462));
        bda.insert("ticks_lo(0x46C)".into(), s.state.read_mem16(0x46C));
        bda.insert("ticks_hi(0x46E)".into(), s.state.read_mem16(0x46E));

        // ---- Stack: 32 bytes at SS:SP ----
        let stack_hex: String = (0..32)
            .map(|i| format!("{:02X}", s.state.read_mem(ss_lin + sp + i) as u8))
            .collect();

        // ---- Code at CS:IP and CS:0x100 ----
        let code_at_pc_hex: String = (0..32)
            .map(|i| format!("{:02X}", s.state.read_mem(cs_lin + ip + i) as u8))
            .collect();
        let code_at_com_entry_hex: String = (0..32)
            .map(|i| format!("{:02X}", s.state.read_mem(cs_lin + 0x100 + i) as u8))
            .collect();

        // ---- PSP cmdline (CS:0x80 = length, CS:0x81.. = text) ----
        let cmd_len = s.state.read_mem(cs_lin + 0x80) as u8;
        let psp_cmdline = if cmd_len > 0 && cmd_len < 127 {
            let mut s_str = String::new();
            for i in 0..cmd_len as i32 {
                let c = s.state.read_mem(cs_lin + 0x81 + i) as u8;
                if (32..127).contains(&c) {
                    s_str.push(c as char);
                } else {
                    s_str.push('.');
                }
            }
            Some(s_str)
        } else {
            None
        };

        // ---- Sanity warnings ----
        // For a .COM, DOS sets DS=ES=PSP=CS at entry. If we caught this BEFORE
        // fire's `push cs; push cs; pop ds; pop es` runs, DS should equal CS.
        if found && ip == 0x100 && ds != cs {
            warnings.push(format!(
                ".COM convention violated: DS={:04X} but CS={:04X} (DOS should set DS=PSP=CS for a .COM at entry)",
                ds, cs
            ));
        }
        // Stack should not be in the IVT.
        if ss < 0x0050 {
            warnings.push(format!("SS={:04X} is suspiciously low (in IVT/BDA)", ss));
        }
        // SP looks reasonable for a .COM (typically 0xFFFE — top of segment).
        if ip == 0x100 && sp < 0x100 {
            warnings.push(format!(
                "SP={:04X} is suspiciously low for a .COM at entry (DOS typically sets SP=0xFFFE)",
                sp
            ));
        }

        Ok(Json(EntryStateCheckResult {
            found,
            tick: s.current_tick(),
            reason,
            registers,
            ivt,
            bda,
            stack_hex,
            code_at_pc_hex,
            code_at_com_entry_hex,
            psp_cmdline,
            warnings,
        }))
    }
}

#[tool_handler]
impl ServerHandler for DebuggerHandler {
    fn get_info(&self) -> ServerInfo {
        // ServerInfo / ServerCapabilities are #[non_exhaustive] — build via Default+assign.
        let mut info = ServerInfo::default();
        info.instructions = Some(
            "calcite-debugger MCP server. Drives one CSS program through tick-by-tick \
             execution. Use `info` for session state, `tick`/`seek` to navigate, \
             `get_state`/`read_memory`/`render_screen` to inspect, and `compare_paths` / \
             `trace_property` to find compiler/interpreter divergences. \
             You MUST pass `session` on every tool call. The name you choose in `open` \
             is the name all subsequent calls must use. If the server restarts and loses \
             state, `open` again with the same name to rehydrate."
                .into(),
        );
        // Declare tool_list_changed so the client (Claude Desktop, etc.)
        // knows we may push tool-catalog updates and re-fetches when we
        // emit `notifications/tools/list_changed` on every accept(). That
        // notification path is how a client running across debugger
        // rebuilds picks up newly-added tools without restarting itself
        // — the daemon binary changed, but the client keeps the connection
        // and the cached tool list otherwise.
        info.capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_tool_list_changed()
            .build();
        info
    }
}

// ---------------------------------------------------------------------------
// Operator REPL (--listen mode)
// ---------------------------------------------------------------------------
//
// Runs in a dedicated OS thread so blocking reads on stdin don't tangle with
// the Tokio runtime serving MCP clients. Commands operate directly on the
// shared DebuggerHandler — no separate state, no IPC.
//
// Design choices for v1:
//   - No persistent prompt. Events stream to stdout; commands are typed
//     "into the stream" and the response prints below. Avoids needing a
//     TUI library.
//   - Output goes through `println!` so it interleaves cleanly with
//     `log_event` (also stdout). Both use Rust's line-buffered Stdout
//     mutex, so lines won't tear.
//   - Errors from typos are tolerant: unknown command prints `help`-like
//     hint; bad session name prints "no such session".

fn run_operator_repl(handler: DebuggerHandler) {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        // Flush stdout so the prompt-feel is consistent: any pending event
        // log lines hit the screen before we wait on stdin.
        let _ = std::io::stdout().flush();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF on stdin — operator detached (e.g. Ctrl-Z then close).
                // Don't tear the daemon down; just stop reading.
                println!("[repl] stdin closed; operator commands disabled. Daemon keeps serving MCP clients.");
                return;
            }
            Ok(_) => {}
            Err(e) => {
                println!("[repl] stdin read error: {e}");
                return;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let cmd = parts.next().unwrap_or("").to_lowercase();
        let arg = parts.next().map(str::to_string);
        match cmd.as_str() {
            "help" | "h" | "?" => {
                println!("Operator commands:");
                println!("  list, ls            — show all sessions");
                println!("  info <name>         — detailed view of one session");
                println!("  kill <name>         — drop a session and free memory");
                println!("  kill all            — drop every session");
                println!("  quit, exit          — kill all sessions and stop the daemon");
                println!("  help                — this message");
            }
            "list" | "ls" => {
                let entries = handler.list_sessions();
                if entries.is_empty() {
                    println!("(no sessions loaded)");
                } else {
                    for (name, summary) in entries {
                        println!("  {name:<16}  {summary}");
                    }
                }
            }
            "info" => {
                let Some(name) = arg else {
                    println!("usage: info <session>");
                    continue;
                };
                match handler.session_info_lines(&name) {
                    Some(lines) => for l in lines { println!("{l}"); },
                    None => println!("no such session: {name}"),
                }
            }
            "kill" => {
                let Some(target) = arg else {
                    println!("usage: kill <session> | kill all");
                    continue;
                };
                if target == "all" {
                    let names: Vec<String> = handler.list_sessions().into_iter().map(|(n, _)| n).collect();
                    if names.is_empty() {
                        println!("(no sessions to kill)");
                    } else {
                        for n in &names {
                            handler.drop_session(n);
                            log_event("kill", &format!("session={n}  by operator"));
                        }
                        println!("killed {} session(s)", names.len());
                    }
                } else if handler.drop_session(&target) {
                    log_event("kill", &format!("session={target}  by operator"));
                    println!("killed: {target}");
                } else {
                    println!("no such session: {target}");
                }
            }
            "quit" | "exit" => {
                let names: Vec<String> = handler.list_sessions().into_iter().map(|(n, _)| n).collect();
                for n in &names {
                    handler.drop_session(n);
                }
                println!("daemon shutting down ({} session(s) killed)", names.len());
                std::process::exit(0);
            }
            other => {
                println!("unknown command: {other}  (try `help`)");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CRITICAL: stdout is reserved for MCP frames. All logging → stderr.
    env_logger::Builder::from_default_env()
        .target(env_logger::Target::Stderr)
        .init();

    let cli = Cli::parse();

    // `-i` requires `--session` so the pre-loaded program lives under a
    // caller-chosen name — session names are required on every MCP tool call.
    if cli.input.is_some() && cli.session.is_none() {
        return Err(
            "`-i/--input` requires `--session <name>` so the pre-loaded program has a session name"
                .into(),
        );
    }

    // If -i was given, load that file now. If not, start with no program —
    // the client will call the `open` tool to pick one.
    let initial_session = match &cli.input {
        Some(path) => Some(load_css(path, cli.base_interval).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?),
        None => {
            eprintln!("No -i flag; starting without a loaded program. Use the `open` tool to load a CSS file.");
            None
        }
    };

    let handler = DebuggerHandler::new(
        cli.session.as_deref(),
        initial_session,
        cli.base_interval,
    );

    // Two transport modes:
    //   - stdio (default): one client, one process. State dies with the
    //     process — which happens whenever the MCP host tears the
    //     subprocess down (timeout, client restart, rebuild).
    //   - TCP listen (daemon): accept connections in a loop, each served
    //     on its own task sharing the same handler (all handler state
    //     is Arc-wrapped so concurrent tasks coordinate via the session
    //     map and the run-jobs map, not a single top-level lock). The
    //     daemon process outlives individual clients, so sessions
    //     survive reconnects.
    if let Some(addr) = cli.listen.as_deref() {
        use tokio::net::TcpListener;
        // Daemon mode: stdout becomes the operator console. Switch event-log
        // routing before any session activity so [open] etc. don't go to
        // stderr first, then stdout, in mixed order.
        EVENTS_TO_STDOUT.store(true, Ordering::Relaxed);
        let listener = TcpListener::bind(addr).await.map_err(|e| {
            format!("failed to bind listener on {addr}: {e}")
        })?;
        println!("calcite-debugger daemon listening on {addr}");
        println!("Type `help` for operator commands. Events appear below.");
        println!();

        // Operator REPL on stdin. Runs in a blocking thread so stdin reads
        // don't fight with Tokio. Commands are dispatched against the same
        // `handler` that serves MCP clients — they share session state.
        let repl_handler = handler.clone();
        std::thread::spawn(move || run_operator_repl(repl_handler));

        loop {
            let (stream, peer) = match listener.accept().await {
                Ok(pair) => pair,
                Err(e) => {
                    log_event("error", &format!("accept: {e} (continuing)"));
                    continue;
                }
            };
            let handler_clone = handler.clone();
            log_event("conn", &format!("{peer} connected"));
            tokio::spawn(async move {
                match handler_clone.serve(stream).await {
                    Ok(service) => {
                        // Force the client to re-fetch the tool catalog. Without
                        // this, a client that already has a cached catalog from
                        // an earlier connection (typical when the daemon was
                        // rebuilt + restarted but the MCP host kept its
                        // subprocess shim alive) won't see newly-added tools.
                        // notify_tool_list_changed only does anything if the
                        // server declared tool_list_changed in capabilities
                        // (which we did in get_info above).
                        let p = service.peer().clone();
                        let peer_addr = peer.to_string();
                        tokio::spawn(async move {
                            match p.notify_tool_list_changed().await {
                                Ok(_) => log_event("conn", &format!("{peer_addr} notified tool_list_changed")),
                                Err(e) => log_event("conn", &format!("{peer_addr} notify_tool_list_changed failed: {e}")),
                            }
                        });
                        match service.waiting().await {
                            Ok(reason) => log_event("conn", &format!("{peer} closed: {reason:?}")),
                            Err(e) => log_event("conn", &format!("{peer} error: {e}")),
                        }
                    }
                    Err(e) => log_event("conn", &format!("{peer} serve failed: {e}")),
                }
            });
        }
    } else {
        eprintln!("MCP server ready on stdio. Awaiting initialize...");
        let service = handler.serve(stdio()).await?;
        let quit_reason = service.waiting().await?;
        eprintln!("MCP server stopped: {:?}", quit_reason);
    }
    Ok(())
}
