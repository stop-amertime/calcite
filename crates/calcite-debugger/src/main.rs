use calcite_core::state::{self, State};
use calcite_core::Evaluator;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tiny_http::{Header, Method, Response, Server};

/// calcite-debugger — HTTP debug server for CSS execution.
///
/// Parses a CSS file once, then serves an HTTP API for stepping through
/// execution tick by tick, inspecting state, comparing traces, etc.
#[derive(Parser, Debug)]
#[command(name = "calcite-debugger", version, about)]
struct Cli {
    /// Path to the CSS file to debug.
    #[arg(short, long)]
    input: PathBuf,

    /// Port to listen on.
    #[arg(short, long, default_value = "3333")]
    port: u16,

    /// Snapshot interval (ticks between automatic checkpoints).
    #[arg(long, default_value = "1000")]
    snapshot_interval: u32,
}

// ---------------------------------------------------------------------------
// JSON request/response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TickRequest {
    count: Option<u32>,
}

#[derive(Deserialize)]
struct SeekRequest {
    tick: u32,
}

#[derive(Deserialize)]
struct MemoryRequest {
    addr: i32,
    len: Option<usize>,
}

#[derive(Deserialize)]
struct ScreenRequest {
    addr: Option<i32>,
    width: Option<usize>,
    height: Option<usize>,
}

#[derive(Deserialize)]
struct CompareRequest {
    reference: Vec<RefTick>,
    stop_at_first: Option<bool>,
}

#[derive(Deserialize, Clone)]
struct RefTick {
    tick: u32,
    #[serde(flatten)]
    registers: HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct StateResponse {
    tick: u32,
    registers: RegisterState,
    properties: HashMap<String, i64>,
}

#[derive(Serialize)]
struct RegisterState {
    #[serde(rename = "AX")]
    ax: i32,
    #[serde(rename = "CX")]
    cx: i32,
    #[serde(rename = "DX")]
    dx: i32,
    #[serde(rename = "BX")]
    bx: i32,
    #[serde(rename = "SP")]
    sp: i32,
    #[serde(rename = "BP")]
    bp: i32,
    #[serde(rename = "SI")]
    si: i32,
    #[serde(rename = "DI")]
    di: i32,
    #[serde(rename = "IP")]
    ip: i32,
    #[serde(rename = "ES")]
    es: i32,
    #[serde(rename = "CS")]
    cs: i32,
    #[serde(rename = "SS")]
    ss: i32,
    #[serde(rename = "DS")]
    ds: i32,
    #[serde(rename = "FLAGS")]
    flags: i32,
}

#[derive(Serialize)]
struct MemoryResponse {
    addr: i32,
    len: usize,
    /// Hex dump of memory bytes.
    hex: String,
    /// Raw byte values.
    bytes: Vec<u8>,
    /// 16-bit words (little-endian) for convenience.
    words: Vec<u16>,
}

#[derive(Serialize)]
struct ScreenResponse {
    addr: i32,
    width: usize,
    height: usize,
    text: String,
}

#[derive(Serialize)]
struct DivergenceInfo {
    tick: u32,
    register: String,
    expected: i64,
    actual: i64,
}

#[derive(Serialize)]
struct CompareResponse {
    divergences: Vec<DivergenceInfo>,
    ticks_compared: u32,
}

#[derive(Serialize)]
struct DiffEntry {
    property: String,
    compiled: i64,
    interpreted: i64,
}

#[derive(Serialize)]
struct ComparePathsResponse {
    tick: u32,
    register_diffs: Vec<DiffEntry>,
    memory_diffs: Vec<DiffEntry>,
    total_diffs: usize,
}

#[derive(Serialize)]
struct TickResponse {
    tick: u32,
    ticks_executed: u32,
    changes: Vec<(String, String)>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct InfoResponse {
    css_file: String,
    current_tick: u32,
    properties_count: usize,
    functions_count: usize,
    assignments_count: usize,
    snapshots: Vec<u32>,
    endpoints: Vec<&'static str>,
}

// ---------------------------------------------------------------------------
// Debug session — holds all mutable state behind a Mutex
// ---------------------------------------------------------------------------

struct DebugSession {
    evaluator: Evaluator,
    state: State,
    /// Evaluator is immutable between ticks — only State needs snapshots.
    snapshots: Vec<(u32, State)>,
    snapshot_interval: u32,
    /// Property counts from parsed program (for info).
    properties_count: usize,
    functions_count: usize,
    assignments_count: usize,
    css_file: String,
    /// Video memory config if detected.
    video_config: Option<(usize, usize)>,
    /// All property names known to the compiled program.
    property_names: Vec<String>,
    /// The parsed program, kept for creating interpreted-path evaluators.
    parsed_program: calcite_core::types::ParsedProgram,
}

impl DebugSession {
    fn current_tick(&self) -> u32 {
        self.state.frame_counter
    }

    fn registers(&self) -> RegisterState {
        let r = &self.state.registers;
        RegisterState {
            ax: r[state::reg::AX],
            cx: r[state::reg::CX],
            dx: r[state::reg::DX],
            bx: r[state::reg::BX],
            sp: r[state::reg::SP],
            bp: r[state::reg::BP],
            si: r[state::reg::SI],
            di: r[state::reg::DI],
            ip: r[state::reg::IP],
            es: r[state::reg::ES],
            cs: r[state::reg::CS],
            ss: r[state::reg::SS],
            ds: r[state::reg::DS],
            flags: r[state::reg::FLAGS],
        }
    }

    fn get_properties(&self) -> HashMap<String, i64> {
        let mut props = HashMap::new();
        for name in &self.property_names {
            if let Some(val) = self.evaluator.get_slot_value(name) {
                props.insert(name.clone(), val as i64);
            }
        }
        props
    }

    fn tick(&mut self, count: u32) -> TickResponse {
        let start_tick = self.current_tick();
        let mut all_changes = Vec::new();

        for _ in 0..count {
            // Auto-snapshot at interval boundaries
            if self.snapshot_interval > 0
                && self.current_tick() > 0
                && self.current_tick() % self.snapshot_interval == 0
            {
                let tick = self.current_tick();
                if !self.snapshots.iter().any(|(t, _)| *t == tick) {
                    self.snapshots.push((tick, self.state.clone()));
                }
            }
            let result = self.evaluator.tick(&mut self.state);
            all_changes.extend(result.changes);
        }

        TickResponse {
            tick: self.current_tick(),
            ticks_executed: self.current_tick() - start_tick,
            changes: all_changes,
        }
    }

    fn seek(&mut self, target_tick: u32) {
        // Find nearest snapshot at or before target
        let mut best: Option<(u32, &State)> = None;
        for (tick, snap) in &self.snapshots {
            if *tick <= target_tick {
                if best.is_none() || *tick > best.unwrap().0 {
                    best = Some((*tick, snap));
                }
            }
        }

        if let Some((snap_tick, snap_state)) = best {
            if snap_tick > self.current_tick() || self.current_tick() > target_tick {
                // Restore from snapshot
                self.state = snap_state.clone();
            }
        } else if self.current_tick() > target_tick {
            // No useful snapshot — reset to tick 0
            self.state = State::default();
            self.state.load_properties(
                &self.parsed_program.properties,
            );
        }

        // Run forward to target
        while self.current_tick() < target_tick {
            // Auto-snapshot at interval boundaries
            if self.snapshot_interval > 0
                && self.current_tick() > 0
                && self.current_tick() % self.snapshot_interval == 0
            {
                let tick = self.current_tick();
                if !self.snapshots.iter().any(|(t, _)| *t == tick) {
                    self.snapshots.push((tick, self.state.clone()));
                }
            }
            self.evaluator.tick(&mut self.state);
        }
    }

    fn read_memory(&self, addr: i32, len: usize) -> MemoryResponse {
        let mut bytes = Vec::with_capacity(len);
        for i in 0..len {
            bytes.push(self.state.read_mem(addr + i as i32) as u8);
        }
        let hex = bytes.iter().map(|b| format!("{:02X}", b)).collect::<Vec<_>>().join(" ");
        let words: Vec<u16> = bytes
            .chunks(2)
            .map(|c| {
                let lo = c[0] as u16;
                let hi = if c.len() > 1 { c[1] as u16 } else { 0 };
                lo | (hi << 8)
            })
            .collect();
        MemoryResponse { addr, len, hex, bytes, words }
    }

    fn render_screen(&self, addr: i32, width: usize, height: usize) -> ScreenResponse {
        let text = self.state.render_screen(addr as usize, width, height);
        ScreenResponse { addr, width, height, text }
    }

    fn compare_paths(&mut self) -> ComparePathsResponse {
        let tick = self.current_tick();

        // Save current state
        let saved_state = self.state.clone();

        // Run one tick via compiled path (normal)
        let mut compiled_state = saved_state.clone();
        self.evaluator.tick(&mut compiled_state);

        // Run one tick via interpreted path
        let mut interpreted_state = saved_state.clone();
        self.evaluator.tick_interpreted(&mut interpreted_state);

        // Restore original state (don't advance)
        self.state = saved_state;

        // Compare registers
        let reg_names = [
            "AX", "CX", "DX", "BX", "SP", "BP", "SI", "DI", "IP", "ES", "CS", "SS", "DS", "FLAGS",
        ];
        let mut register_diffs = Vec::new();
        for (i, name) in reg_names.iter().enumerate() {
            let c = compiled_state.registers[i];
            let interp = interpreted_state.registers[i];
            if c != interp {
                register_diffs.push(DiffEntry {
                    property: name.to_string(),
                    compiled: c as i64,
                    interpreted: interp as i64,
                });
            }
        }

        // Compare memory (only check where they differ)
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

        let total_diffs = register_diffs.len() + memory_diffs.len();
        ComparePathsResponse {
            tick,
            register_diffs,
            memory_diffs,
            total_diffs,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

fn json_response<T: Serialize>(data: &T) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::to_string_pretty(data).unwrap();
    let len = body.len();
    Response::new(
        tiny_http::StatusCode(200),
        vec![Header::from_bytes("Content-Type", "application/json").unwrap()],
        std::io::Cursor::new(body.into_bytes()),
        Some(len),
        None,
    )
}

fn error_response(status: u16, msg: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::to_string(&ErrorResponse {
        error: msg.to_string(),
    })
    .unwrap();
    let len = body.len();
    Response::new(
        tiny_http::StatusCode(status),
        vec![Header::from_bytes("Content-Type", "application/json").unwrap()],
        std::io::Cursor::new(body.into_bytes()),
        Some(len),
        None,
    )
}

fn read_body(request: &mut tiny_http::Request) -> String {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body).unwrap_or(0);
    body
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Parse CSS
    eprintln!("Loading {}...", cli.input.display());
    let css = match std::fs::read_to_string(&cli.input) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading {}: {e}", cli.input.display());
            std::process::exit(1);
        }
    };

    let t0 = std::time::Instant::now();
    let parsed = match calcite_core::parser::parse_css(&css) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {e}");
            std::process::exit(1);
        }
    };
    let parse_time = t0.elapsed();
    eprintln!(
        "Parsed: {} @property, {} @function, {} assignments ({:.2}s)",
        parsed.properties.len(),
        parsed.functions.len(),
        parsed.assignments.len(),
        parse_time.as_secs_f64(),
    );

    let t1 = std::time::Instant::now();
    let evaluator = Evaluator::from_parsed(&parsed);
    let compile_time = t1.elapsed();
    eprintln!("Compiled in {:.2}s", compile_time.as_secs_f64());

    let mut state = State::default();
    state.load_properties(&parsed.properties);

    let video_config = calcite_core::detect_video_memory();
    if let Some((addr, size)) = video_config {
        eprintln!("Video memory detected at 0x{:X} ({} bytes)", addr, size);
    }

    // Collect all property names from the compiled program
    let property_names: Vec<String> = evaluator
        .assignments
        .iter()
        .map(|a| a.property.clone())
        .collect();

    let properties_count = parsed.properties.len();
    let functions_count = parsed.functions.len();
    let assignments_count = parsed.assignments.len();

    // Save initial state as snapshot 0
    let initial_snapshot = (0, state.clone());

    let session = Mutex::new(DebugSession {
        evaluator,
        state,
        snapshots: vec![initial_snapshot],
        snapshot_interval: cli.snapshot_interval,
        properties_count,
        functions_count,
        assignments_count,
        css_file: cli.input.display().to_string(),
        video_config,
        property_names,
        parsed_program: parsed,
    });

    // Start HTTP server
    let addr = format!("0.0.0.0:{}", cli.port);
    let server = Server::http(&addr).unwrap_or_else(|e| {
        eprintln!("Failed to start server on {}: {e}", addr);
        std::process::exit(1);
    });
    eprintln!("Debug server listening on http://localhost:{}", cli.port);
    eprintln!();
    eprintln!("Endpoints:");
    eprintln!("  GET  /info              — session info and available endpoints");
    eprintln!("  GET  /state             — current registers + computed properties");
    eprintln!("  POST /tick              — advance ticks: {{\"count\": N}}");
    eprintln!("  POST /seek              — seek to tick: {{\"tick\": N}}");
    eprintln!("  POST /memory            — read memory: {{\"addr\": N, \"len\": N}}");
    eprintln!("  POST /screen            — render screen: {{\"addr\": 0xB8000, \"width\": 80, \"height\": 25}}");
    eprintln!("  POST /compare           — compare vs reference: {{\"reference\": [...]}}");
    eprintln!("  GET  /compare-paths     — diff compiled vs interpreted for current tick");
    eprintln!("  POST /snapshot          — create snapshot at current tick");
    eprintln!("  GET  /snapshots         — list all snapshots");

    for mut request in server.incoming_requests() {
        let path = request.url().split('?').next().unwrap_or("").to_string();
        let method = request.method().clone();

        let response = match (method, path.as_str()) {
            (Method::Get, "/info") => {
                let s = session.lock().unwrap();
                json_response(&InfoResponse {
                    css_file: s.css_file.clone(),
                    current_tick: s.current_tick(),
                    properties_count: s.properties_count,
                    functions_count: s.functions_count,
                    assignments_count: s.assignments_count,
                    snapshots: s.snapshots.iter().map(|(t, _)| *t).collect(),
                    endpoints: vec![
                        "GET /info",
                        "GET /state",
                        "POST /tick {count}",
                        "POST /seek {tick}",
                        "POST /memory {addr, len}",
                        "POST /screen {addr, width, height}",
                        "POST /compare {reference}",
                        "GET /compare-paths",
                        "POST /snapshot",
                        "GET /snapshots",
                    ],
                })
            }

            (Method::Get, "/state") => {
                let s = session.lock().unwrap();
                json_response(&StateResponse {
                    tick: s.current_tick(),
                    registers: s.registers(),
                    properties: s.get_properties(),
                })
            }

            (Method::Post, "/tick") => {
                let body = read_body(&mut request);
                let count = if body.is_empty() {
                    1
                } else {
                    match serde_json::from_str::<TickRequest>(&body) {
                        Ok(r) => r.count.unwrap_or(1),
                        Err(e) => {
                            let _ = request.respond(error_response(400, &format!("Bad JSON: {e}")));
                            continue;
                        }
                    }
                };
                let mut s = session.lock().unwrap();
                let t0 = std::time::Instant::now();
                let resp = s.tick(count);
                let elapsed = t0.elapsed();
                eprintln!(
                    "[tick] {} -> {} ({} ticks, {:.3}s)",
                    resp.tick - resp.ticks_executed,
                    resp.tick,
                    resp.ticks_executed,
                    elapsed.as_secs_f64()
                );
                json_response(&resp)
            }

            (Method::Post, "/seek") => {
                let body = read_body(&mut request);
                match serde_json::from_str::<SeekRequest>(&body) {
                    Ok(r) => {
                        let mut s = session.lock().unwrap();
                        let from = s.current_tick();
                        let t0 = std::time::Instant::now();
                        s.seek(r.tick);
                        let elapsed = t0.elapsed();
                        eprintln!(
                            "[seek] {} -> {} ({:.3}s)",
                            from,
                            s.current_tick(),
                            elapsed.as_secs_f64()
                        );
                        json_response(&StateResponse {
                            tick: s.current_tick(),
                            registers: s.registers(),
                            properties: s.get_properties(),
                        })
                    }
                    Err(e) => error_response(400, &format!("Bad JSON: {e}")),
                }
            }

            (Method::Post, "/memory") => {
                let body = read_body(&mut request);
                match serde_json::from_str::<MemoryRequest>(&body) {
                    Ok(r) => {
                        let s = session.lock().unwrap();
                        let len = r.len.unwrap_or(256);
                        json_response(&s.read_memory(r.addr, len))
                    }
                    Err(e) => error_response(400, &format!("Bad JSON: {e}")),
                }
            }

            (Method::Post, "/screen") => {
                let body = read_body(&mut request);
                let (addr, width, height) = if body.is_empty() {
                    // Use detected video config or defaults
                    let s = session.lock().unwrap();
                    if let Some((vaddr, _)) = s.video_config {
                        (vaddr as i32, 80, 25)
                    } else {
                        (0xB8000, 80, 25)
                    }
                } else {
                    match serde_json::from_str::<ScreenRequest>(&body) {
                        Ok(r) => (
                            r.addr.unwrap_or(0xB8000),
                            r.width.unwrap_or(80),
                            r.height.unwrap_or(25),
                        ),
                        Err(e) => {
                            let _ = request.respond(error_response(400, &format!("Bad JSON: {e}")));
                            continue;
                        }
                    }
                };
                let s = session.lock().unwrap();
                json_response(&s.render_screen(addr, width, height))
            }

            (Method::Post, "/compare") => {
                let body = read_body(&mut request);
                match serde_json::from_str::<CompareRequest>(&body) {
                    Ok(r) => {
                        let mut s = session.lock().unwrap();
                        let stop_at_first = r.stop_at_first.unwrap_or(true);

                        // Reset to tick 0
                        s.seek(0);

                        let mut divergences = Vec::new();
                        let reg_keys = [
                            ("AX", state::reg::AX),
                            ("CX", state::reg::CX),
                            ("DX", state::reg::DX),
                            ("BX", state::reg::BX),
                            ("SP", state::reg::SP),
                            ("BP", state::reg::BP),
                            ("SI", state::reg::SI),
                            ("DI", state::reg::DI),
                            ("IP", state::reg::IP),
                            ("ES", state::reg::ES),
                            ("CS", state::reg::CS),
                            ("SS", state::reg::SS),
                            ("DS", state::reg::DS),
                            ("FLAGS", state::reg::FLAGS),
                        ];

                        let ticks_to_compare = r.reference.len() as u32;
                        for ref_tick in &r.reference {
                            // Advance to the right tick
                            while s.current_tick() < ref_tick.tick {
                                s.tick(1);
                            }
                            // Also run the tick AT ref_tick.tick if we're behind
                            if s.current_tick() == ref_tick.tick {
                                s.tick(1);
                            }

                            // Compare registers
                            for (name, reg_idx) in &reg_keys {
                                if let Some(expected) = ref_tick.registers.get(*name) {
                                    if let Some(expected_val) = expected.as_i64() {
                                        let actual = s.state.registers[*reg_idx] as i64;
                                        if actual != expected_val {
                                            divergences.push(DivergenceInfo {
                                                tick: ref_tick.tick,
                                                register: name.to_string(),
                                                expected: expected_val,
                                                actual,
                                            });
                                        }
                                    }
                                }
                            }

                            if stop_at_first && !divergences.is_empty() {
                                break;
                            }
                        }

                        json_response(&CompareResponse {
                            divergences,
                            ticks_compared: ticks_to_compare,
                        })
                    }
                    Err(e) => error_response(400, &format!("Bad JSON: {e}")),
                }
            }

            (Method::Get, "/compare-paths") => {
                let mut s = session.lock().unwrap();
                let resp = s.compare_paths();
                json_response(&resp)
            }

            (Method::Post, "/snapshot") => {
                let mut s = session.lock().unwrap();
                let tick = s.current_tick();
                if !s.snapshots.iter().any(|(t, _)| *t == tick) {
                    let cloned = s.state.clone();
                    s.snapshots.push((tick, cloned));
                }
                json_response(&serde_json::json!({
                    "snapshot_created": tick,
                    "total_snapshots": s.snapshots.len(),
                }))
            }

            (Method::Get, "/snapshots") => {
                let s = session.lock().unwrap();
                let ticks: Vec<u32> = s.snapshots.iter().map(|(t, _)| *t).collect();
                json_response(&serde_json::json!({
                    "snapshots": ticks,
                    "count": ticks.len(),
                }))
            }

            (Method::Post, "/shutdown") => {
                let _ = request.respond(json_response(&serde_json::json!({"status": "shutting down"})));
                eprintln!("Shutdown requested");
                std::process::exit(0);
            }

            _ => error_response(404, &format!("Unknown endpoint: {}", path)),
        };

        if let Err(e) = request.respond(response) {
            eprintln!("Failed to send response: {e}");
        }
    }
}
