//! Packed broadcast-write pattern recognizer for CSS-DOS PACK_SIZE=2 memory.
//!
//! In packed memory, each `--mc{N}` cell holds two bytes (`b0 | b1<<8`).
//! Instead of the unpacked `if(style(--memAddrK: N): valK; ...; else: keep)`
//! shape (handled by `broadcast_write.rs`), the packed CSS uses a function
//! call chain:
//!
//! ```css
//! --mcN: --applySlot(--applySlot(... --applySlot(
//!    var(--__1mcN),
//!    var(--_slot{K-1}Live),
//!    calc(var(--memAddr{K-1}) - 2N),
//!    calc(var(--memAddr{K-1}) + 1 - 2N),
//!    var(--memVal{K-1}),
//!    var(--_slot{K-1}Width)),
//!    ...
//!    var(--_slot0Live),
//!    calc(var(--memAddr0) - 2N),
//!    calc(var(--memAddr0) + 1 - 2N),
//!    var(--memVal0),
//!    var(--_slot0Width));
//! ```
//!
//! `--applySlot(cell, live, loOff, hiOff, val, width)` returns:
//! - `cell` if `live == 0`
//! - `lowerBytes(val, 16)` when width=2 AND loOff=0 AND hiOff=1
//!   — width-2 aligned word write, whole cell becomes val.
//! - `(val & 0xff) << 8 | (cell & 0xff)` when width=2 AND loOff=1
//!   — width-2 straddle, slot's lo half lands here at off 1.
//! - `(val >> 8) | (cell & 0xff00)` when width=2 AND hiOff=0
//!   — width-2 straddle, slot's hi half lands here at off 0.
//! - byte splice at loOff for width=1.
//! - `cell` otherwise (slot doesn't touch this cell).
//!
//! Slot 0 is outermost (wins on collisions), slot K-1 is innermost.
//!
//! This module recognises the assignment shape and groups cells by
//! (gate_property, address_property, value_property, width_property) — each
//! group becomes one `PackedSlotPort` covering all cells written through that
//! slot.
//!
//! At runtime, instead of evaluating ~30 ops per cell × thousands of cells
//! per tick, the executor iterates K ports (one per write slot): if the gate
//! is live and the address is in range, look up the cell address from a dense
//! table, splice 1 or 2 bytes (width-aware) into the cell(s), write back.

use std::collections::HashMap;

use crate::types::*;

/// One write port through a single memory slot.
///
/// Conceptually: `if --gate_property == 1, take address A from --addr_property,
/// take value V from --val_property, and width W from --width_property`.
/// Width=1 splices a single byte at A. Width=2 writes a 16-bit word with
/// lo at A and hi at A+1 — the writes may straddle two adjacent cells when
/// A is odd-aligned.
#[derive(Debug, Clone)]
pub struct PackedSlotPort {
    /// e.g. `--_slot0Live`. Per-tick gate; skip the port entirely when 0.
    pub gate_property: String,
    /// e.g. `--memAddr0`. Byte address being written (lo byte for width=2).
    pub addr_property: String,
    /// e.g. `--memVal0`. Byte value (width=1) or 16-bit word (width=2;
    /// lo at addr, hi at addr+1).
    pub val_property: String,
    /// e.g. `--_slot0Width`. 1 = byte splice, 2 = word splice.
    pub width_property: String,
    /// Map from cell-base byte address → cell variable name (e.g. `--mc42`).
    /// Each port covers every writable cell; the runtime looks up the
    /// affected cell(s) via their byte-aligned cell base.
    pub address_map: HashMap<i64, String>,
    /// Pack size in bytes per cell (matches PackedBroadcastResult::pack).
    /// Carried per-port so downstream code doesn't have to thread the result
    /// alongside the port slice.
    pub pack: u8,
}

/// Result of packed-broadcast pattern recognition.
pub struct PackedBroadcastResult {
    /// One port per (gate, addr, val) triple — typically 6 (one per slot).
    pub ports: Vec<PackedSlotPort>,
    /// Cell property names absorbed (`--mc0` ... `--mcN`). These should be
    /// removed from the normal assignment loop because the ports cover them.
    pub absorbed_properties: std::collections::HashSet<String>,
    /// Pack size (currently always 2 — bytes per cell).
    pub pack: u8,
}

/// Threshold: don't absorb unless we can prove this is the dominant pattern.
/// CSS-DOS has thousands of cells; anything below this is a false positive.
const MIN_CELLS_FOR_RECOGNITION: usize = 100;

/// Analyse the given assignments and recognise the packed-broadcast pattern.
///
/// Walks each assignment whose property looks like `--mc{N}` (where N is a
/// non-negative integer). For each one that successfully decomposes into a
/// nested `--applySlot` chain ending in `var(--__1mcN)`, extracts the per-layer
/// (gate, addr, val, width) and inserts (cell_byte_addr → property_name) into
/// the matching port's address map.
///
/// Returns empty result if fewer than `MIN_CELLS_FOR_RECOGNITION` cells matched.
pub fn recognise_packed_broadcast(assignments: &[Assignment]) -> PackedBroadcastResult {
    // Map from (gate, addr, val, width) → port being built.
    //
    // A cabinet can carry several PORT FAMILIES — disjoint groups of cells
    // written through different address properties (e.g. one family keyed
    // on a guest-linear address, another on a window-local offset). Each
    // family's layers reference its own address property, so families land
    // in distinct ports here. Within one port, the cell's NAME index may
    // sit at a constant offset from the offset-arithmetic's byte address
    // (`delta` below): the address_map is keyed by the ARITHMETIC byte
    // address — the value space the addr property takes at runtime — so
    // the runtime lookup needs no delta at all.
    type PortKey = (String, String, String, String);
    struct PortBuild {
        port: PackedSlotPort,
        /// name byte addr − arithmetic byte addr; must be constant per port.
        delta: i64,
        /// Number of assignments that contributed a layer to this port.
        expected: usize,
    }
    let mut ports: HashMap<PortKey, PortBuild> = HashMap::new();
    let mut absorbed: std::collections::HashSet<String> = std::collections::HashSet::new();

    for assignment in assignments {
        let Some(cell_idx) = parse_packed_cell_index(&assignment.property) else {
            continue;
        };
        // Cell N → NAME byte address 2N (PACK_SIZE=2 hardcoded for now).
        let name_byte_addr = (cell_idx as i64) * 2;
        let layers = match decompose_apply_slot_chain(&assignment.value, &assignment.property) {
            Some(layers) => layers,
            None => continue,
        };
        // Validate before inserting: every layer's (name − arith) delta must
        // match its port's established delta, and no duplicate arithmetic key
        // may appear within a port. On any mismatch skip the whole assignment
        // — partial absorption is dangerous.
        let ok = layers.iter().all(|layer| {
            let key = (
                layer.gate_property.clone(),
                layer.addr_property.clone(),
                layer.val_property.clone(),
                layer.width_property.clone(),
            );
            match ports.get(&key) {
                Some(pb) => {
                    pb.delta == name_byte_addr - layer.cell_byte_addr
                        && !pb.port.address_map.contains_key(&layer.cell_byte_addr)
                }
                None => true,
            }
        });
        if !ok {
            continue;
        }
        for layer in layers {
            let key = (
                layer.gate_property.clone(),
                layer.addr_property.clone(),
                layer.val_property.clone(),
                layer.width_property.clone(),
            );
            let delta = name_byte_addr - layer.cell_byte_addr;
            let pb = ports.entry(key).or_insert_with(|| PortBuild {
                port: PackedSlotPort {
                    gate_property: layer.gate_property,
                    addr_property: layer.addr_property,
                    val_property: layer.val_property,
                    width_property: layer.width_property,
                    address_map: HashMap::new(),
                    pack: 2,
                },
                delta,
                expected: 0,
            });
            pb.port.address_map.insert(layer.cell_byte_addr, assignment.property.clone());
            pb.expected += 1;
        }
        absorbed.insert(assignment.property.clone());
    }

    // Threshold: abort recognition if too few cells matched. Returning a
    // partially-recognised set would absorb properties whose siblings still
    // run via the normal path, breaking same-cell collision semantics.
    if absorbed.len() < MIN_CELLS_FOR_RECOGNITION {
        return PackedBroadcastResult {
            ports: Vec::new(),
            absorbed_properties: std::collections::HashSet::new(),
            pack: 2,
        };
    }

    // Sanity check: every port must hold exactly one entry per assignment
    // that contributed a layer to it (no silent overwrites). Different
    // ports may cover different cell sets — that is the family model —
    // but each port's map must be internally complete.
    for pb in ports.values() {
        if pb.port.address_map.len() != pb.expected {
            return PackedBroadcastResult {
                ports: Vec::new(),
                absorbed_properties: std::collections::HashSet::new(),
                pack: 2,
            };
        }
    }

    let mut ports_vec: Vec<PackedSlotPort> = ports.into_values().map(|pb| pb.port).collect();
    // Sort by gate property so the runtime iteration order is deterministic
    // and matches CSS layer order (slot 0 first). Slot ordering is important:
    // slot 0 is OUTERMOST in the applySlot chain, so it should be applied LAST
    // at runtime to win same-cell collisions. Families with the same gate can
    // apply in any relative order — their cell sets are disjoint.
    ports_vec.sort_by(|a, b| {
        a.gate_property
            .cmp(&b.gate_property)
            .then_with(|| a.addr_property.cmp(&b.addr_property))
    });

    PackedBroadcastResult {
        ports: ports_vec,
        absorbed_properties: absorbed,
        pack: 2,
    }
}

/// Parse `--mc{N}` → `Some(N)`. Returns None for non-matching properties.
/// Rejects `--mc` (no digits), `--mcfoo` (non-numeric), `--m0` (no `c`).
fn parse_packed_cell_index(property: &str) -> Option<u32> {
    let rest = property.strip_prefix("--mc")?;
    if rest.is_empty() {
        return None;
    }
    rest.parse::<u32>().ok()
}

/// One layer of an `--applySlot` chain.
struct ApplySlotLayer {
    gate_property: String,
    addr_property: String,
    val_property: String,
    width_property: String,
    /// The byte address this layer is checking against (the constant subtracted
    /// from the address arg). Must be consistent across all layers of the same
    /// cell.
    cell_byte_addr: i64,
}

/// Decompose `--applySlot(--applySlot(... --applySlot(var(--__1mcN), ...), ...), ...)`
/// into a flat list of layers (outermost first — slot 0 first in CSS-DOS layout).
///
/// Each layer must match exactly:
///   `--applySlot(<inner>,
///                var(--gate),
///                calc(var(--addr) - K),
///                calc(var(--addr) + 1 - K),
///                var(--val),
///                var(--width))`
/// where K is a non-negative integer constant. The two off args must
/// reference the same `--addr` and use K identical to the cell's byte
/// base — the second arg differs by `+1` to expose the slot's hi half
/// position to width=2 splice handling.
///
/// The innermost `<inner>` must be `var(--__1{cell_property})` — the simple-keep
/// fallback. Anything else means there's side-channel logic and we bail.
///
/// Returns `None` if the structure doesn't match.
fn decompose_apply_slot_chain(expr: &Expr, cell_property: &str) -> Option<Vec<ApplySlotLayer>> {
    let mut layers = Vec::new();
    let mut cur = expr;
    loop {
        match cur {
            Expr::FunctionCall { name, args } => {
                // Accept anything ending with "applySlot" (handles future renames
                // and per-cart prefixes uniformly with the rest of the codebase).
                if !name.ends_with("applySlot") || args.len() != 6 {
                    return None;
                }
                let layer = extract_apply_slot_layer(&args[1], &args[2], &args[3], &args[4], &args[5])?;
                layers.push(layer);
                cur = &args[0];
            }
            Expr::Var { name, .. } => {
                // Innermost must be the simple-keep fallback `var(--__1mcN)`
                // (or any prefix variant `--__0mcN`, etc — match the broadcast
                // recogniser's `is_simple_keep`).
                if is_simple_keep_var(name, cell_property) {
                    return Some(layers);
                }
                return None;
            }
            _ => return None,
        }
    }
}

/// Extract one (gate, addr, val, width) tuple from an applySlot's five trailing args.
/// args[0] is the inner cell expression (handled by the caller).
/// args[1] = gate (var)
/// args[2] = loOff = calc(var(--addr) - K)        — splice byte at addr
/// args[3] = hiOff = calc(var(--addr) + 1 - K)    — splice hi byte at addr+1 (width=2 only)
/// args[4] = val (var)
/// args[5] = width (var)
fn extract_apply_slot_layer(
    gate_arg: &Expr,
    lo_off_arg: &Expr,
    hi_off_arg: &Expr,
    val_arg: &Expr,
    width_arg: &Expr,
) -> Option<ApplySlotLayer> {
    let gate_property = match gate_arg {
        Expr::Var { name, .. } => name.clone(),
        _ => return None,
    };
    let val_property = match val_arg {
        Expr::Var { name, .. } => name.clone(),
        _ => return None,
    };
    let width_property = match width_arg {
        Expr::Var { name, .. } => name.clone(),
        _ => return None,
    };
    // lo_off_arg should be `calc(var(--addr) - K)` where K is a non-negative constant.
    // K may be either a bare literal or `<literal> * <literal>` (kiln emits
    // `${cellIdx} * ${PACK_SIZE}` so the cell index sits at a position the
    // parser fast-path can template as an Addr hole; without that split, the
    // pre-folded base would be classified as a Free hole and the fast-path
    // would bail on the entire run).
    let (addr_property, cell_byte_addr) = parse_byte_off_arg(lo_off_arg)?;
    // hi_off_arg should be `calc(var(--addr) + 1 - K)` for the SAME addr and K.
    // The parser may have associated this differently — accept any form whose
    // const_eval over the non-addr atoms equals 1 - cell_byte_addr.
    if !hi_off_matches(hi_off_arg, &addr_property, cell_byte_addr) {
        return None;
    }
    Some(ApplySlotLayer {
        gate_property,
        addr_property,
        val_property,
        width_property,
        cell_byte_addr,
    })
}

/// Parse the lo-off argument shape `calc(var(--addr) - K)` (or bare `var(--addr)`
/// when K=0).
fn parse_byte_off_arg(off_arg: &Expr) -> Option<(String, i64)> {
    match off_arg {
        Expr::Calc(CalcOp::Sub(lhs, rhs)) => {
            let addr_name = match lhs.as_ref() {
                Expr::Var { name, .. } => name.clone(),
                _ => return None,
            };
            let k = const_eval(rhs.as_ref())?;
            if k < 0 {
                return None;
            }
            Some((addr_name, k))
        }
        // Special case: cell 0 might emit `calc(var(--addr) - 0)`, which the
        // parser may or may not collapse to bare `var(--addr)`. Handle both.
        Expr::Var { name, .. } => Some((name.clone(), 0)),
        _ => None,
    }
}

/// Match the hi-off shape: `calc(var(--addr) + 1 - K)` (or any algebraic
/// rearrangement thereof) where addr matches `addr_property` and the
/// constant offset evaluates to `1 - cell_byte_addr`.
///
/// Tolerated parse trees:
///   calc(var(addr) + 1 - K)
///   calc(var(addr) + (1 - K))
///   calc(var(addr) + 1 - <const>)   where <const> evals to K
///   calc(var(addr) - <const>)       where <const> evals to (K - 1) and K >= 1
fn hi_off_matches(expr: &Expr, addr_property: &str, cell_byte_addr: i64) -> bool {
    let target = 1 - cell_byte_addr;
    matches_addr_plus_const(expr, addr_property, target)
}

/// True iff `expr` evaluates to `var(addr_property) + target_const` for some
/// constant `target_const`. Walks the expression tree applying calc-add/sub
/// algebra to find the offset.
fn matches_addr_plus_const(expr: &Expr, addr_property: &str, target_const: i64) -> bool {
    if let Some(c) = eval_addr_offset(expr, addr_property) {
        return c == target_const;
    }
    false
}

/// Evaluate `expr` to `Some(c)` where it represents `var(addr_property) + c`,
/// or `None` if the expression isn't of that form.
fn eval_addr_offset(expr: &Expr, addr_property: &str) -> Option<i64> {
    match expr {
        Expr::Var { name, .. } if name == addr_property => Some(0),
        Expr::Calc(op) => match op {
            CalcOp::Add(a, b) => {
                if let Some(ac) = eval_addr_offset(a, addr_property) {
                    let bc = const_eval(b)?;
                    Some(ac + bc)
                } else if let Some(bc) = eval_addr_offset(b, addr_property) {
                    let ac = const_eval(a)?;
                    Some(ac + bc)
                } else {
                    None
                }
            }
            CalcOp::Sub(a, b) => {
                if let Some(ac) = eval_addr_offset(a, addr_property) {
                    let bc = const_eval(b)?;
                    Some(ac - bc)
                } else {
                    // var(addr) shows up only on the left side of a relevant
                    // sub here — anything else means the expression doesn't
                    // reduce to addr + const.
                    None
                }
            }
            _ => None,
        },
        _ => None,
    }
}

/// Evaluate a constant-only expression tree to an integer, or return None
/// if the expression references any variables or function calls.
/// Supports bare literals and literal-only Add/Sub/Mul/Div calcs so we can
/// accept both `K` and `${cellIdx} * ${PACK_SIZE}` as the RHS of the byte
/// offset subtraction.
fn const_eval(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Literal(v) => Some(*v as i64),
        Expr::Calc(op) => match op {
            CalcOp::Add(a, b) => Some(const_eval(a)? + const_eval(b)?),
            CalcOp::Sub(a, b) => Some(const_eval(a)? - const_eval(b)?),
            CalcOp::Mul(a, b) => Some(const_eval(a)? * const_eval(b)?),
            CalcOp::Div(a, b) => {
                let bv = const_eval(b)?;
                if bv == 0 {
                    return None;
                }
                Some(const_eval(a)? / bv)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Match `var(--{prefix}{cell_property})` where prefix is "__0", "__1", "__2",
/// or empty. Same convention as `broadcast_write::is_simple_keep`.
fn is_simple_keep_var(var_name: &str, cell_property: &str) -> bool {
    let bare_var = if let Some(rest) = var_name.strip_prefix("--__") {
        // --__0X, --__1X, --__2X → X
        if rest.is_empty() {
            return false;
        }
        &rest[1..]
    } else if let Some(rest) = var_name.strip_prefix("--") {
        rest
    } else {
        return false;
    };
    let bare_cell = cell_property.strip_prefix("--").unwrap_or(cell_property);
    bare_var == bare_cell
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Expr {
        Expr::Var {
            name: name.to_string(),
            fallback: None,
        }
    }

    fn lit(v: f64) -> Expr {
        Expr::Literal(v)
    }

    /// Build `calc(var(--addr) - k)` with a bare literal subtrahend.
    /// Models the legacy pre-folded emit shape.
    fn off(addr: &str, k: i64) -> Expr {
        if k == 0 {
            // Mirror what the parser may produce for cell 0.
            return var(addr);
        }
        Expr::Calc(CalcOp::Sub(Box::new(var(addr)), Box::new(lit(k as f64))))
    }

    /// Build `calc(var(--addr) + 1 - k)` — the hiOff form passed as the 4th
    /// applySlot arg. Differs from `off` by a constant +1 (the byte-after).
    /// When k=0, kiln emits `calc(var(--addr) + 1 - 0 * pack)`; the parser
    /// may collapse to `calc(var(--addr) + 1)`. Match the collapsed form.
    fn hi_off(addr: &str, k: i64) -> Expr {
        if k == 0 {
            return Expr::Calc(CalcOp::Add(Box::new(var(addr)), Box::new(lit(1.0))));
        }
        // var(addr) + 1 - k
        Expr::Calc(CalcOp::Sub(
            Box::new(Expr::Calc(CalcOp::Add(
                Box::new(var(addr)),
                Box::new(lit(1.0)),
            ))),
            Box::new(lit(k as f64)),
        ))
    }

    /// Build `calc(var(--addr) - cellIdx * pack)` with the subtrahend as a
    /// `Mul(Literal, Literal)`. Models the current kiln emit shape.
    fn off_mul(addr: &str, cell_idx: i64, pack: i64) -> Expr {
        Expr::Calc(CalcOp::Sub(
            Box::new(var(addr)),
            Box::new(Expr::Calc(CalcOp::Mul(
                Box::new(lit(cell_idx as f64)),
                Box::new(lit(pack as f64)),
            ))),
        ))
    }

    /// Build `calc(var(--addr) + 1 - cellIdx * pack)`.
    fn hi_off_mul(addr: &str, cell_idx: i64, pack: i64) -> Expr {
        Expr::Calc(CalcOp::Sub(
            Box::new(Expr::Calc(CalcOp::Add(
                Box::new(var(addr)),
                Box::new(lit(1.0)),
            ))),
            Box::new(Expr::Calc(CalcOp::Mul(
                Box::new(lit(cell_idx as f64)),
                Box::new(lit(pack as f64)),
            ))),
        ))
    }

    /// Build a NUM-deep applySlot chain for cell N. Three slots in the
    /// post-2026-04-28 word-pair scheme.
    fn make_packed_cell(cell_idx: u32) -> Assignment {
        let cell_addr = (cell_idx as i64) * 2;
        let cell_prop = format!("--mc{cell_idx}");
        let mut expr = var(&format!("--__1mc{cell_idx}"));
        // Slot N-1 innermost, slot 0 outermost.
        for slot in (0..3).rev() {
            expr = Expr::FunctionCall {
                name: "--applySlot".to_string(),
                args: vec![
                    expr,
                    var(&format!("--_slot{slot}Live")),
                    off(&format!("--memAddr{slot}"), cell_addr),
                    hi_off(&format!("--memAddr{slot}"), cell_addr),
                    var(&format!("--memVal{slot}")),
                    var(&format!("--_slot{slot}Width")),
                ],
            };
        }
        Assignment {
            property: cell_prop,
            value: expr,
        }
    }

    #[test]
    fn recognises_thousand_cell_packed_broadcast() {
        let assignments: Vec<_> = (0..1000).map(make_packed_cell).collect();
        let result = recognise_packed_broadcast(&assignments);
        assert_eq!(result.ports.len(), 3, "Should have 3 slot ports");
        assert_eq!(result.absorbed_properties.len(), 1000);
        assert_eq!(result.pack, 2);
        for (i, port) in result.ports.iter().enumerate() {
            assert_eq!(port.gate_property, format!("--_slot{i}Live"));
            assert_eq!(port.addr_property, format!("--memAddr{i}"));
            assert_eq!(port.val_property, format!("--memVal{i}"));
            assert_eq!(port.width_property, format!("--_slot{i}Width"));
            assert_eq!(port.address_map.len(), 1000);
            assert_eq!(
                port.address_map.get(&0).map(|s| s.as_str()),
                Some("--mc0")
            );
            assert_eq!(
                port.address_map.get(&1998).map(|s| s.as_str()),
                Some("--mc999")
            );
        }
    }

    #[test]
    fn recognises_packed_broadcast_with_mul_offset() {
        // Build 200 cells whose offset is `calc(var(--addr) - idx * 2)` rather
        // than the pre-folded `calc(var(--addr) - base)`.  Every other shape
        // matches the first test.
        let mut assignments = Vec::new();
        for cell_idx in 0..200u32 {
            let cell_prop = format!("--mc{cell_idx}");
            let mut expr = var(&format!("--__1mc{cell_idx}"));
            for slot in (0..3).rev() {
                expr = Expr::FunctionCall {
                    name: "--applySlot".to_string(),
                    args: vec![
                        expr,
                        var(&format!("--_slot{slot}Live")),
                        off_mul(&format!("--memAddr{slot}"), cell_idx as i64, 2),
                        hi_off_mul(&format!("--memAddr{slot}"), cell_idx as i64, 2),
                        var(&format!("--memVal{slot}")),
                        var(&format!("--_slot{slot}Width")),
                    ],
                };
            }
            assignments.push(Assignment { property: cell_prop, value: expr });
        }
        let result = recognise_packed_broadcast(&assignments);
        assert_eq!(result.ports.len(), 3);
        assert_eq!(result.absorbed_properties.len(), 200);
        for port in &result.ports {
            assert_eq!(port.address_map.len(), 200);
            // Cell 7 → byte addr 14.
            assert_eq!(port.address_map.get(&14).map(|s| s.as_str()), Some("--mc7"));
        }
    }

    #[test]
    fn rejects_too_few_cells() {
        let assignments: Vec<_> = (0..50).map(make_packed_cell).collect();
        let result = recognise_packed_broadcast(&assignments);
        assert!(result.ports.is_empty());
        assert!(result.absorbed_properties.is_empty());
    }

    #[test]
    fn rejects_non_mc_properties() {
        let assignments = vec![Assignment {
            property: "--m42".to_string(),
            value: var("--__1m42"),
        }];
        let result = recognise_packed_broadcast(&assignments);
        assert!(result.ports.is_empty());
    }

    #[test]
    fn rejects_chain_with_wrong_inner_var() {
        // Innermost is `var(--something_else)`, not `var(--__1mcN)`.
        let mut a = make_packed_cell(0);
        // Replace innermost var with an unrelated one.
        let mut cur = &mut a.value;
        loop {
            match cur {
                Expr::FunctionCall { args, .. } => {
                    cur = &mut args[0];
                }
                Expr::Var { name, .. } => {
                    *name = "--unrelated".to_string();
                    break;
                }
                _ => unreachable!(),
            }
        }
        let mut assignments = vec![a];
        // Also need to push enough other cells to clear the threshold so we're
        // testing the "this cell rejected" path, not the "too few cells" path.
        assignments.extend((1..200).map(make_packed_cell));
        let result = recognise_packed_broadcast(&assignments);
        // The malformed cell 0 should not appear in any port's address_map.
        for port in &result.ports {
            assert!(
                !port.address_map.contains_key(&0),
                "Malformed cell should not be absorbed"
            );
        }
        assert!(!result.absorbed_properties.contains("--mc0"));
        assert_eq!(result.absorbed_properties.len(), 199);
    }

    #[test]
    fn rejects_when_offset_does_not_match_cell_addr() {
        // Build a cell where one layer subtracts the wrong constant.
        let mut a = make_packed_cell(5);
        // Tweak the outermost (slot 0) layer to use a wrong cell address.
        if let Expr::FunctionCall { args, .. } = &mut a.value {
            args[2] = off("--memAddr0", 999); // wrong: should be 10 for cell 5
        }
        let mut assignments = vec![a];
        assignments.extend((10..200).map(make_packed_cell));
        let result = recognise_packed_broadcast(&assignments);
        assert!(!result.absorbed_properties.contains("--mc5"));
    }

    #[test]
    fn cell_zero_handled_with_collapsed_offset() {
        // Cell 0 uses bare `var(--memAddrK)` (no `- 0`).
        // Make sure we don't reject it.
        let assignments: Vec<_> = (0..200).map(make_packed_cell).collect();
        let result = recognise_packed_broadcast(&assignments);
        assert!(result.absorbed_properties.contains("--mc0"));
        assert_eq!(
            result.ports[0].address_map.get(&0).map(|s| s.as_str()),
            Some("--mc0")
        );
    }

    #[test]
    fn parse_packed_cell_index_basic() {
        assert_eq!(parse_packed_cell_index("--mc0"), Some(0));
        assert_eq!(parse_packed_cell_index("--mc42"), Some(42));
        assert_eq!(parse_packed_cell_index("--mc999999"), Some(999999));
        assert_eq!(parse_packed_cell_index("--mc"), None);
        assert_eq!(parse_packed_cell_index("--m42"), None);
        assert_eq!(parse_packed_cell_index("--mcfoo"), None);
        assert_eq!(parse_packed_cell_index("--mc-1"), None);
    }
}
