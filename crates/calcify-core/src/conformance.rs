//! Conformance testing harness for comparing calc(ify) output against Chrome.
//!
//! The conformance test protocol:
//! 1. Chrome baseline: Run x86CSS in Chrome, capture state after each tick via
//!    getComputedStyle(). Save as JSON snapshots.
//! 2. Engine comparison: Run the same CSS through calc(ify) for the same number
//!    of ticks. Compare state after each tick against Chrome snapshots.
//! 3. Divergence: Binary-search to find the first divergent tick and property.

use serde::{Deserialize, Serialize};

use crate::state::{reg, State};

/// A snapshot of the machine state at a specific tick.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateSnapshot {
    pub tick: u32,
    pub registers: RegisterSnapshot,
    /// Changed memory addresses and their values (sparse — only non-zero or changed).
    #[serde(default)]
    pub memory: Vec<MemoryEntry>,
}

/// Register values at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterSnapshot {
    pub ax: i32,
    pub cx: i32,
    pub dx: i32,
    pub bx: i32,
    pub sp: i32,
    pub bp: i32,
    pub si: i32,
    pub di: i32,
    pub ip: i32,
    pub es: i32,
    pub cs: i32,
    pub ss: i32,
    pub ds: i32,
    pub flags: i32,
}

/// A single memory address and its value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryEntry {
    pub address: u32,
    pub value: u8,
}

/// Result of comparing two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    pub tick: u32,
    pub register_diffs: Vec<RegisterDiff>,
    pub memory_diffs: Vec<MemoryDiff>,
}

#[derive(Debug, Clone)]
pub struct RegisterDiff {
    pub name: &'static str,
    pub expected: i32,
    pub actual: i32,
}

#[derive(Debug, Clone)]
pub struct MemoryDiff {
    pub address: u32,
    pub expected: u8,
    pub actual: u8,
}

impl StateSnapshot {
    /// Capture a snapshot from the current engine state.
    pub fn from_state(state: &State, tick: u32) -> Self {
        Self {
            tick,
            registers: RegisterSnapshot::from_state(state),
            memory: Vec::new(), // Populated by capture_memory if needed
        }
    }

    /// Capture a snapshot including all non-zero memory.
    pub fn from_state_with_memory(state: &State, tick: u32) -> Self {
        let memory = state
            .memory
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(addr, &value)| MemoryEntry {
                address: addr as u32,
                value,
            })
            .collect();

        Self {
            tick,
            registers: RegisterSnapshot::from_state(state),
            memory,
        }
    }

    /// Compare this snapshot against another, returning differences.
    pub fn diff(&self, other: &StateSnapshot) -> Option<SnapshotDiff> {
        let register_diffs = self.registers.diff(&other.registers);
        let memory_diffs = diff_memory(&self.memory, &other.memory);

        if register_diffs.is_empty() && memory_diffs.is_empty() {
            None
        } else {
            Some(SnapshotDiff {
                tick: self.tick,
                register_diffs,
                memory_diffs,
            })
        }
    }
}

impl RegisterSnapshot {
    pub fn from_state(state: &State) -> Self {
        Self {
            ax: state.registers[reg::AX],
            cx: state.registers[reg::CX],
            dx: state.registers[reg::DX],
            bx: state.registers[reg::BX],
            sp: state.registers[reg::SP],
            bp: state.registers[reg::BP],
            si: state.registers[reg::SI],
            di: state.registers[reg::DI],
            ip: state.registers[reg::IP],
            es: state.registers[reg::ES],
            cs: state.registers[reg::CS],
            ss: state.registers[reg::SS],
            ds: state.registers[reg::DS],
            flags: state.registers[reg::FLAGS],
        }
    }

    /// Apply this snapshot to a State.
    pub fn apply_to(&self, state: &mut State) {
        state.registers[reg::AX] = self.ax;
        state.registers[reg::CX] = self.cx;
        state.registers[reg::DX] = self.dx;
        state.registers[reg::BX] = self.bx;
        state.registers[reg::SP] = self.sp;
        state.registers[reg::BP] = self.bp;
        state.registers[reg::SI] = self.si;
        state.registers[reg::DI] = self.di;
        state.registers[reg::IP] = self.ip;
        state.registers[reg::ES] = self.es;
        state.registers[reg::CS] = self.cs;
        state.registers[reg::SS] = self.ss;
        state.registers[reg::DS] = self.ds;
        state.registers[reg::FLAGS] = self.flags;
    }

    fn diff(&self, other: &RegisterSnapshot) -> Vec<RegisterDiff> {
        let mut diffs = Vec::new();
        let checks: &[(&str, i32, i32)] = &[
            ("AX", self.ax, other.ax),
            ("CX", self.cx, other.cx),
            ("DX", self.dx, other.dx),
            ("BX", self.bx, other.bx),
            ("SP", self.sp, other.sp),
            ("BP", self.bp, other.bp),
            ("SI", self.si, other.si),
            ("DI", self.di, other.di),
            ("IP", self.ip, other.ip),
            ("ES", self.es, other.es),
            ("CS", self.cs, other.cs),
            ("SS", self.ss, other.ss),
            ("DS", self.ds, other.ds),
            ("FLAGS", self.flags, other.flags),
        ];

        for &(name, expected, actual) in checks {
            if expected != actual {
                // name is &str, but we need &'static str for RegisterDiff
                // Since these are all static strings from the match above, this is safe
                diffs.push(RegisterDiff {
                    name,
                    expected,
                    actual,
                });
            }
        }

        diffs
    }
}

fn diff_memory(expected: &[MemoryEntry], actual: &[MemoryEntry]) -> Vec<MemoryDiff> {
    use std::collections::HashMap;

    let expected_map: HashMap<u32, u8> = expected.iter().map(|e| (e.address, e.value)).collect();
    let actual_map: HashMap<u32, u8> = actual.iter().map(|e| (e.address, e.value)).collect();

    let mut diffs = Vec::new();
    for (&addr, &exp_val) in &expected_map {
        let act_val = actual_map.get(&addr).copied().unwrap_or(0);
        if exp_val != act_val {
            diffs.push(MemoryDiff {
                address: addr,
                expected: exp_val,
                actual: act_val,
            });
        }
    }
    for (&addr, &act_val) in &actual_map {
        if !expected_map.contains_key(&addr) && act_val != 0 {
            diffs.push(MemoryDiff {
                address: addr,
                expected: 0,
                actual: act_val,
            });
        }
    }

    diffs.sort_by_key(|d| d.address);
    diffs
}

/// A collection of state snapshots for conformance testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceTrace {
    /// Source CSS file hash (for cache validation).
    pub css_hash: String,
    /// Chrome version used for baseline capture.
    pub chrome_version: String,
    /// Snapshots at each tick.
    pub snapshots: Vec<StateSnapshot>,
}

impl ConformanceTrace {
    /// Load a trace from a JSON file.
    pub fn load(path: &str) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
        serde_json::from_str(&content).map_err(|e| format!("failed to parse {path}: {e}"))
    }

    /// Save a trace to a JSON file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let content =
            serde_json::to_string_pretty(self).map_err(|e| format!("failed to serialize: {e}"))?;
        std::fs::write(path, content).map_err(|e| format!("failed to write {path}: {e}"))
    }
}

impl std::fmt::Display for SnapshotDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tick {} diverged:", self.tick)?;
        for d in &self.register_diffs {
            writeln!(
                f,
                "  {} expected={} actual={}",
                d.name, d.expected, d.actual
            )?;
        }
        for d in &self.memory_diffs {
            writeln!(
                f,
                "  mem[{}] expected={} actual={}",
                d.address, d.expected, d.actual
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip() {
        let mut state = State::default();
        state.registers[reg::AX] = 0x1234;
        state.registers[reg::IP] = 0x100;
        state.memory[0] = 0xFF;
        state.memory[1] = 0x42;

        let snapshot = StateSnapshot::from_state_with_memory(&state, 0);
        let json = serde_json::to_string(&snapshot).unwrap();
        let restored: StateSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(snapshot, restored);
        assert_eq!(restored.registers.ax, 0x1234);
        assert_eq!(restored.memory.len(), 2);
    }

    #[test]
    fn snapshot_diff_detects_changes() {
        let mut state1 = State::default();
        state1.registers[reg::AX] = 100;

        let mut state2 = State::default();
        state2.registers[reg::AX] = 200;

        let snap1 = StateSnapshot::from_state(&state1, 0);
        let snap2 = StateSnapshot::from_state(&state2, 0);

        let diff = snap1.diff(&snap2).expect("should differ");
        assert_eq!(diff.register_diffs.len(), 1);
        assert_eq!(diff.register_diffs[0].name, "AX");
        assert_eq!(diff.register_diffs[0].expected, 100);
        assert_eq!(diff.register_diffs[0].actual, 200);
    }

    #[test]
    fn identical_snapshots_no_diff() {
        let state = State::default();
        let snap1 = StateSnapshot::from_state(&state, 0);
        let snap2 = StateSnapshot::from_state(&state, 0);

        assert!(snap1.diff(&snap2).is_none());
    }
}
