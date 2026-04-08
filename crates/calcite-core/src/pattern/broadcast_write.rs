//! Broadcast write pattern: `if(style(--dest: {addr}): value; else: keep)` → direct store.
//!
//! Detects: a set of assignments where each has the form:
//!   `--varN: if(style(--dest: N): value; else: var(--__1varN))`
//! All checking the same destination property against their own address.
//!
//! Works with both legacy (`--addrDestA/B/C`) and v2 (`--memAddr0/1/2`) patterns.
//!
//! Replaces: evaluate `--dest` once, write `value` to `state[dest]` directly.

use std::collections::{HashMap, HashSet};

use crate::types::*;

/// A recognised broadcast write pattern.
#[derive(Debug, Clone)]
pub struct BroadcastWrite {
    /// The destination address property (e.g., `--addrDestA`).
    pub dest_property: String,
    /// The value expression property (e.g., `--addrValA`).
    pub value_expr: Expr,
    /// The address → variable name mapping (O(1) lookup).
    /// Key: the integer address that each variable checks against.
    /// Value: the variable name that should be written to.
    pub address_map: HashMap<i64, String>,
    /// Word-write spillover: when `isWordWrite == 1` and dest is address N,
    /// also write the high byte to address N+1.
    /// Key: the address being tested (N-1 in the condition).
    /// Value: (target_var_name, value_expr for the high byte).
    pub spillover_map: HashMap<i64, (String, Expr)>,
    /// The property that gates spillover writes (e.g., `--isWordWrite`).
    pub spillover_guard: Option<String>,
}

/// Result of broadcast pattern recognition.
pub struct BroadcastResult {
    /// Recognised broadcast write patterns.
    pub writes: Vec<BroadcastWrite>,
    /// Property names absorbed into broadcast writes (should be removed from the assignment loop).
    pub absorbed_properties: HashSet<String>,
}

/// Analyse a set of assignments to detect the broadcast write pattern.
///
/// Memory cells in x86CSS check multiple write ports:
///   `--m0: if(style(--addrDestA:0): valA; style(--addrDestB:0): valB; else: keep)`
///
/// We create one `BroadcastWrite` per port. Assignments where ALL branches are
/// pure `--addrDest*` checks are absorbed; register assignments that mix in
/// execution logic (e.g. `--addrJump`) are left in the normal assignment loop.
pub fn recognise_broadcast(assignments: &[Assignment]) -> BroadcastResult {
    // Phase 1: Collect all broadcast port entries, grouped by dest_property.
    // Each entry records the address, property name, and value expression.
    // We DON'T assume all entries share the same value_expr — split registers
    // (e.g. DX) contribute ports with byte-merge expressions that differ from
    // the simple var(--addrValA) used by memory cells.
    let mut direct_groups: HashMap<String, Vec<(i64, String, Expr)>> = HashMap::new();
    let mut spillover_groups: HashMap<String, Vec<(i64, String, String, Expr)>> = HashMap::new();
    let mut pure_broadcast: HashSet<String> = HashSet::new();

    // Track how many direct ports each assignment contributes (across all dest
    // properties). An assignment can only be absorbed if ALL its ports land in
    // broadcast groups — otherwise the non-absorbed ports lose their evaluation.
    let mut port_counts: HashMap<String, usize> = HashMap::new();

    for assignment in assignments {
        // Skip buffer copies — they're no-ops in mutable state and never broadcast targets.
        if assignment.property.starts_with("--__") {
            continue;
        }
        if let Some(ports) = extract_broadcast_ports(assignment) {
            pure_broadcast.insert(assignment.property.clone());
            for port in ports {
                match port {
                    BroadcastPort::Direct {
                        dest_property,
                        address,
                        value_expr,
                    } => {
                        *port_counts.entry(assignment.property.clone()).or_insert(0) += 1;
                        direct_groups
                            .entry(dest_property)
                            .or_default()
                            .push((address, assignment.property.clone(), value_expr));
                    }
                    BroadcastPort::Spillover {
                        dest_property,
                        source_address,
                        guard_property,
                        value_expr,
                    } => {
                        spillover_groups.entry(dest_property).or_default().push((
                            source_address,
                            assignment.property.clone(),
                            guard_property,
                            value_expr,
                        ));
                    }
                }
            }
        }
    }

    // Track how many of each assignment's ports were actually absorbed.
    let mut absorbed_port_counts: HashMap<String, usize> = HashMap::new();

    // Phase 2: For each dest_property, find the majority value expression and
    // only absorb entries that use it. Entries with different expressions (e.g.
    // split register byte-merges) are left in the normal assignment loop.
    let writes: Vec<BroadcastWrite> = direct_groups
        .into_iter()
        .filter_map(|(dest_property, entries)| {
            // Find the most common value expression in this group.
            // This is O(n²) in the number of distinct expressions, but in practice
            // there are only 1-3 distinct expressions per group.
            let mut expr_groups: Vec<(Expr, Vec<(i64, String)>)> = Vec::new();
            for (addr, name, value_expr) in entries {
                if let Some(group) = expr_groups.iter_mut().find(|(e, _)| *e == value_expr) {
                    group.1.push((addr, name));
                } else {
                    expr_groups.push((value_expr, vec![(addr, name)]));
                }
            }

            // Pick the largest group (the true broadcast targets).
            let (value_expr, entries) = expr_groups
                .into_iter()
                .max_by_key(|(_, entries)| entries.len())?;

            if entries.len() < 10 {
                return None;
            }

            // A broadcast write maps different addresses to different target
            // properties. If most entries map to the same property (e.g. a single
            // register's opcode dispatch), this is not a broadcast pattern.
            let unique_names: HashSet<&str> = entries.iter().map(|(_, n)| n.as_str()).collect();
            if unique_names.len() * 2 < entries.len() {
                // Fewer than half the entries are unique targets — not broadcast.
                return None;
            }

            let mut address_map = HashMap::with_capacity(entries.len());
            for (addr, name) in entries {
                *absorbed_port_counts.entry(name.clone()).or_insert(0) += 1;
                address_map.insert(addr, name);
            }

            // Build spillover map for this dest_property
            let spillovers = spillover_groups.remove(&dest_property);
            let (spillover_map, spillover_guard) = if let Some(spills) = spillovers {
                let guard = spills.first().map(|(_, _, g, _)| g.clone());
                let map = spills
                    .into_iter()
                    .map(|(src_addr, var_name, _, val_expr)| (src_addr, (var_name, val_expr)))
                    .collect();
                (map, guard)
            } else {
                (HashMap::new(), None)
            };

            Some(BroadcastWrite {
                dest_property,
                value_expr,
                address_map,
                spillover_map,
                spillover_guard,
            })
        })
        .collect();

    // An assignment is only absorbed if:
    // 1. It's a pure broadcast target (all branches are addrDest checks)
    // 2. ALL of its direct ports were absorbed into broadcast groups
    //
    // Split registers like DX have some ports that match the majority expression
    // (e.g. var(--addrValA) for word writes) and others that don't (byte-merge
    // expressions). If ANY port is excluded, the assignment must stay in the
    // normal loop so those cases are still evaluated.
    let mut absorbed_properties = HashSet::new();
    for (name, total) in &port_counts {
        let absorbed = absorbed_port_counts.get(name).copied().unwrap_or(0);
        if absorbed == *total && pure_broadcast.contains(name) {
            absorbed_properties.insert(name.clone());
        }
    }

    BroadcastResult {
        writes,
        absorbed_properties,
    }
}

/// A port extracted from a broadcast write assignment.
#[derive(Debug)]
enum BroadcastPort {
    /// Direct write: `style(--addrDestX: ADDR) → value`
    Direct {
        dest_property: String,
        address: i64,
        value_expr: Expr,
    },
    /// Spillover write: `style(--addrDestX: ADDR) and style(--isWordWrite: 1) → value`
    /// The address is the *source* address (N-1); the target cell is at N.
    Spillover {
        dest_property: String,
        source_address: i64,
        guard_property: String,
        value_expr: Expr,
    },
}

/// Extract all broadcast write ports from an assignment.
///
/// Returns `None` if:
/// - Any branch tests a non-address property (execution logic mixed in)
/// - The fallback (else branch) is not a simple keep of the previous value
///
/// Registers like SP, SI, DI have side-channel deltas in their else branches
/// (e.g., `else: calc(var(--__1SP) + var(--moveStack))`) and must NOT be absorbed.
///
/// The address property is recognised structurally (any `style(--X: <int>)` condition),
/// not by name prefix. This handles both legacy `--addrDestA/B/C` and v2 `--memAddr0/1/2`.
///
/// Returns `Some(vec of BroadcastPort)` — one per branch.
fn extract_broadcast_ports(assignment: &Assignment) -> Option<Vec<BroadcastPort>> {
    match &assignment.value {
        Expr::StyleCondition { branches, fallback } => {
            // Check the fallback: only absorb if it's a simple keep (var(--__1X) or var(--X)).
            // If the fallback has computation (calc, function call, etc.), this register
            // has side-channel logic that must be evaluated on every tick.
            if !is_simple_keep(fallback, &assignment.property) {
                return None;
            }

            let mut ports = Vec::with_capacity(branches.len());
            for branch in branches {
                match &branch.condition {
                    StyleTest::Single {
                        property,
                        value: Expr::Literal(v),
                    } => {
                        ports.push(BroadcastPort::Direct {
                            dest_property: property.clone(),
                            address: *v as i64,
                            value_expr: branch.then.clone(),
                        });
                    }
                    StyleTest::Single { .. } => return None,
                    StyleTest::And(tests) if tests.len() == 2 => {
                        // Match: style(--addrX: N) and style(--guard: 1)
                        // Identify the address test (the one whose property also appears
                        // in single-condition branches) vs the guard test (value == 1).
                        let (mut addr_test, mut guard_test) = (None, None);
                        for t in tests {
                            if let StyleTest::Single {
                                property,
                                value: Expr::Literal(v),
                            } = t
                            {
                                if *v as i64 == 1 && guard_test.is_none() {
                                    // Candidate guard (value == 1)
                                    guard_test = Some(property.clone());
                                } else if addr_test.is_none() {
                                    // Candidate address
                                    addr_test = Some((property.clone(), *v as i64));
                                }
                            }
                        }
                        // If both tests matched as guard candidates (both value == 1),
                        // the first was taken as guard and second as address, which is wrong.
                        // Re-check: the address property should match a Direct port's property.
                        // For now, accept any valid pair — Phase 2 grouping will discard
                        // nonsense combinations via the min-count threshold.
                        match (addr_test, guard_test) {
                            (Some((dest_property, source_address)), Some(guard_property)) => {
                                ports.push(BroadcastPort::Spillover {
                                    dest_property,
                                    source_address,
                                    guard_property,
                                    value_expr: branch.then.clone(),
                                });
                            }
                            _ => return None,
                        }
                    }
                    _ => return None,
                }
            }
            if ports.is_empty() {
                None
            } else {
                Some(ports)
            }
        }
        _ => None,
    }
}

/// Check if a fallback expression is a simple "keep previous value" pattern.
///
/// Pure broadcast targets use `var(--__1X)` as their fallback — just keeping the
/// previous tick's value when no write port targets them. Registers with side channels
/// (SP, SI, DI, CS, flags) have computation in their else branches and must not be
/// absorbed into the broadcast write optimization.
fn is_simple_keep(fallback: &Expr, property_name: &str) -> bool {
    match fallback {
        Expr::Var { name, .. } => {
            // Accept var(--__1X) or var(--__0X) or var(--X) as simple keeps
            let bare = if let Some(rest) = name.strip_prefix("--__") {
                // --__0X, --__1X, --__2X → X
                &rest[1..]
            } else if let Some(rest) = name.strip_prefix("--") {
                rest
            } else {
                return false;
            };
            let prop_bare = property_name.strip_prefix("--").unwrap_or(property_name);
            bare == prop_bare
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_broadcast_assignment(name: &str, addr: i64) -> Assignment {
        Assignment {
            property: format!("--{name}"),
            value: Expr::StyleCondition {
                branches: vec![StyleBranch {
                    condition: StyleTest::Single {
                        property: "--addrDestA".to_string(),
                        value: Expr::Literal(addr as f64),
                    },
                    then: Expr::Var {
                        name: "--addrValA".to_string(),
                        fallback: None,
                    },
                }],
                fallback: Box::new(Expr::Var {
                    name: format!("--__1{name}"),
                    fallback: None,
                }),
            },
        }
    }

    /// Build a compound memory cell assignment matching the x86CSS pattern:
    /// `--mN: if(style(--addrDestA:N):val1; style(--addrDestA:N-1) and style(--isWordWrite:1):val2; style(--addrDestB:N):valB; else:keep)`
    fn make_compound_broadcast_assignment(name: &str, addr: i64) -> Assignment {
        let mut branches = vec![StyleBranch {
            condition: StyleTest::Single {
                property: "--addrDestA".to_string(),
                value: Expr::Literal(addr as f64),
            },
            then: Expr::Var {
                name: "--addrValA".to_string(),
                fallback: None,
            },
        }];
        if addr > 0 {
            // Add spillover branch: style(--addrDestA: addr-1) and style(--isWordWrite: 1) → high byte
            branches.push(StyleBranch {
                condition: StyleTest::And(vec![
                    StyleTest::Single {
                        property: "--addrDestA".to_string(),
                        value: Expr::Literal((addr - 1) as f64),
                    },
                    StyleTest::Single {
                        property: "--isWordWrite".to_string(),
                        value: Expr::Literal(1.0),
                    },
                ]),
                then: Expr::Var {
                    name: "--addrValA1".to_string(),
                    fallback: None,
                },
            });
        }
        // Add second write port
        branches.push(StyleBranch {
            condition: StyleTest::Single {
                property: "--addrDestB".to_string(),
                value: Expr::Literal(addr as f64),
            },
            then: Expr::Var {
                name: "--addrValB".to_string(),
                fallback: None,
            },
        });

        Assignment {
            property: format!("--{name}"),
            value: Expr::StyleCondition {
                branches,
                fallback: Box::new(Expr::Var {
                    name: format!("--__1{name}"),
                    fallback: None,
                }),
            },
        }
    }

    #[test]
    fn detects_simple_broadcast() {
        let assignments: Vec<Assignment> = (0..20)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();
        let result = recognise_broadcast(&assignments);
        assert_eq!(result.writes.len(), 1);
        assert_eq!(result.writes[0].dest_property, "--addrDestA");
        assert_eq!(result.writes[0].address_map.len(), 20);
        assert_eq!(result.absorbed_properties.len(), 20);
    }

    #[test]
    fn compound_memory_cell_broadcast() {
        let assignments: Vec<Assignment> = (0..20)
            .map(|i| make_compound_broadcast_assignment(&format!("m{i}"), i))
            .collect();
        let result = recognise_broadcast(&assignments);
        assert_eq!(
            result.writes.len(),
            2,
            "Should have 2 write ports (A and B)"
        );
        let write_a = result
            .writes
            .iter()
            .find(|w| w.dest_property == "--addrDestA")
            .expect("Should have --addrDestA");
        assert_eq!(write_a.address_map.len(), 20);
        assert!(!write_a.spillover_map.is_empty());
        assert_eq!(
            write_a.spillover_guard.as_deref(),
            Some("--isWordWrite"),
            "Spillover should be gated by --isWordWrite"
        );
    }

    #[test]
    fn side_channel_not_absorbed() {
        // SP has a side channel: else: calc(var(--__1SP) + var(--moveStack))
        let sp_assignment = Assignment {
            property: "--SP".to_string(),
            value: Expr::StyleCondition {
                branches: vec![StyleBranch {
                    condition: StyleTest::Single {
                        property: "--addrDestA".to_string(),
                        value: Expr::Literal(-5.0),
                    },
                    then: Expr::Var {
                        name: "--addrValA".to_string(),
                        fallback: None,
                    },
                }],
                fallback: Box::new(Expr::Calc(CalcOp::Add(
                    Box::new(Expr::Var {
                        name: "--__1SP".to_string(),
                        fallback: None,
                    }),
                    Box::new(Expr::Var {
                        name: "--moveStack".to_string(),
                        fallback: None,
                    }),
                ))),
            },
        };
        let mut assignments: Vec<Assignment> = (0..20)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();
        assignments.push(sp_assignment);
        let result = recognise_broadcast(&assignments);
        assert!(!result.absorbed_properties.contains("--SP"));
    }

    #[test]
    fn too_few_entries_not_recognised() {
        let assignments: Vec<Assignment> = (0..5)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();
        let result = recognise_broadcast(&assignments);
        assert!(result.writes.is_empty());
    }

    /// Build a v2-style memory cell: 3 write ports with --memAddr0/1/2.
    /// `--mN: if(style(--memAddr0:N):var(--memVal0); style(--memAddr1:N):var(--memVal1); style(--memAddr2:N):var(--memVal2); else:var(--__1mN))`
    fn make_v2_broadcast_assignment(name: &str, addr: i64) -> Assignment {
        Assignment {
            property: format!("--{name}"),
            value: Expr::StyleCondition {
                branches: vec![
                    StyleBranch {
                        condition: StyleTest::Single {
                            property: "--memAddr0".to_string(),
                            value: Expr::Literal(addr as f64),
                        },
                        then: Expr::Var {
                            name: "--memVal0".to_string(),
                            fallback: None,
                        },
                    },
                    StyleBranch {
                        condition: StyleTest::Single {
                            property: "--memAddr1".to_string(),
                            value: Expr::Literal(addr as f64),
                        },
                        then: Expr::Var {
                            name: "--memVal1".to_string(),
                            fallback: None,
                        },
                    },
                    StyleBranch {
                        condition: StyleTest::Single {
                            property: "--memAddr2".to_string(),
                            value: Expr::Literal(addr as f64),
                        },
                        then: Expr::Var {
                            name: "--memVal2".to_string(),
                            fallback: None,
                        },
                    },
                ],
                fallback: Box::new(Expr::Var {
                    name: format!("--__1{name}"),
                    fallback: None,
                }),
            },
        }
    }

    #[test]
    fn detects_v2_broadcast() {
        let assignments: Vec<Assignment> = (0..20)
            .map(|i| make_v2_broadcast_assignment(&format!("m{i}"), i))
            .collect();
        let result = recognise_broadcast(&assignments);
        assert_eq!(result.writes.len(), 3, "Should have 3 write ports (memAddr0/1/2)");
        for port_name in &["--memAddr0", "--memAddr1", "--memAddr2"] {
            let write = result
                .writes
                .iter()
                .find(|w| w.dest_property == *port_name)
                .unwrap_or_else(|| panic!("Should have {port_name}"));
            assert_eq!(write.address_map.len(), 20);
        }
        assert_eq!(result.absorbed_properties.len(), 20);
    }

    #[test]
    fn opcode_dispatch_not_absorbed_as_broadcast() {
        // Simulate a register dispatch: --nextAX: if(style(--opcode: 0): calc(1); style(--opcode: 1): calc(2); ...)
        // All branches from the same assignment, different value exprs → should NOT be a broadcast.
        let assignments: Vec<Assignment> = vec![Assignment {
            property: "--nextAX".to_string(),
            value: Expr::StyleCondition {
                branches: (0..20)
                    .map(|i| StyleBranch {
                        condition: StyleTest::Single {
                            property: "--opcode".to_string(),
                            value: Expr::Literal(i as f64),
                        },
                        then: Expr::Literal(i as f64 + 100.0),
                    })
                    .collect(),
                fallback: Box::new(Expr::Var {
                    name: "--__1nextAX".to_string(),
                    fallback: None,
                }),
            },
        }];
        let result = recognise_broadcast(&assignments);
        assert!(
            result.writes.is_empty(),
            "Opcode dispatch should not be recognised as broadcast"
        );
        assert!(result.absorbed_properties.is_empty());
    }
}
