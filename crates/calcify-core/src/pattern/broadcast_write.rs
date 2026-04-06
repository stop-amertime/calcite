//! Broadcast write pattern: `if(style(--dest: {addr}): value; else: keep)` → direct store.
//!
//! Detects: a set of assignments where each has the form:
//!   `--varN: if(style(--addrDest: N): value; else: var(--__1varN))`
//! All checking the same destination property against their own address.
//!
//! Replaces: evaluate `--addrDest` once, write `value` to `state[dest]` directly.

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
    // Group direct ports by dest_property → (address, target_var_name, value_expr)
    let mut direct_groups: HashMap<String, Vec<(i64, String, Expr)>> = HashMap::new();
    // Group spillover ports by dest_property → (source_address, target_var_name, guard_property, value_expr)
    let mut spillover_groups: HashMap<String, Vec<(i64, String, String, Expr)>> = HashMap::new();
    let mut pure_broadcast: HashSet<String> = HashSet::new();

    for assignment in assignments {
        if let Some(ports) = extract_broadcast_ports(assignment) {
            pure_broadcast.insert(assignment.property.clone());
            for port in ports {
                match port {
                    BroadcastPort::Direct {
                        dest_property,
                        address,
                        value_expr,
                    } => {
                        direct_groups.entry(dest_property).or_default().push((
                            address,
                            assignment.property.clone(),
                            value_expr,
                        ));
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

    let mut absorbed_properties = HashSet::new();

    // Only keep groups that are large enough to be a real broadcast pattern
    let writes = direct_groups
        .into_iter()
        .filter(|(_, entries)| entries.len() >= 10)
        .map(|(dest_property, entries)| {
            let value_expr = entries
                .first()
                .map(|(_, _, expr)| expr.clone())
                .unwrap_or(Expr::Literal(0.0));
            for (_, name, _) in &entries {
                absorbed_properties.insert(name.clone());
            }
            let address_map = entries
                .into_iter()
                .map(|(addr, name, _)| (addr, name))
                .collect();

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

            BroadcastWrite {
                dest_property,
                value_expr,
                address_map,
                spillover_map,
                spillover_guard,
            }
        })
        .collect();

    // Only absorb properties that are pure broadcast targets
    absorbed_properties.retain(|p| pure_broadcast.contains(p));

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
/// - Any branch tests a non-`--addrDest*` property (execution logic mixed in)
/// - The fallback (else branch) is not a simple keep of the previous value
///
/// Registers like SP, SI, DI have side-channel deltas in their else branches
/// (e.g., `else: calc(var(--__1SP) + var(--moveStack))`) and must NOT be absorbed.
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
                    StyleTest::Single { property, value } => {
                        if !property.starts_with("--addrDest") {
                            return None;
                        }
                        match value {
                            Expr::Literal(v) => {
                                ports.push(BroadcastPort::Direct {
                                    dest_property: property.clone(),
                                    address: *v as i64,
                                    value_expr: branch.then.clone(),
                                });
                            }
                            _ => return None,
                        }
                    }
                    StyleTest::And(tests) if tests.len() == 2 => {
                        // Match: style(--addrDestX: N) and style(--isWordWrite: 1)
                        let (mut addr_test, mut guard_test) = (None, None);
                        for t in tests {
                            if let StyleTest::Single { property, value } = t {
                                if property.starts_with("--addrDest") {
                                    if let Expr::Literal(v) = value {
                                        addr_test = Some((property.clone(), *v as i64));
                                    }
                                } else {
                                    // Guard condition (e.g., --isWordWrite: 1)
                                    if let Expr::Literal(v) = value {
                                        if *v as i64 == 1 {
                                            guard_test = Some(property.clone());
                                        }
                                    }
                                }
                            }
                        }
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
                name: "--addrValA1".to_string(),
                fallback: None,
            },
        }];
        // Spillover branch: only for addr > 0
        if addr > 0 {
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
                    name: "--addrValA2".to_string(),
                    fallback: None,
                },
            });
        }
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
    fn recognises_broadcast_pattern() {
        let assignments: Vec<_> = (0..20)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();

        let result = recognise_broadcast(&assignments);
        assert_eq!(result.writes.len(), 1);
        assert_eq!(result.writes[0].dest_property, "--addrDestA");
        assert_eq!(result.writes[0].address_map.len(), 20);
        assert_eq!(result.absorbed_properties.len(), 20);
        assert!(result.absorbed_properties.contains("--m0"));
        assert!(result.absorbed_properties.contains("--m19"));
    }

    #[test]
    fn ignores_small_groups() {
        let assignments: Vec<_> = (0..3)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();

        let result = recognise_broadcast(&assignments);
        assert!(result.writes.is_empty());
        assert!(result.absorbed_properties.is_empty());
    }

    #[test]
    fn recognises_compound_broadcast_with_spillover() {
        let assignments: Vec<_> = (0..20)
            .map(|i| make_compound_broadcast_assignment(&format!("m{i}"), i))
            .collect();

        let result = recognise_broadcast(&assignments);
        // Should have writes for both --addrDestA and --addrDestB
        assert_eq!(result.writes.len(), 2);

        let port_a = result
            .writes
            .iter()
            .find(|w| w.dest_property == "--addrDestA")
            .expect("should have port A");
        let port_b = result
            .writes
            .iter()
            .find(|w| w.dest_property == "--addrDestB")
            .expect("should have port B");

        assert_eq!(port_a.address_map.len(), 20);
        assert_eq!(port_b.address_map.len(), 20);

        // Port A should have spillover entries (19 — all except m0)
        assert_eq!(port_a.spillover_map.len(), 19);
        assert_eq!(port_a.spillover_guard, Some("--isWordWrite".to_string()));
        // Spillover at source address 0 targets --m1
        let (var_name, _) = port_a.spillover_map.get(&0).unwrap();
        assert_eq!(var_name, "--m1");

        // Port B should have no spillover
        assert!(port_b.spillover_map.is_empty());

        // All 20 should be absorbed
        assert_eq!(result.absorbed_properties.len(), 20);
    }

    /// Build an assignment with a side-channel delta in the else branch:
    /// `--SP: if(style(--addrDestA:-5):val; else: calc(var(--__1SP) + var(--moveStack)))`
    fn make_side_channel_assignment(name: &str, addr: i64, side_channel: &str) -> Assignment {
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
                fallback: Box::new(Expr::Calc(CalcOp::Add(
                    Box::new(Expr::Var {
                        name: format!("--__1{name}"),
                        fallback: None,
                    }),
                    Box::new(Expr::Var {
                        name: format!("--{side_channel}"),
                        fallback: None,
                    }),
                ))),
            },
        }
    }

    #[test]
    fn excludes_side_channel_registers() {
        // Mix of pure broadcast cells and side-channel registers
        let mut assignments: Vec<_> = (0..20)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();
        // SP has a --moveStack side channel
        assignments.push(make_side_channel_assignment("SP", -5, "moveStack"));

        let result = recognise_broadcast(&assignments);

        // Memory cells should be absorbed
        assert!(result.absorbed_properties.contains("--m0"));
        assert!(result.absorbed_properties.contains("--m19"));

        // SP should NOT be absorbed (has side-channel logic)
        assert!(
            !result.absorbed_properties.contains("--SP"),
            "SP should not be absorbed — it has --moveStack side channel"
        );
    }

    #[test]
    fn is_simple_keep_accepts_buffer_var() {
        // var(--__1AX) is a simple keep for --AX
        assert!(is_simple_keep(
            &Expr::Var {
                name: "--__1AX".to_string(),
                fallback: None,
            },
            "--AX"
        ));
        // var(--__0m5) is a simple keep for --m5
        assert!(is_simple_keep(
            &Expr::Var {
                name: "--__0m5".to_string(),
                fallback: None,
            },
            "--m5"
        ));
    }

    #[test]
    fn is_simple_keep_rejects_calc() {
        // calc(var(--__1SP) + var(--moveStack)) is NOT a simple keep
        assert!(!is_simple_keep(
            &Expr::Calc(CalcOp::Add(
                Box::new(Expr::Var {
                    name: "--__1SP".to_string(),
                    fallback: None,
                }),
                Box::new(Expr::Var {
                    name: "--moveStack".to_string(),
                    fallback: None,
                }),
            )),
            "--SP"
        ));
    }

    #[test]
    fn is_simple_keep_rejects_different_var() {
        // var(--jumpCS) is NOT a simple keep for --CS (it references a different property)
        assert!(!is_simple_keep(
            &Expr::Var {
                name: "--jumpCS".to_string(),
                fallback: None,
            },
            "--CS"
        ));
        // var(--newFlags) is NOT a simple keep for --flags
        assert!(!is_simple_keep(
            &Expr::Var {
                name: "--newFlags".to_string(),
                fallback: None,
            },
            "--flags"
        ));
    }
}
