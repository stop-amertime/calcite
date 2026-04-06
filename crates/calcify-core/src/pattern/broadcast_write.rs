//! Broadcast write pattern: `if(style(--dest: {addr}): value; else: keep)` → direct store.
//!
//! Detects: a set of assignments where each has the form:
//!   `--varN: if(style(--addrDest: N): value; else: var(--__1varN))`
//! All checking the same destination property against their own address.
//!
//! Replaces: evaluate `--addrDest` once, write `value` to `state[dest]` directly.

use crate::types::*;

/// A recognised broadcast write pattern.
#[derive(Debug, Clone)]
pub struct BroadcastWrite {
    /// The destination address property (e.g., `--addrDestA`).
    pub dest_property: String,
    /// The value expression property (e.g., `--addrValA`).
    pub value_expr: Expr,
    /// The address → variable name mapping.
    /// Key: the integer address that each variable checks against.
    /// Value: the variable name that should be written to.
    pub address_map: Vec<(i64, String)>,
}

/// Analyse a set of assignments to detect the broadcast write pattern.
///
/// Returns `Some(BroadcastWrite)` if a group of assignments all follow the pattern:
///   `--var: if(style(--dest: ADDR): val_expr; else: var(--__1var))`
///
/// This is the pattern where all 1,583 state variables each check whether they're
/// the write target. Only 1-2 match per tick.
pub fn recognise_broadcast(assignments: &[Assignment]) -> Vec<BroadcastWrite> {
    use std::collections::HashMap;

    // Group assignments by their broadcast destination property
    // Key: dest_property name, Value: list of (address, target_var_name, value_expr)
    let mut groups: HashMap<String, Vec<(i64, String, Expr)>> = HashMap::new();

    for assignment in assignments {
        if let Some((dest_prop, addr, val_expr)) = extract_broadcast_assignment(assignment) {
            groups.entry(dest_prop).or_default().push((
                addr,
                assignment.property.clone(),
                val_expr,
            ));
        }
    }

    // Only keep groups that are large enough to be a real broadcast pattern
    groups
        .into_iter()
        .filter(|(_, entries)| entries.len() >= 10)
        .map(|(dest_property, entries)| {
            let value_expr = entries
                .first()
                .map(|(_, _, expr)| expr.clone())
                .unwrap_or(Expr::Literal(0.0));
            let address_map = entries
                .into_iter()
                .map(|(addr, name, _)| (addr, name))
                .collect();
            BroadcastWrite {
                dest_property,
                value_expr,
                address_map,
            }
        })
        .collect()
}

/// Extract the broadcast write pattern from a single assignment.
///
/// Pattern: `if(style(--destProp: LITERAL): val_expr; else: var(--__1name))`
///
/// Returns `(dest_property, address, value_expression)` if it matches.
fn extract_broadcast_assignment(assignment: &Assignment) -> Option<(String, i64, Expr)> {
    match &assignment.value {
        Expr::StyleCondition { branches, .. } => {
            let first = branches.first()?;

            // Must be a simple Single test
            match &first.condition {
                StyleTest::Single { property, value } => {
                    if !property.starts_with("--addrDest") && !property.starts_with("--addr") {
                        return None;
                    }
                    match value {
                        Expr::Literal(v) => Some((property.clone(), *v as i64, first.then.clone())),
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
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

    #[test]
    fn recognises_broadcast_pattern() {
        let assignments: Vec<_> = (0..20)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();

        let patterns = recognise_broadcast(&assignments);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].dest_property, "--addrDestA");
        assert_eq!(patterns[0].address_map.len(), 20);
    }

    #[test]
    fn ignores_small_groups() {
        let assignments: Vec<_> = (0..3)
            .map(|i| make_broadcast_assignment(&format!("m{i}"), i))
            .collect();

        let patterns = recognise_broadcast(&assignments);
        assert!(patterns.is_empty());
    }
}
