//! Dispatch table pattern: large `if(style(--param: N))` chains → array lookup.
//!
//! Detects: an `if()` expression where all branches check the same property
//! against integer literal values.
//!
//! Replaces: linear scan with `table[key]` lookup.

use std::collections::HashMap;

use crate::types::*;

/// A dispatch table built from a large `if(style())` chain.
///
/// All branches check the same property against integer constants.
/// At runtime: look up `state[property]` in the table to get the result expression.
#[derive(Debug, Clone)]
pub struct DispatchTable {
    /// The property being dispatched on (e.g., `--at` in readMem).
    pub key_property: String,
    /// Map from integer key value → result expression.
    pub entries: HashMap<i64, Expr>,
    /// Fallback expression when the key doesn't match any entry.
    pub fallback: Expr,
}

/// Try to recognise a `StyleCondition` as a dispatch table pattern.
///
/// Returns `Some(DispatchTable)` if:
/// - All branches test the same property
/// - All test values are integer literals
/// - There are enough branches to justify a table (threshold: 4)
pub fn recognise_dispatch(branches: &[StyleBranch], fallback: &Expr) -> Option<DispatchTable> {
    if branches.len() < 4 {
        return None;
    }

    // Check that all branches are simple Single tests on the same property with integer literals
    let key_property = match &branches[0].condition {
        StyleTest::Single { property, .. } => property,
        _ => return None, // Compound conditions can't form a dispatch table
    };
    let mut entries = HashMap::with_capacity(branches.len());

    for branch in branches {
        match &branch.condition {
            StyleTest::Single { property, value } => {
                if property != key_property {
                    return None; // Different properties — not a dispatch table
                }
                match value {
                    Expr::Literal(v) => {
                        entries.insert(*v as i64, branch.then.clone());
                    }
                    _ => return None, // Non-literal comparison
                }
            }
            _ => return None, // Compound condition — not a dispatch table
        }
    }

    Some(DispatchTable {
        key_property: key_property.clone(),
        entries,
        fallback: fallback.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_branch(prop: &str, val: f64, then: f64) -> StyleBranch {
        StyleBranch {
            condition: StyleTest::Single {
                property: prop.to_string(),
                value: Expr::Literal(val),
            },
            then: Expr::Literal(then),
        }
    }

    #[test]
    fn recognises_dispatch_table() {
        let branches: Vec<_> = (0..10)
            .map(|i| make_branch("--at", i as f64, (i * 100) as f64))
            .collect();
        let fallback = Expr::Literal(0.0);

        let table = recognise_dispatch(&branches, &fallback).unwrap();
        assert_eq!(table.key_property, "--at");
        assert_eq!(table.entries.len(), 10);
        assert!(matches!(table.entries[&5], Expr::Literal(v) if (v - 500.0).abs() < f64::EPSILON));
    }

    #[test]
    fn rejects_mixed_properties() {
        let branches = vec![
            make_branch("--a", 1.0, 10.0),
            make_branch("--b", 2.0, 20.0),
            make_branch("--a", 3.0, 30.0),
            make_branch("--a", 4.0, 40.0),
        ];
        assert!(recognise_dispatch(&branches, &Expr::Literal(0.0)).is_none());
    }

    #[test]
    fn rejects_small_chains() {
        let branches = vec![make_branch("--x", 1.0, 10.0), make_branch("--x", 2.0, 20.0)];
        assert!(recognise_dispatch(&branches, &Expr::Literal(0.0)).is_none());
    }
}
