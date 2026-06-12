//! Dispatch table pattern: large `if(style(--param: N))` chains → array lookup.
//!
//! Detects: an `if()` expression where all branches check the same property
//! against integer literal values.
//!
//! Replaces: linear scan with `table[key]` lookup.

use crate::types::*;

/// Entry map for dispatch tables. FxHashMap rather than std: cabinets
/// produce tables with millions of i64 keys, and SipHash dominates the
/// insert/clone cost at that scale (especially in wasm).
pub type DispatchEntries = rustc_hash::FxHashMap<i64, Expr>;

/// A dispatch table built from a large `if(style())` chain.
///
/// All branches check the same property against integer constants.
/// At runtime: look up `state[property]` in the table to get the result expression.
#[derive(Debug, Clone)]
pub struct DispatchTable {
    /// The property being dispatched on (e.g., `--at` in readMem).
    pub key_property: String,
    /// Map from integer key value → result expression.
    pub entries: DispatchEntries,
    /// Fallback expression when the key doesn't match any entry.
    pub fallback: Expr,
}

/// Validation-only twin of [`recognise_dispatch`]: returns the shared key
/// property if the branches form a dispatch table (≥4 single tests on one
/// property, all against integer literals), without building or cloning
/// anything. Owning callers use this to decide whether to *drain* the
/// branches into a table instead of cloning them.
pub fn recognise_dispatch_key(branches: &[StyleBranch]) -> Option<&str> {
    if branches.len() < 4 {
        return None;
    }
    let StyleTest::Single { property: key, .. } = &branches[0].condition else {
        return None; // Compound conditions can't form a dispatch table
    };
    for branch in branches {
        match &branch.condition {
            StyleTest::Single {
                property,
                value: Expr::Literal(_),
            } if property == key => {}
            _ => return None, // mixed key / non-literal / compound
        }
    }
    Some(key)
}

/// Try to recognise a `StyleCondition` as a dispatch table pattern.
///
/// Returns `Some(DispatchTable)` if:
/// - All branches test the same property
/// - All test values are integer literals
/// - There are enough branches to justify a table (threshold: 4)
pub fn recognise_dispatch(branches: &[StyleBranch], fallback: &Expr) -> Option<DispatchTable> {
    let key_property = recognise_dispatch_key(branches)?.to_string();
    let mut entries =
        DispatchEntries::with_capacity_and_hasher(branches.len(), Default::default());
    for branch in branches {
        if let StyleTest::Single {
            value: Expr::Literal(v),
            ..
        } = &branch.condition
        {
            entries.insert(*v as i64, branch.then.clone());
        }
    }
    Some(DispatchTable {
        key_property,
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
