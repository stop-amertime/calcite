use std::fs;

fn main() {
    env_logger::init();
    let css = fs::read_to_string("tests/fixtures/x86css-computational.css")
        .expect("should read CSS file");

    println!(
        "CSS size: {} bytes, {} lines",
        css.len(),
        css.lines().count()
    );

    let parsed = calcify_core::parser::parse_css(&css).expect("should parse");

    println!("=== Parse Results ===");
    println!("@property declarations: {}", parsed.properties.len());
    println!("@function definitions:  {}", parsed.functions.len());
    println!("Property assignments:   {}", parsed.assignments.len());

    println!("\n=== Functions ===");
    for f in &parsed.functions {
        let result_summary = format!("{:?}", f.result);
        let result_short: String = result_summary.chars().take(100).collect();
        println!(
            "  {} ({} params, {} locals) → {}{}",
            f.name,
            f.parameters.len(),
            f.locals.len(),
            result_short,
            if result_summary.len() > 100 {
                "..."
            } else {
                ""
            }
        );
    }

    // Count expression types in assignments
    let mut style_cond_count = 0;
    let mut var_count = 0;
    let mut literal_count = 0;
    let mut calc_count = 0;
    let mut func_call_count = 0;

    for a in &parsed.assignments {
        match &a.value {
            calcify_core::types::Expr::StyleCondition { branches, .. } => {
                style_cond_count += 1;
                if branches.len() > 100 {
                    println!(
                        "\n  Large if() in {}: {} branches",
                        a.property,
                        branches.len()
                    );
                }
            }
            calcify_core::types::Expr::StringLiteral(_) => literal_count += 1,
            calcify_core::types::Expr::Var { .. } => var_count += 1,
            calcify_core::types::Expr::Literal(_) => literal_count += 1,
            calcify_core::types::Expr::Calc(_) => calc_count += 1,
            calcify_core::types::Expr::FunctionCall { .. } => func_call_count += 1,
        }
    }

    println!("\n=== Assignment Expression Types ===");
    println!("  StyleCondition (if): {style_cond_count}");
    println!("  Var:                 {var_count}");
    println!("  Literal:             {literal_count}");
    println!("  Calc:                {calc_count}");
    println!("  FunctionCall:        {func_call_count}");

    // Try building an evaluator to see pattern recognition
    let evaluator = calcify_core::Evaluator::from_parsed(&parsed);
    println!("\n=== Pattern Recognition ===");
    println!("Dispatch tables found: {}", evaluator.dispatch_tables.len());
    for (name, table) in &evaluator.dispatch_tables {
        println!(
            "  {} → {} entries on '{}'",
            name,
            table.entries.len(),
            table.key_property
        );
    }
    println!(
        "Broadcast writes found: {}",
        evaluator.broadcast_writes.len()
    );
    for bw in &evaluator.broadcast_writes {
        println!("  {} → {} targets", bw.dest_property, bw.address_map.len());
    }
}
