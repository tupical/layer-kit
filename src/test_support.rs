//! Shared assertions for layer server tests.

use std::collections::HashSet;

/// Asserts that an MCP tool catalogue is non-empty and exactly covers `methods`.
///
/// `methods` use MCP dispatch names (`torii.ingest_raw`); catalogue entries use
/// tool names (`torii_ingest_raw`). Only the first underscore is the separator.
pub fn assert_catalogue_matches(catalogue: &[serde_json::Value], methods: &[&str]) {
    assert!(!catalogue.is_empty(), "tool catalogue must not be empty");

    let catalogue_methods: HashSet<String> = catalogue
        .iter()
        .map(|tool| {
            tool["name"]
                .as_str()
                .expect("catalogue tool must have a string name")
                .replacen('_', ".", 1)
        })
        .collect();
    let methods: HashSet<&str> = methods.iter().copied().collect();
    let missing: Vec<_> = methods
        .iter()
        .filter(|method| !catalogue_methods.contains(**method))
        .collect();
    let extra: Vec<_> = catalogue_methods
        .iter()
        .filter(|method| !methods.contains(method.as_str()))
        .collect();

    assert!(
        missing.is_empty(),
        "methods missing from catalogue: {missing:?}"
    );
    assert!(
        extra.is_empty(),
        "catalogue methods missing from METHODS: {extra:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_empty_and_both_mismatch_directions() {
        assert!(
            std::panic::catch_unwind(|| assert_catalogue_matches(&[], &["torii.parse"])).is_err()
        );
        assert!(std::panic::catch_unwind(|| {
            assert_catalogue_matches(
                &[json!({"name": "torii_parse"})],
                &["torii.parse", "torii.ingest_raw"],
            )
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            assert_catalogue_matches(
                &[
                    json!({"name": "torii_parse"}),
                    json!({"name": "torii_ingest_raw"}),
                ],
                &["torii.parse"],
            )
        })
        .is_err());
    }
}
