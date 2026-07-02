//! `where` / `where_document` filter compiler — pure Rust, no PyO3.
//!
//! Faithful port of the compiler that used to live directly in
//! `turbovecdb-py`. Operates on `serde_json::Value` instead of a Python
//! object so it's testable via plain `cargo test` (see `docs/rust-core-split-design.md`).
//! Translates Chroma/Mongo-style `where` / `where_document` dicts into a
//! parameterised SQL fragment `(sql, params)` over the JSON `metadata`
//! column. The JSON path and every operand are returned as *bound
//! parameters* (never interpolated), so arbitrary field names and values
//! cannot inject SQL.

use serde_json::Value;
use std::fmt;

const MAX_DEPTH: i32 = 10;
const MAX_IN_LIST: usize = 900;

/// Filter-compilation failure. The PyO3 adapter maps this to the public
/// `UnsupportedFilterError`.
#[derive(Debug, Clone)]
pub struct FilterError(pub String);

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FilterError {}

fn err(msg: impl Into<String>) -> FilterError {
    FilterError(msg.into())
}

/// Map a comparison operator to its SQL symbol.
fn cmp_symbol(op: &str) -> Option<&'static str> {
    match op {
        "$eq" => Some("="),
        "$ne" => Some("!="),
        "$gt" => Some(">"),
        "$gte" => Some(">="),
        "$lt" => Some("<"),
        "$lte" => Some("<="),
        _ => None,
    }
}

/// Coerce a value that is expected to be a non-empty array into its
/// elements, or `None` if it isn't an array.
fn as_sequence(value: &Value) -> Option<&Vec<Value>> {
    value.as_array()
}

/// LIKE-escape backslash, percent, and underscore (order matters: backslash first).
fn like_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

/// Compile a single `field: value` predicate. Pushes its bound params (path
/// first, then operands) onto `params` and returns the SQL fragment.
fn field_clause(field: &str, value: &Value, params: &mut Vec<Value>) -> Result<String, FilterError> {
    let path = format!("$.{}", field);

    if let Some(obj) = value.as_object() {
        if obj.len() != 1 {
            return Err(err(format!(
                "field '{}' predicate must have exactly one operator",
                field
            )));
        }
        let (op, operand) = obj.iter().next().unwrap();

        if op == "$in" || op == "$nin" {
            let items = as_sequence(operand).filter(|v| !v.is_empty()).ok_or_else(|| {
                err(format!("{} on '{}' requires a non-empty list", op, field))
            })?;
            if items.len() > MAX_IN_LIST {
                return Err(err(format!(
                    "{} on '{}': list length {} exceeds maximum of {}",
                    op,
                    field,
                    items.len(),
                    MAX_IN_LIST
                )));
            }
            let placeholders = vec!["?"; items.len()].join(",");
            let negate = if op == "$nin" { "NOT " } else { "" };
            params.push(Value::String(path));
            for item in items {
                params.push(item.clone());
            }
            return Ok(format!(
                "json_extract(metadata, ?) {}IN ({})",
                negate, placeholders
            ));
        }

        if let Some(sym) = cmp_symbol(op) {
            params.push(Value::String(path));
            params.push(operand.clone());
            return Ok(format!("json_extract(metadata, ?) {} ?", sym));
        }

        return Err(err(format!(
            "unsupported operator '{}' on field '{}'",
            op, field
        )));
    }

    // Bare scalar → equality.
    params.push(Value::String(path));
    params.push(value.clone());
    Ok("json_extract(metadata, ?) = ?".to_string())
}

/// Core recursive compiler for a `where` mapping.
pub fn compile_where(where_obj: &Value, depth: i32) -> Result<(String, Vec<Value>), FilterError> {
    if depth > MAX_DEPTH {
        return Err(err(format!(
            "filter nesting depth exceeds maximum of {}",
            MAX_DEPTH
        )));
    }
    if where_obj.is_null() {
        return Ok((String::new(), Vec::new()));
    }
    let dict = where_obj.as_object().ok_or_else(|| err("where must be a mapping"))?;
    if dict.is_empty() {
        return Ok((String::new(), Vec::new()));
    }

    let has_and = dict.contains_key("$and");
    let has_or = dict.contains_key("$or");
    if has_and || has_or {
        if dict.len() != 1 {
            return Err(err("a logical operator ($and/$or) cannot have sibling keys"));
        }
        let (op, subs) = dict.iter().next().unwrap();
        let sub_items = as_sequence(subs).filter(|v| !v.is_empty()).ok_or_else(|| {
            err(format!("{} requires a non-empty list of clauses", op))
        })?;
        let joiner = if op == "$and" { " AND " } else { " OR " };
        let mut frags: Vec<String> = Vec::new();
        let mut params: Vec<Value> = Vec::new();
        for sub in sub_items {
            let (frag, sub_params) = compile_where(sub, depth + 1)?;
            if !frag.is_empty() {
                frags.push(format!("({})", frag));
                params.extend(sub_params);
            }
        }
        return Ok((frags.join(joiner), params));
    }

    // Reject any other top-level operator ($not, $nor, ...).
    for key in dict.keys() {
        if key.starts_with('$') {
            return Err(err(format!("unsupported top-level operator '{}'", key)));
        }
    }

    let mut frags: Vec<String> = Vec::new();
    let mut params: Vec<Value> = Vec::new();
    for (field, value) in dict.iter() {
        frags.push(field_clause(field, value, &mut params)?);
    }
    Ok((frags.join(" AND "), params))
}

/// Core compiler for a `where_document` mapping (only `$contains`).
pub fn compile_where_document(wd: &Value) -> Result<(String, Vec<Value>), FilterError> {
    if wd.is_null() {
        return Ok((String::new(), Vec::new()));
    }
    let dict = wd.as_object().ok_or_else(|| err("where_document must be a mapping"))?;
    if dict.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    if dict.len() != 1 || !dict.contains_key("$contains") {
        return Err(err("where_document supports only $contains"));
    }
    let needle = dict
        .get("$contains")
        .unwrap()
        .as_str()
        .ok_or_else(|| err("$contains requires a string"))?;
    let params = vec![Value::String(format!("%{}%", like_escape(needle)))];
    Ok(("document LIKE ? ESCAPE '\\'".to_string(), params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_or_null_where_compiles_to_nothing() {
        assert_eq!(compile_where(&Value::Null, 0).unwrap(), (String::new(), vec![]));
        assert_eq!(compile_where(&json!({}), 0).unwrap(), (String::new(), vec![]));
    }

    #[test]
    fn bare_scalar_is_equality() {
        let (sql, params) = compile_where(&json!({"lang": "en"}), 0).unwrap();
        assert_eq!(sql, "json_extract(metadata, ?) = ?");
        assert_eq!(params, vec![json!("$.lang"), json!("en")]);
    }

    #[test]
    fn comparison_operators() {
        for (op, sym) in [("$eq", "="), ("$ne", "!="), ("$gt", ">"), ("$gte", ">="), ("$lt", "<"), ("$lte", "<=")] {
            let (sql, params) = compile_where(&json!({"year": {op: 2020}}), 0).unwrap();
            assert_eq!(sql, format!("json_extract(metadata, ?) {sym} ?"));
            assert_eq!(params, vec![json!("$.year"), json!(2020)]);
        }
    }

    #[test]
    fn in_and_nin() {
        let (sql, params) = compile_where(&json!({"lang": {"$in": ["fr", "la"]}}), 0).unwrap();
        assert_eq!(sql, "json_extract(metadata, ?) IN (?,?)");
        assert_eq!(params, vec![json!("$.lang"), json!("fr"), json!("la")]);

        let (sql, _) = compile_where(&json!({"lang": {"$nin": ["fr"]}}), 0).unwrap();
        assert_eq!(sql, "json_extract(metadata, ?) NOT IN (?)");
    }

    #[test]
    fn in_rejects_empty_list() {
        let e = compile_where(&json!({"lang": {"$in": []}}), 0).unwrap_err();
        assert!(e.0.contains("non-empty list"));
    }

    #[test]
    fn and_or_recurse_and_join() {
        let (sql, params) =
            compile_where(&json!({"$and": [{"lang": "en"}, {"year": {"$gte": 2020}}]}), 0).unwrap();
        assert_eq!(
            sql,
            "(json_extract(metadata, ?) = ?) AND (json_extract(metadata, ?) >= ?)"
        );
        assert_eq!(params.len(), 4);

        let (sql, _) = compile_where(&json!({"$or": [{"a": 1}, {"b": 2}]}), 0).unwrap();
        assert!(sql.contains(" OR "));
    }

    #[test]
    fn logical_operator_rejects_sibling_keys() {
        let e = compile_where(&json!({"$and": [{"a": 1}], "b": 2}), 0).unwrap_err();
        assert!(e.0.contains("sibling keys"));
    }

    #[test]
    fn unsupported_top_level_operator_rejected() {
        let e = compile_where(&json!({"$not": {"a": 1}}), 0).unwrap_err();
        assert!(e.0.contains("unsupported top-level operator"));
    }

    #[test]
    fn unsupported_field_operator_rejected() {
        let e = compile_where(&json!({"a": {"$bogus": 1}}), 0).unwrap_err();
        assert!(e.0.contains("unsupported operator"));
    }

    #[test]
    fn multi_operator_predicate_rejected() {
        let e = compile_where(&json!({"a": {"$gt": 1, "$lt": 5}}), 0).unwrap_err();
        assert!(e.0.contains("exactly one operator"));
    }

    #[test]
    fn nesting_depth_is_bounded() {
        // Build a chain of MAX_DEPTH + 2 nested $and wrappers.
        let mut inner = json!({"a": 1});
        for _ in 0..(MAX_DEPTH as usize + 2) {
            inner = json!({"$and": [inner]});
        }
        let e = compile_where(&inner, 0).unwrap_err();
        assert!(e.0.contains("nesting depth"));
    }

    #[test]
    fn in_list_length_is_bounded() {
        let items: Vec<i64> = (0..(MAX_IN_LIST as i64 + 1)).collect();
        let e = compile_where(&json!({"a": {"$in": items}}), 0).unwrap_err();
        assert!(e.0.contains("exceeds maximum"));
    }

    #[test]
    fn where_must_be_a_mapping() {
        let e = compile_where(&json!([1, 2]), 0).unwrap_err();
        assert!(e.0.contains("must be a mapping"));
    }

    #[test]
    fn where_document_contains() {
        let (sql, params) = compile_where_document(&json!({"$contains": "fox"})).unwrap();
        assert_eq!(sql, "document LIKE ? ESCAPE '\\'");
        assert_eq!(params, vec![json!("%fox%")]);
    }

    #[test]
    fn where_document_escapes_like_wildcards() {
        let (_, params) = compile_where_document(&json!({"$contains": "100%_off\\"})).unwrap();
        assert_eq!(params, vec![json!("%100\\%\\_off\\\\%")]);
    }

    #[test]
    fn where_document_rejects_other_operators() {
        let e = compile_where_document(&json!({"$startswith": "x"})).unwrap_err();
        assert!(e.0.contains("only $contains"));
    }

    #[test]
    fn null_scalar_value_is_valid_equality_operand() {
        let (_, params) = compile_where(&json!({"a": null}), 0).unwrap();
        assert_eq!(params, vec![json!("$.a"), Value::Null]);
    }
}
