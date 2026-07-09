use std::path::PathBuf;
use std::process::Command;

fn binary_path() -> PathBuf {
    std::env::var("CARGO_BIN_EXE_TURBOVECDB_CLI").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("target")
            .join("debug")
            .join("turbovecdb-cli")
    })
}

fn unique_db() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("turbovecdb_test_{}", nanos));
    std::fs::create_dir_all(&dir).ok();
    dir.join("test.db").to_str().unwrap().to_string()
}

fn run(args: &[&str]) -> String {
    let output = Command::new(binary_path())
        .args(args)
        .output()
        .expect("failed to execute turbovecdb-cli");
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

#[test]
fn test_create_add_twice_and_count() {
    let db = unique_db();

    run(&["--db", &db, "add", "--collection", "test", "--id", "a", "--vector", "1,0,0,0,0,0,0,0"]);
    run(&["--db", &db, "add", "--collection", "test", "--id", "b", "--vector", "0,1,0,0,0,0,0,0"]);

    let out = run(&["--db", &db, "count", "--collection", "test"]);
    assert_eq!(out, r#"{"count": 2}"#);
}

#[test]
fn test_query_returns_results() {
    let db = unique_db();

    run(&["--db", &db, "add", "--collection", "test", "--id", "cat", "--vector", "1,0,0,0,0,0,0,0"]);
    run(&["--db", &db, "add", "--collection", "test", "--id", "dog", "--vector", "0,1,0,0,0,0,0,0"]);
    run(&["--db", &db, "add", "--collection", "test", "--id", "bird", "--vector", "0,0,1,0,0,0,0,0"]);

    let out = run(&["--db", &db, "query", "--collection", "test", "--vector", "0.9,0.1,0,0,0,0,0,0", "--k", "2"]);

    let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
    let results = parsed.as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["id"], "cat");
}

#[test]
fn test_cache_works_second_add_does_not_require_reinit() {
    let db = unique_db();

    run(&["--db", &db, "add", "--collection", "test", "--id", "x", "--vector", "1,0,0,0,0,0,0,0"]);
    run(&["--db", &db, "add", "--collection", "test", "--id", "y", "--vector", "0,1,0,0,0,0,0,0"]);

    let out = run(&["--db", &db, "count", "--collection", "test"]);
    assert_eq!(out, r#"{"count": 2}"#);
}
