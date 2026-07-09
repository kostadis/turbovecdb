use turbovecdb_core::database::CachedDatabase;
use turbovecdb_core::embedder::NoEmbedder;
use turbovecdb_core::index::TurbovecIndex;

type Db = CachedDatabase<NoEmbedder, TurbovecIndex>;

fn get_flag(args: &[String], flag: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == flag)?;
    args.get(pos + 1).cloned()
}

fn parse_vector(s: &str) -> Vec<f32> {
    s.split(',')
        .map(|x| x.trim().parse().expect("invalid float in --vector"))
        .collect()
}

fn result_to_json(r: &turbovecdb_core::types::QueryResult) -> serde_json::Value {
    let mut out = Vec::new();
    for i in 0..r.ids.len() {
        let mut obj = serde_json::Map::new();
        obj.insert("id".into(), serde_json::Value::String(r.ids[i].clone()));
        if i < r.distances.len() {
            obj.insert("score".into(), serde_json::json!(r.distances[i]));
        }
        if i < r.documents.len() {
            obj.insert("document".into(), serde_json::Value::String(r.documents[i].clone()));
        }
        if i < r.metadatas.len() {
            obj.insert("metadata".into(), serde_json::Value::String(r.metadatas[i].clone()));
        }
        if let Some(vecs) = &r.vectors {
            if i < vecs.len() {
                obj.insert("vector".into(), serde_json::json!(vecs[i]));
            }
        }
        out.push(serde_json::Value::Object(obj));
    }
    serde_json::Value::Array(out)
}

fn cmd_add(db: &Db, args: &[String]) {
    let name = get_flag(args, "--collection").expect("--collection required");
    let id = get_flag(args, "--id").expect("--id required");
    let vector_str = get_flag(args, "--vector").expect("--vector required");
    let metadata = get_flag(args, "--metadata");

    let raw = parse_vector(&vector_str);
    let dim = raw.len();
    let vectors = ndarray::Array2::from_shape_vec((1, dim), raw).unwrap();

    let handle = db.collection(&name, Some(dim as i64), 4, Some("cosine".into()), None, 5.0).unwrap();
    let mut coll = handle.lock().unwrap();

    let metadatas = metadata.map(|m| vec![m]);
    coll.add(vec![id.clone()], None, metadatas, Some(vectors)).unwrap();

    println!("{{\"ok\": true, \"id\": \"{}\"}}", id);
}

fn cmd_query(db: &Db, args: &[String]) {
    let name = get_flag(args, "--collection").expect("--collection required");
    let vector_str = get_flag(args, "--vector").expect("--vector required");
    let k: usize = get_flag(args, "--k")
        .expect("--k required")
        .parse()
        .expect("--k must be an integer");

    let raw = parse_vector(&vector_str);
    let dim = raw.len();
    let query_vec = ndarray::Array2::from_shape_vec((1, dim), raw).unwrap();

    let handle = db.collection(&name, None, 4, None, None, 5.0).unwrap();
    let mut coll = handle.lock().unwrap();
    let result = coll.query(None, Some(query_vec), k, None, None, None).unwrap();

    println!("{}", serde_json::to_string(&result_to_json(&result)).unwrap());
}

fn cmd_count(db: &Db, args: &[String]) {
    let name = get_flag(args, "--collection").expect("--collection required");

    let handle = db.collection(&name, None, 4, None, None, 5.0).unwrap();
    let coll = handle.lock().unwrap();
    let n = coll.count().unwrap();

    println!("{{\"count\": {}}}", n);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: turbovecdb-cli --db <path> <add|query|count> ...");
        std::process::exit(1);
    }

    let db_path = get_flag(&args, "--db").expect("--db required");
    let db = Db::new(&db_path);

    let subcmd = if let Some(pos) = args.iter().position(|a| a == "add" || a == "query" || a == "count") {
        args[pos].clone()
    } else {
        eprintln!("Usage: turbovecdb-cli --db <path> <add|query|count> ...");
        std::process::exit(1);
    };

    match subcmd.as_str() {
        "add" => cmd_add(&db, &args),
        "query" => cmd_query(&db, &args),
        "count" => cmd_count(&db, &args),
        _ => unreachable!(),
    }
}
