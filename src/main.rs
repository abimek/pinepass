use std::{env, path::Path, sync::Arc, process, fs::{self, File}, collections::BTreeMap, io::{BufReader, BufRead, Read}};

/// pinepass fill 

use clap::Parser;
use walkdir::{WalkDir, DirEntry};
use pinenut::{models::{Vector, MappedValue}, Client, Index};
use uuid::Uuid;
use awaitgroup::WaitGroup;
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Sets the pinecone api key, if not set it will read from env var "PINECONE_API_KEY"
    #[arg(short, long)]
    key: Option<String>,

    /// Sets the pinecone environment, if not set it will read from env var "PINECONE_ENV"
    #[arg(short, long)]
    environment: Option<String>,

    /// Sets the pinecone index name, if not set it will read from env var "INDEX_NAME_ENV"
    #[arg(short, long)]
    index: Option<String>,

    /// namespace is the pinecone namespace to insert the value into
    #[arg(short, long, default_value="")]
    namespace: String,

    /// String to delimiter string, when given a value length based delimination is not used
    #[arg(short, long)]
    delimiter: Option<String>,

    /// Sets the length delimiter value, this default to character  
    #[arg(short, long, default_value_t=100)]
    length: usize,

    /// file extensions to search for, examples are "md" or "txt", don't include the period
    #[arg(short, long)]
    fileextensions: Vec<String>,

    /// Directories or files to ignore during the scan
    #[arg(long)]
    ignore: Option<Vec<String>>,

}

/// Environment variables that will be read when the values for key, environment, and index are not
/// set
const API_KEY_ENV: &str = "PINECONE_API_KEY";
const ENVIRONMENT_ENV: &str ="PINECONE_ENV";
const INDEX_NAME_ENV: &str = "PINECONE_INDEX_NAME";

/// This is the recommended number of vectors that should be in a single upsert (according to
/// pinecone).
const VECTORS_PER_UPSERT: usize = 1000;

const METADATA_KEY: &str = "content";

#[tokio::main]
async fn main() {
    let mut args = Args::parse();
    
    args.key = Some(args.key.clone().unwrap_or(env::var(API_KEY_ENV).unwrap()));
    args.environment = Some(args.environment.clone().unwrap_or(env::var(ENVIRONMENT_ENV).unwrap()));
    args.index = Some(args.index.clone().unwrap_or(env::var(INDEX_NAME_ENV).unwrap()));

    // We create an instance of client first and firstmost. Panics if it couldn't authenticate.
    let client = Client::new(args.key.clone().unwrap(), args.environment.clone().unwrap())
        .await
        .unwrap();

    // creates an index, will not authenticate.
    let mut index = client.index(args.index.clone().unwrap());

    if let Err(e) = index.describe().await {
        eprintln!("Invalid Pinecone Credentials: {:?}", e);
        process::exit(1);
    }

    search(args, index).await;
}

fn is_valid(entry: &DirEntry, ignore: &Option<Vec<String>>, extensions: &[String]) -> bool {
    if let Some(ignore_paths) = ignore {
        let dir = entry.path().to_str().unwrap();
        if entry.path().is_dir() && ignore_paths.iter().any(|e| dir.starts_with(e)) {
            return false;
        }
        if !entry.path().is_dir() && extensions.iter().any(|e| !dir.ends_with(e)) {
            return false;
        }
    }
    true
}

async fn search(args: Args, index: Index){
    let a = Arc::new(args);
    let i = Arc::new(index);
    let mut wg = WaitGroup::new();
    for entry in WalkDir::new(".").into_iter().filter_entry(|e| is_valid(e, &a.ignore, &a.fileextensions)).filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let arg = Arc::clone(&a);
            let ind = Arc::clone(&i);
            let worker = wg.worker();
            tokio::spawn(async move {
                handle_file(&arg, &ind, entry.path()).await;
                worker.done();
            });
        }
    }
    wg.wait().await;
    println!("Testing");
}

async fn handle_file(args: &Args, index: &Index, path: impl AsRef<Path>){
    let split_err = match args.delimiter {
        Some(_) => split_file_by_delimiter(args, path),
        None => split_file_by_length(args, path)
    };

    let split = match split_err {
        Ok(v) => v,
        Err(_) => {
            return
        }
    } ;

    for content in split.chunks(VECTORS_PER_UPSERT) {
        let mut vecs: Vec<Vector> = Vec::with_capacity(content.len());
        for group in content {
            vecs.push(Vector{
                id: Uuid::new_v4().to_string(),
                values: vec![0.0;1536], //TODO: Implement the OpenAI Embeddings API
                sparse_values: None,
                metadata: Some(
                    MappedValue::from([(METADATA_KEY.to_string(), serde_json::Value::String(group.to_string()))])
                )
            });
        }
        if let Err(e) = index.upsert(args.namespace.clone(), vecs.clone()).await {
            eprintln!("Failed to upload to index: {:?}", e);
        }
    }
}

fn split_file_by_length(args: &Args, path: impl AsRef<Path>) -> Result<Vec<String>, ()> {
    let mut vec: Vec<String> = Vec::new();

    let f = match File::open(path) {
        Ok(file) => file,
        Err(_) => return Err(()),
    };

    let mut reader = BufReader::new(f);
    let mut buf = vec![0u8; args.length * 4];
    loop {
        let bytes = reader.read(&mut buf);
        vec.push(std::str::from_utf8(&buf).unwrap().to_string());
        if let Ok(read_bytes) = bytes {
            if read_bytes != 4 * args.length {
                break;
            }
        }
    }
    Ok(vec)
}

fn split_file_by_delimiter(args: &Args, path: impl AsRef<Path>) -> Result<Vec<String>, ()> {
    if let Ok(file_content) = fs::read_to_string(path) {
        Ok(Vec::from_iter(file_content.split(args.delimiter.clone().unwrap().as_str()).map(String::from)))
    } else {
        Err(())
    }
}
