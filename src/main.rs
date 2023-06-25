/// Pinepass is a command line utility meant to assist in upserting embeddings to pinecone db. It
/// comes with two options for length based or charachter based delimination. It will read
/// information from you're environment variables if not passed for configuration of pinecone and
/// openai.
///
/// This is a part of an overarching project to make it easier to upload Obsidian MD data to
/// pinecone and utelize it in openai. More option will be comeing such as deleteing everything
/// within the specific namespace before uploading. I plan on making a website that allows you to
/// easily connect OpenAI with pinecone and use it in you're searchers.

use std::{env, path::Path, sync::Arc, process, fs::{self, File}, io::{BufReader, Read}};

/// pinepass fill 

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::Mutex;
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

    if args.fileextensions.is_empty() {
        eprintln!("Please select atleast one file extension to search for");
        process::exit(1);
    }
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

    let (file_count, vector_count) = search(args, index).await;
    eprintln!("Pinepass Success! {} files and {} vectors handeled and upserted", file_count, vector_count);

}

fn is_valid(entry: &DirEntry, ignore: &Option<Vec<String>>, extensions: &[String]) -> bool {
    let dir = entry.path().to_str().unwrap();
    if let Some(ignore_paths) = ignore {
        if entry.path().is_dir() && ignore_paths.iter().any(|e| dir.starts_with(e)) {
            return false;
        }
    }
    if !entry.path().is_dir() && extensions.iter().any(|e| !dir.ends_with(e)) {
        return false;
    }
    true
}

async fn search(args: Args, index: Index) -> (usize, usize) {
    let a = Arc::new(args);
    let i = Arc::new(index);

    let mut files_operated_on: usize = 0;
    let vectors_uploaded = Arc::new(Mutex::new(0));

    let mut wg = WaitGroup::new();
    for entry in WalkDir::new(".").into_iter().filter_entry(|e| is_valid(e, &a.ignore, &a.fileextensions)).filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let arg = Arc::clone(&a);
            let ind = Arc::clone(&i);
            let mutex = Arc::clone(&vectors_uploaded);
            let worker = wg.worker();
            files_operated_on += 1;
            tokio::spawn(async move {
                println!("{:?}", entry.path());
                let c = handle_file(&arg, &ind, entry.path(), entry.path()).await; worker.done();
                let mut lock = mutex.lock().await;
                *lock += c;
            });
        }
    }
    wg.wait().await;
    let x = (files_operated_on, vectors_uploaded.lock().await.to_owned());
    x
}

async fn handle_file(args: &Args, index: &Index, path: impl AsRef<Path>, path2: impl AsRef<Path>) -> usize {
    let split_err = match args.delimiter {
        Some(_) => split_file_by_delimiter(args, path),
        None => split_file_by_length(args, path)
    };

    let split = match split_err {
        Ok(v) => v,
        Err(_) => {
            return 0
        }
    };

    let pb = ProgressBar::new(split.len() as u64);
    let mut count =  0;
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {vec_count}/{total_vectors} ({eta})")
            .unwrap()
    //        .with_key("", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
            .progress_chars("#>-"));
    for content in split.chunks(VECTORS_PER_UPSERT) {
        let mut vecs: Vec<Vector> = Vec::with_capacity(content.len());
        for group in content {
            count += 1;

            pb.inc(1);

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
    pb.finish_and_clear();
    count
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
        vec.push(String::from_utf8_lossy(&buf).to_string());
        if let Ok(read_bytes) = bytes {
            if read_bytes != 4 * args.length {
                break;
            }
        }
    }
    println!("{:?}", vec);
    Ok(vec)
}

fn split_file_by_delimiter(args: &Args, path: impl AsRef<Path>) -> Result<Vec<String>, ()> {
    if let Ok(file_content) = fs::read_to_string(path) {
        Ok(Vec::from_iter(file_content.split(args.delimiter.clone().unwrap().as_str()).map(String::from)))
    } else {
        Err(())
    }
}
