/// Pinepass is a command line utility meant to assist in upserting embeddings to pinecone db. It
/// comes with two options for length based or charachter based delimination. It will read
/// information from you're environment variables if not passed for configuration of pinecone and
/// openai.
///
/// This is a part of an overarching project to make it easier to upload Obsidian MD data to
/// pinecone and utelize it in openai. More option will be comeing such as deleteing everything
/// within the specific namespace before uploading. I plan on making a website that allows you to
/// easily connect OpenAI with pinecone and use it in you're searchers.

use std::{env, path::Path, sync::Arc, process, fs::{self, File}, io::{BufReader, Read}, time::Duration};

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use openai_api_rs::v1::{api::Client as OClient, embedding::EmbeddingRequest};
use tokio::sync::{Mutex, mpsc};
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

    /// Sets the openai api key for getting embeddings
    #[arg(short, long)]
    openaikey: Option<String>,

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
const OPENAI_KEY_ENV: &str = "OPENAI_API_KEY";
const OPENAI_REQUESTS_PER_MINUTE: f64 = 3000.0;


/// This is the recommended number of vectors that should be in a single upsert (according to
/// pinecone).
const VECTORS_PER_UPSERT: usize = 100;

const METADATA_KEY: &str = "content";

#[tokio::main]
async fn main() {
    let mut args = Args::parse();
    
    args.key = Some(args.key.clone().unwrap_or(env::var(API_KEY_ENV).unwrap()));
    args.environment = Some(args.environment.clone().unwrap_or(env::var(ENVIRONMENT_ENV).unwrap()));
    args.index = Some(args.index.clone().unwrap_or(env::var(INDEX_NAME_ENV).unwrap()));
    args.openaikey = Some(args.openaikey.clone().unwrap_or(env::var(OPENAI_KEY_ENV).unwrap()));

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

    let v = search(args, index).await;
    println!("{} vectors upserted", v);
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

async fn search(args: Args, index: Index) -> usize {
    let a = Arc::new(args);
    let i = Arc::new(index);


    // This asynchronously and recursivly gets the text for the data within the directory
    // that the utility is ran in. It then uploads it to a channel to notify the embedding process
    // to run.
    
    let mut wg = WaitGroup::new();
    let upserts: Arc<Mutex<Vec<Vec<Vector>>>> = Arc::new(Mutex::new(Vec::new()));
    let v: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (txs, mut rxs) = mpsc::channel(1);
    let (txv, mut rxv) = mpsc::channel(1);
    for entry in WalkDir::new(".").into_iter().filter_entry(|e| is_valid(e, &a.ignore, &a.fileextensions)).filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let arg = Arc::clone(&a);
            let worker = wg.worker();
            let my_v = Arc::clone(&v);
            let tx2 = txs.clone();
            tokio::spawn(async move {
                let mut p_vecs = process_file(&arg, entry.path()).await; 
                if let Some(ref mut vecs) = p_vecs {
                    my_v.lock().await.append(vecs);
                    let _ = tx2.send(1).await;
//                    println!("{:?}", entry.path());
                }
                worker.done();
            });
        }
    }


    drop(txs);
    let u = Arc::clone(&upserts);
    let worker = wg.worker();
    let arg = Arc::clone(&a);

    // This runs thorugh every vector recieved asynchronously and gets it's embedding. This is done
    // in a singular task in order synchronize the timing between requests as to not exceed the
    // openai api limit. This then send the embeddings through a channel to be used for pinecone
    // uploads.
    tokio::spawn(async move {
        while rxs.recv().await.is_some() {
            let split: Vec<String>;
            {
                let mut lock = v.lock().await;
                split = lock.clone();
                *lock = Vec::new();
            }
            if split.is_empty() {
                continue;
            }
            for content in split.chunks(VECTORS_PER_UPSERT) {
                let pb = Arc::new(Mutex::new(ProgressBar::new(split.len() as u64)));
                pb.lock().await.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} ({eta})")
                            .unwrap()
                            .progress_chars("#>-"));
                let vecs: Arc<Mutex<Vec<Vector>>> = Arc::new(Mutex::new(Vec::with_capacity(content.len())));
                let mut wg2 = WaitGroup::new();
                for group in content {
                    let g = group.clone();
                    let work2 = wg2.worker();
                    let ve = Arc::clone(&vecs);
                    let key = arg.openaikey.clone();
                    let p = Arc::clone(&pb);
                  //  println!("{}", i);
                    tokio::spawn(async move {
                        let client = OClient::new(key.unwrap());
                        let req = EmbeddingRequest{
                            model: "text-embedding-ada-002".to_string(),
                            input: g.to_string(),
                            user: None
                        };
                        if let Ok(p_emb) = client.embedding(req).await {
                            if let Some(emb) = p_emb.data.get(0) {
                                let mut lock = ve.lock().await;
                                lock.push(Vector{
                                    id: Uuid::new_v4().to_string(),
                                    values: emb.embedding.clone(), 
                                    sparse_values: None,
                                    metadata: Some(
                                        MappedValue::from([(METADATA_KEY.to_string(), serde_json::Value::String(g.to_string()))])
                                    )
                                });
                            } 
                        }
                        p.lock().await.inc(1);
                        work2.done()
                    });
                    tokio::time::sleep(Duration::from_nanos(((10000000.0)/OPENAI_REQUESTS_PER_MINUTE).round() as u64)).await;
                }
                wg2.wait().await;
               // println!("clean {}", i);
                pb.lock().await.finish_and_clear();
                let mut lock = vecs.lock().await;
                u.lock().await.push((&mut lock).to_vec());
                let _ = txv.send(1).await;
            }
        }
        worker.done();
    });

    // Reads the embedding and uploads it to pinecone, it waits until all of them are finished.
    let mut vec_count = 0;
    while rxv.recv().await.is_some() {
        let arg = Arc::clone(&a);
        let ups: Vec<Vec<Vector>>;
        {
            let mut lock = upserts.lock().await;
            ups = lock.clone();
            *lock = Vec::new();
        }
        for x in ups.iter() {
            vec_count += x.len()
        }
        let index = Arc::clone(&i);
        let worker2 = wg.worker();
        tokio::spawn(async move {
            for vec_list in ups {
                if let Err(e) = index.upsert(arg.namespace.clone(), vec_list).await {
                    eprintln!("Failed to upload to index: {:?}", e);
                }
            }
            worker2.done();
        });
    }
    wg.wait().await;
    vec_count
}

async fn process_file(args: &Args, path: impl AsRef<Path>) -> Option<Vec<String>> {
    let split_err = match args.delimiter {
        Some(_) => split_file_by_delimiter(args, path),
        None => split_file_by_length(args, path)
    };

    match split_err {
        Ok(v) => Some(v),
        Err(_) => None
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
        vec.push(String::from_utf8_lossy(&buf).to_string());
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
