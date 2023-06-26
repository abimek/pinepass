Pinepass is a command line library ment to asynchronously upload data to pinecone using ada-002 embeddings from OpenAI.
To run it please just use is as such:

**Warning**
This is probably some of the worst code I've ever written in my life, I have no clue how I even got it to function, and it's most likely not optimal for most applications.

```console
./pinepass -d "##" -f "md"
```

This will upload everything in a directory that ends with the filetype .md spliting up everything by the string "##" / H2. For more settings please run
```console
./pinepass --help
```

