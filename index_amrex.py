"""
Step 2 of the AMReX RAG pipeline: embed every chunk and store it in ChromaDB.

Needs: Ollama running, and the model nomic-embed-text pulled.
Run:   python index_amrex.py
Out:   folder chroma_db_amrex  (the vector index)

Running it again deletes the old index and builds a new one.
"""
import json
import sys
import time

import chromadb
import requests

# ----- settings you may change -----
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHUNKS_FILE = "chunks.jsonl"
DB_PATH = "chroma_db_amrex"
COLLECTION = "amrex"
AMREX_VERSION = "373a152"   # put the commit here, e.g. "a1b2c3d"
BATCH = 16
EMBED_CHARS = 2500           # only the start of a long chunk is embedded
# -----------------------------------


def embed(texts):
    """Turn a list of texts into a list of vectors using Ollama."""
    r = requests.post(
        OLLAMA + "/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    if r.status_code == 404:  # older Ollama versions use another endpoint
        vectors = []
        for text in texts:
            rr = requests.post(
                OLLAMA + "/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=300,
            )
            if rr.status_code != 200:
                raise RuntimeError("Ollama said: " + rr.text)
            vectors.append(rr.json()["embedding"])
        return vectors
    if r.status_code != 200:
        raise RuntimeError("Ollama said: " + r.text)
    return r.json()["embeddings"]


def main():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print("chunks to embed:", len(chunks))

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "amrex_version": AMREX_VERSION},
    )

    started = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [
            "search_document: " + c["symbol"] + " (" + c["file"] + ")\n" + c["text"][:EMBED_CHARS]
            for c in batch
        ]
        vectors = embed(texts)
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "file": c["file"],
                    "symbol": c["symbol"],
                    "kind": c["kind"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                }
                for c in batch
            ],
            embeddings=vectors,
        )
        done = min(i + BATCH, len(chunks))
        print("embedded", done, "/", len(chunks), "(%d s)" % (time.time() - started))

    print("done. index stored in:", DB_PATH)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Cannot reach Ollama at", OLLAMA, "- is it running? Try: ollama list")
        sys.exit(1)