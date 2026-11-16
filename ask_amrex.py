"""
Step 3 of the AMReX RAG pipeline: ask a question.

Examples:
  python ask_amrex.py "fill a MultiFab with a ParallelFor loop"
  python ask_amrex.py "fill a MultiFab with a ParallelFor loop" --no-rag
  python ask_amrex.py "fill a MultiFab with a ParallelFor loop" --model qwen2.5-coder:1.5b --k 3

--no-rag sends the question to the model without any retrieved code.
Use it to get the baseline you compare against.
"""
import argparse
import sys

import chromadb
import requests

from index_amrex import COLLECTION, DB_PATH, OLLAMA, embed

PROMPT_CHARS = 3000  # max characters of each retrieved snippet sent to the model

SYSTEM = (
    "You are an expert in AMReX (the C++ adaptive mesh refinement framework). "
    "Write correct, compilable C++17. Use the loop style shown in the snippets; "
    "otherwise use amrex::ParallelFor with AMREX_GPU_DEVICE lambdas so the code "
    "runs on CPU and GPU. When reference snippets are given, use the exact AMReX "
    "names and signatures from them. If a snippet shows an existing AMReX "
    "function that already does the job (for example MultiFab::Copy), call that "
    "function instead of writing your own loop. Copy formulas from snippets "
    "exactly and do not change them. Only #include headers that appear in the "
    "snippets or that you are sure exist. If a part of your answer is not "
    "covered by the snippets, say that it comes from general knowledge. Give the "
    "code first, then at most three short sentences of explanation."
)



def retrieve(question, k):
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
    vector = embed(["search_query: " + question])[0]
    result = collection.query(query_embeddings=[vector], n_results=k * 6)
    hits, seen = [], set()
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        key = (meta["file"], meta["symbol"])
        if key in seen:
            continue
        seen.add(key)
        hits.append({"text": doc, "meta": meta, "distance": dist})
        if len(hits) == k:
            break
    return hits


def build_user_message(question, hits):
    if not hits:
        return question
    parts = ["REFERENCE SNIPPETS FROM THE AMREX SOURCE:"]
    for number, hit in enumerate(hits, start=1):
        m = hit["meta"]
        parts.append(
            "--- [%d] %s lines %s-%s (%s)\n%s"
            % (number, m["file"], m["start_line"], m["end_line"], m["symbol"], hit["text"][:PROMPT_CHARS])
        )
    parts.append("USER REQUEST:\n" + question)
    return "\n\n".join(parts)


def chat(model, user_message):
    try:
        r = requests.post(
            OLLAMA + "/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1024},
            },
            timeout=600,
        )
    except requests.exceptions.Timeout:
        return "[no answer: the model did not finish within 600 seconds]"
    if r.status_code != 200:
        raise RuntimeError("Ollama said: " + r.text)
    return r.json()["message"]["content"]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()

    hits = [] if args.no_rag else retrieve(args.question, args.k)
    answer = chat(args.model, build_user_message(args.question, hits))

    print(answer)
    print()
    if hits:
        print("Sources used (smaller distance = closer match):")
        for hit in hits:
            m = hit["meta"]
            print("  %s lines %s-%s | %s | distance %.3f"
                  % (m["file"], m["start_line"], m["end_line"], m["symbol"], hit["distance"]))
    else:
        print("(no retrieval: baseline run)")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Cannot reach Ollama at", OLLAMA, "- is it running? Try: ollama list")
        sys.exit(1)