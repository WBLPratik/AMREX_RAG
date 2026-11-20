"""
Show what the search finds for a question, without calling the language model.

Run:  python search_amrex.py "your question"
      python search_amrex.py "your question" --k 10 --text
"""
import argparse
import sys

import requests

from ask_amrex import retrieve


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--text", action="store_true", help="also print the first lines of each chunk")
    args = ap.parse_args()

    for number, hit in enumerate(retrieve(args.question, args.k), start=1):
        m = hit["meta"]
        print("%d. %s lines %s-%s | %s | distance %.3f"
              % (number, m["file"], m["start_line"], m["end_line"], m["symbol"], hit["distance"]))
        if args.text:
            for line in hit["text"].splitlines()[:6]:
                print("      " + line)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Cannot reach Ollama - is it running? Try: ollama list")
        sys.exit(1)