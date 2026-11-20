"""
Measure how often the search finds the right AMReX code (hit@5).

For each of 15 questions it searches three ways:
  plain             - the question as a beginner would type it
  rewritten         - the question rewritten into AMReX words by the local model
  plain + rewritten - both together
A question counts as a hit if the name or file of one of the top 5 results
contains one of the expected words.

Run:  python retrieval_test.py
      python retrieval_test.py --model qwen2.5-coder:7b
Out:  retrieval_results.md
"""
import argparse
import sys

import requests

from ask_amrex import retrieve
from index_amrex import OLLAMA

# (question, words that should appear in the name or file of a correct result)
QUESTIONS = [
    ("copy the contents of one MultiFab into another MultiFab", ["MultiFab::Copy"]),
    ("compute dst = a*x + b*y for three MultiFabs", ["LinComb"]),
    ("fill the ghost cells of a MultiFab with a Neumann zero-gradient boundary condition", ["FillDomainBoundary", "AMReX_BCUtil.H"]),
    ("set every cell of a MultiFab to a constant value", ["setVal"]),
    ("add one MultiFab to another", ["MultiFab::Add"]),
    ("find the largest absolute value in a MultiFab", ["norm0"]),
    ("sum all the values in a MultiFab", ["MultiFab::sum"]),
    ("exchange ghost cell data between neighbouring boxes", ["FillBoundary"]),
    ("copy data between two MultiFabs that have different box layouts", ["ParallelCopy"]),
    ("read a number from the inputs file", ["ParmParse"]),
    ("start up and shut down AMReX in main", ["Initialize", "Finalize"]),
    ("write a plotfile of a MultiFab", ["Plotfile"]),
    ("solve a Poisson equation with the multigrid solver", ["CellPoisson", "MLMG", "MLPoisson"]),
    ("write a 7-point Laplacian on an Array4", ["laplacian"]),
    ("convert a plain C++ triple loop into a loop that runs on CPU and GPU", ["ParallelFor"]),
]

REWRITE_SYSTEM = (
    "You turn a beginner's question into a short search query for the AMReX C++ "
    "library. Use AMReX class and function names where you are confident, such as "
    "MultiFab, MFIter, Array4, Box, ParallelFor, ParmParse, FillBoundary. "
    "Output only the query on one line, at most 25 words. Do not write code."
)

MODES = ["plain", "rewritten", "plain + rewritten"]


def rewrite(model, question):
    try:
        r = requests.post(
            OLLAMA + "/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": question},
                ],
                "options": {"temperature": 0, "num_predict": 80},
            },
            timeout=180,
        )
        if r.status_code != 200:
            return question
        text = r.json()["message"]["content"].strip().splitlines()
        return text[0].strip().strip("`\"'") if text else question
    except requests.exceptions.Timeout:
        return question


def first_hit_rank(hits, expected):
    for rank, hit in enumerate(hits, start=1):
        label = (hit["meta"]["symbol"] + " " + hit["meta"]["file"]).lower()
        if any(word.lower() in label for word in expected):
            return rank
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="retrieval_results.md")
    args = ap.parse_args()

    score = {mode: 0 for mode in MODES}
    rows = []
    for number, (question, expected) in enumerate(QUESTIONS, start=1):
        rewritten = rewrite(args.model, question)
        queries = {
            "plain": question,
            "rewritten": rewritten,
            "plain + rewritten": question + " " + rewritten,
        }
        ranks = {}
        for mode in MODES:
            rank = first_hit_rank(retrieve(queries[mode], args.k), expected)
            ranks[mode] = rank
            if rank is not None:
                score[mode] += 1
        rows.append((number, question, expected, rewritten, ranks))
        print("question %d/%d | plain %s | rewritten %s | both %s"
              % (number, len(QUESTIONS),
                 ranks["plain"] or "-", ranks["rewritten"] or "-", ranks["plain + rewritten"] or "-"))

    total = len(QUESTIONS)
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# Retrieval test (hit@%d)\n\nModel used to rewrite: %s\n\n" % (args.k, args.model))
        out.write("## Summary\n\n| search | questions answered correctly in the top %d |\n|---|---|\n" % args.k)
        for mode in MODES:
            out.write("| %s | %d / %d |\n" % (mode, score[mode], total))
        out.write("\n## Per question (number = position of the first correct result, - = not found)\n\n")
        out.write("| # | question | expected | plain | rewritten | both |\n|---|---|---|---|---|---|\n")
        for number, question, expected, rewritten, ranks in rows:
            out.write("| %d | %s | %s | %s | %s | %s |\n"
                      % (number, question, ", ".join(expected),
                         ranks["plain"] or "-", ranks["rewritten"] or "-", ranks["plain + rewritten"] or "-"))
        out.write("\n## Rewritten queries\n\n")
        for number, question, expected, rewritten, ranks in rows:
            out.write("%d. %s\n" % (number, rewritten))

    print()
    for mode in MODES:
        print("%-18s %d / %d" % (mode, score[mode], total))
    print("saved:", args.out)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Cannot reach Ollama - is it running? Try: ollama list")
        sys.exit(1)