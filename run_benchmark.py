"""
Run a set of AMReX questions with and without retrieval and save results.md.

Run:  python run_benchmark.py
      python run_benchmark.py --model qwen2.5-coder:7b --k 3 --runs 1

For every answer it also lists the AMReX headers that are #included but do
not exist anywhere in amrex/Src (a sign the model invented them).
It does NOT compile the code. Read the answers yourself.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import requests

from ask_amrex import build_user_message, chat, retrieve

TASKS = [
    "write a function that sets every cell of a MultiFab to i+j+k using amrex::ParallelFor",
    "copy the contents of one MultiFab into another MultiFab",
    "compute dst = a*x + b*y for three MultiFabs dst, x and y",
    "write a 7-point finite-difference Laplacian kernel on an Array4 using amrex::ParallelFor",
    "fill the ghost cells of a MultiFab with a Neumann (zero-gradient) boundary condition",
    "solve a Poisson equation with the AMReX multigrid solver MLMG",
    "convert this plain C++ loop to AMReX so it runs on CPU and GPU: for (int k=0;k<Nz;++k) for (int j=0;j<Ny;++j) for (int i=0;i<Nx;++i) arr[i][j][k] = i + j + k;",
]

MODES = ["with retrieval", "no retrieval"]


def header_names():
    names = set()
    for path in Path("amrex/Src").rglob("*.H"):
        names.add(path.name)
    return names


def missing_includes(answer, known):
    missing = []
    for name in re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', answer):
        base = name.split("/")[-1]
        if base.startswith("AMReX") and base not in known and base not in missing:
            missing.append(base)
    return missing


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    known = header_names()
    if not known:
        print("WARNING: no headers found in amrex/Src - the include check will be empty")
    totals = {mode: [0, 0] for mode in MODES}  # runs, runs with an invented include

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# AMReX RAG benchmark\n\n")
        out.write("Model: %s | k = %d | runs per mode: %d\n\n" % (args.model, args.k, args.runs))
        for number, task in enumerate(TASKS, start=1):
            out.write("## Task %d\n\n%s\n\n" % (number, task))
            for mode in MODES:
                for run in range(1, args.runs + 1):
                    hits = retrieve(task, args.k) if mode == "with retrieval" else []
                    started = time.time()
                    answer = chat(args.model, build_user_message(task, hits))
                    seconds = time.time() - started
                    missing = missing_includes(answer, known)
                    totals[mode][0] += 1
                    if missing:
                        totals[mode][1] += 1
                    out.write("### %s (run %d, %.0f s)\n\n%s\n\n" % (mode, run, seconds, answer))
                    if hits:
                        out.write("Sources:\n\n")
                        for hit in hits:
                            m = hit["meta"]
                            out.write("- %s lines %s-%s | %s | distance %.3f\n"
                                      % (m["file"], m["start_line"], m["end_line"], m["symbol"], hit["distance"]))
                        out.write("\n")
                    out.write("Includes not found in amrex/Src: %s\n\n"
                              % (", ".join(missing) if missing else "none"))
                    out.flush()
                    print("task %d/%d | %s | run %d | %.0f s" % (number, len(TASKS), mode, run, seconds))

        out.write("## Summary\n\n| mode | runs | runs with an invented AMReX include |\n|---|---|---|\n")
        for mode in MODES:
            out.write("| %s | %d | %d |\n" % (mode, totals[mode][0], totals[mode][1]))
        out.write("\nThis table only checks include names. It does not check that the code compiles.\n")

    print("done. results saved in:", args.out)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Cannot reach Ollama - is it running? Try: ollama list")
        sys.exit(1)