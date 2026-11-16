"""
Step 1 of the AMReX RAG pipeline: cut AMReX C++ files into chunks.

One chunk = one function or one class, including its template line and the
comment right above it. If a class is too big, it becomes an "overview"
chunk plus one chunk per member function. Files that tree-sitter cannot
split fall back to a simple blank-line splitter. Chunks longer than
WINDOW_CHARS are cut into overlapping windows so every part gets embedded.

Run:  python chunk_amrex.py
Out:  chunks.jsonl  (one JSON object per line)
"""
import json
import sys
from pathlib import Path

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

# ----- settings you may change -----
SOURCE_DIRS = [
    "amrex/Src/Base",
    "amrex-tutorials/Base",
    "amrex-tutorials/LinearSolvers",
    "amrex-tutorials/AMR",
]
EXTENSIONS = {".H", ".h", ".hpp", ".cpp", ".cc"}
MAX_CHARS = 6000        # bigger chunks are cut or split
MIN_CHARS = 80          # smaller chunks are dropped
FALLBACK_CHARS = 1500   # block size for the blank-line splitter
WINDOW_CHARS = 2000     # chunks longer than this are cut into windows
OVERLAP_LINES = 3       # neighbouring windows share this many lines
OUT_FILE = "chunks.jsonl"
# -----------------------------------

sys.setrecursionlimit(5000)
KINDS = ("function_definition", "class_specifier", "struct_specifier")
parser = Parser(Language(tscpp.language()))


def unwrap(node):
    """Return the function/class inside a node, or None if it has none."""
    if node.type == "template_declaration":
        for child in node.named_children:
            inner = unwrap(child)
            if inner is not None:
                return inner
        return None
    return node if node.type in KINDS else None


def symbol_of(inner):
    """Best-effort name of a function or class."""
    if inner.type == "function_definition":
        declarator = inner.child_by_field_name("declarator")
        if declarator is None:
            return "?"
        return declarator.text.decode("utf-8", "replace").split("(")[0].strip()[:120]
    name = inner.child_by_field_name("name")
    return name.text.decode("utf-8", "replace") if name is not None else "?"


def with_comment(node):
    """Start of the node, moved up to include comment lines right above it."""
    start_byte, start_row = node.start_byte, node.start_point[0]
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment" and prev.end_point[0] >= start_row - 1:
        start_byte, start_row = prev.start_byte, prev.start_point[0]
        prev = prev.prev_sibling
    return start_byte, start_row


def add(chunks, rel, symbol, kind, text, start_line, end_line):
    if len(text.strip()) < MIN_CHARS:
        return
    chunks.append({
        "file": rel,
        "symbol": symbol,
        "kind": kind,
        "start_line": start_line,
        "end_line": end_line,
        "text": text,
    })


def walk(node, src, rel, chunks, owner=""):
    inner = unwrap(node)
    if inner is None:
        for child in node.named_children:
            walk(child, src, rel, chunks, owner)
        return

    start_byte, start_row = with_comment(node)
    text = src[start_byte:node.end_byte].decode("utf-8", "replace")
    name = symbol_of(inner)
    if owner:
        name = owner + "::" + name
    first, last = start_row + 1, node.end_point[0] + 1

    if len(text) <= MAX_CHARS:
        add(chunks, rel, name, inner.type, text, first, last)
        return

    if inner.type in ("class_specifier", "struct_specifier"):
        add(chunks, rel, name + " (overview)", inner.type, text[:MAX_CHARS], first, last)
        body = inner.child_by_field_name("body")
        if body is not None:
            for member in body.named_children:
                walk(member, src, rel, chunks, owner=name)
    else:
        add(chunks, rel, name, inner.type, text[:MAX_CHARS], first, last)


def fallback(text, rel, chunks):
    """Simple splitter used when tree-sitter gives nothing for a file."""
    lines = text.splitlines()
    block, block_start, size = [], 1, 0
    for number, line in enumerate(lines, start=1):
        if not block:
            block_start = number
        block.append(line)
        size += len(line) + 1
        if (size >= FALLBACK_CHARS and line.strip() == "") or size >= MAX_CHARS:
            add(chunks, rel, "(block)", "block", "\n".join(block), block_start, number)
            block, size = [], 0
    if block:
        add(chunks, rel, "(block)", "block", "\n".join(block), block_start, len(lines))


def split_long(chunk):
    """Cut a long chunk into overlapping windows so every part is embedded."""
    text = chunk["text"]
    if len(text) <= WINDOW_CHARS:
        return [chunk]
    lines = text.splitlines()
    windows = []
    start = 0
    while start < len(lines):
        size, end = 0, start
        while end < len(lines) and (end == start or size + len(lines[end]) + 1 <= WINDOW_CHARS):
            size += len(lines[end]) + 1
            end += 1
        windows.append((start, end))
        if end >= len(lines):
            break
        start = max(end - OVERLAP_LINES, start + 1)
    parts = []
    for n, (a, b) in enumerate(windows, start=1):
        part = dict(chunk)
        part["text"] = "\n".join(lines[a:b])
        part["symbol"] = "%s (part %d/%d)" % (chunk["symbol"], n, len(windows))
        part["start_line"] = chunk["start_line"] + a
        part["end_line"] = chunk["start_line"] + b - 1
        parts.append(part)
    return parts


def main():
    chunks, files, fallback_files = [], 0, 0
    for folder in SOURCE_DIRS:
        base = Path(folder)
        if not base.exists():
            print("WARNING: folder not found, skipped:", folder)
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in EXTENSIONS:
                continue
            files += 1
            src = path.read_bytes()
            rel = path.as_posix()
            before = len(chunks)
            try:
                tree = parser.parse(src)
                walk(tree.root_node, src, rel, chunks)
            except Exception as error:
                print("tree-sitter failed on", rel, "->", error)
                del chunks[before:]
            if len(chunks) == before:
                fallback_files += 1
                fallback(src.decode("utf-8", "replace"), rel, chunks)

    before_split = len(chunks)
    final = []
    for chunk in chunks:
        final.extend(split_long(chunk))
    chunks = final

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks):
            chunk["id"] = "c" + str(i)
            out.write(json.dumps(chunk) + "\n")

    print("files scanned:", files)
    print("chunks before windowing:", before_split)
    print("chunks written:", len(chunks), "->", OUT_FILE)
    print("files that used the fallback splitter:", fallback_files)
    print("first 10 chunks:")
    for chunk in chunks[:10]:
        print("  ", chunk["symbol"], "|", chunk["file"], chunk["start_line"], "-", chunk["end_line"])


if __name__ == "__main__":
    main()