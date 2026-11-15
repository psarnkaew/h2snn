"""LR-3 helper (run on the login node as an editor, no test run): remove named test methods, classes and
top-level helpers from a test file by their AST line spans (decorators and the blank lines after them
included). Usage: python lr3_remove_tests.py <file> <name> [<name> ...]
A name is `Class.method`, `Class` (the whole class) or `function` (a top-level def). Every name must
exist; the removed spans are printed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    names = sys.argv[2:]
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.split("\n")
    spans = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            spans.append((node, node.name))
            found.add(node.name)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                key = f"{node.name}.{getattr(item, 'name', '')}"
                if isinstance(item, ast.FunctionDef) and key in names:
                    spans.append((item, key))
                    found.add(key)
    missing = sorted(set(names) - found)
    if missing:
        raise SystemExit(f"not found in {path}: {missing}")
    remove = set()
    for node, name in spans:
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
        end = node.end_lineno  # exclusive index after the last line
        while end < len(lines) and lines[end].strip() == "":
            end += 1
        # keep one separator: give back the blank lines before the next statement
        print(f"remove {name}: lines {start + 1}-{node.end_lineno}")
        remove.update(range(start, end))
    kept = [line for i, line in enumerate(lines) if i not in remove]
    text = "\n".join(kept)
    # normalise: no more than two blank lines in a row
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
