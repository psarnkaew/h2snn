"""E4: per-test difference between two `unittest -v` logs. usage: python -I e4_diff.py <log A> <log B>
A test id is 'module.Class.test'. Status: worst of its lines (ERROR > FAIL > skipped > ok); subtests appear as FAIL/ERROR summary headers.
"""
import re
import sys

ST = {"ok": 0, "skipped": 1, "FAIL": 2, "ERROR": 3}

def parse(path):
    res, pending = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"^\w+ \(([\w.]+)\)", line)
        if m:
            pending = m.group(1)
        m2 = re.search(r" \.\.\. (ok|FAIL|ERROR|skipped.*)$", line)
        m3 = re.match(r"^(ok|FAIL|ERROR|skipped.*)$", line)
        mm = m2 or m3
        if mm and pending:
            s = mm.group(1).split()[0].rstrip("'\"")
            s = "skipped" if s.startswith("skipped") else s
            if ST.get(s, -1) >= ST.get(res.get(pending), -1):
                res[pending] = s
            pending = None
        m4 = re.match(r"^(FAIL|ERROR): \w+ \(([\w.]+)\)", line)
        if m4:
            res[m4.group(2)] = m4.group(1)
    return res

a, b = parse(sys.argv[1]), parse(sys.argv[2])
print(f"A={sys.argv[1]}: {len(a)} tests, " + str({k: sum(v == k for v in a.values()) for k in ST}))
print(f"B={sys.argv[2]}: {len(b)} tests, " + str({k: sum(v == k for v in b.values()) for k in ST}))
print("only in A:", sorted(set(a) - set(b)))
print("only in B:", sorted(set(b) - set(a)))
d = [(t, a[t], b[t]) for t in sorted(set(a) & set(b)) if a[t] != b[t]]
print(f"status differences: {len(d)}")
for t, x, y in d:
    print(f"  {t}: A={x} B={y}")
