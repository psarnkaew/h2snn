"""E2: installed version of every exact pin of a lock file vs the pin (read-only).

usage: python -I e2_versions.py <lock file>
A local build tag the pin does not state ('+cu124', '+pt24cu124') is reported as 'tag', not as a difference.
"""
import re
import sys
from importlib import metadata

diffs = 0
for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.split("#")[0].strip()
    if not line or line.startswith("-"):
        continue
    name, _, want = line.partition("==")
    try:
        have = metadata.version(re.sub(r"[-_.]+", "-", name).lower())
    except metadata.PackageNotFoundError:
        print(f"MISSING    {name}=={want}")
        diffs += 1
        continue
    if have == want:
        print(f"equal      {name}=={want}")
    elif "+" in have and "+" not in want and have.split("+")[0] == want:
        print(f"tag        {name}=={want}  installed {have}")
    else:
        print(f"DIFFERENT  {name}  lock {want}  installed {have}")
        diffs += 1
print(f"differences: {diffs}")
