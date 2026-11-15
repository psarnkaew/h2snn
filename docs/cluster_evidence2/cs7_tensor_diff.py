"""CS7-3: compare two captured reference files tensor by tensor.

    python cs7_tensor_diff.py <reference.pt> <candidate.pt>

Prints: tensors compared, number differing, max |diff| over all differing tensors, and every
differing name with its own max |diff| (dtype/shape/key mismatches are listed too). Exit 1 if any
tensor differs. Equality is torch.equal (bit-for-bit); nothing is rounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def flatten(prefix: str, value, into: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, into)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            flatten(f"{prefix}[{i}]", v, into)
    else:
        into[prefix] = value


def main() -> int:
    ref_path, cand_path = Path(sys.argv[1]), Path(sys.argv[2])
    ref, cand = {}, {}
    flatten("", torch.load(ref_path, weights_only=False), ref)
    flatten("", torch.load(cand_path, weights_only=False), cand)
    ref_t = {k: v for k, v in ref.items() if isinstance(v, torch.Tensor)}
    differing: list[tuple[str, str]] = []
    worst = 0.0
    for key, want in ref_t.items():
        have = cand.get(key)
        if not isinstance(have, torch.Tensor):
            differing.append((key, "missing in candidate"))
        elif have.dtype != want.dtype or have.shape != want.shape:
            differing.append((key, f"dtype/shape {have.dtype}{tuple(have.shape)} vs {want.dtype}{tuple(want.shape)}"))
        elif not torch.equal(have, want):
            if want.is_floating_point():
                d = float((have.double() - want.double()).abs().max())
            else:
                d = float((have.long() - want.long()).abs().max())
            worst = max(worst, d)
            differing.append((key, f"max|diff|={d:.3e}"))
    extra = sorted(k for k, v in cand.items() if isinstance(v, torch.Tensor) and k not in ref_t)
    print(f"reference {ref_path}\ncandidate {cand_path}")
    print(f"tensors={len(ref_t)} differing={len(differing)} max|diff|={worst:.3e} extra_in_candidate={len(extra)}")
    for key, why in differing:
        print(f"  DIFF {key}: {why}")
    for key in extra:
        print(f"  EXTRA {key}")
    return 1 if differing or extra else 0


if __name__ == "__main__":
    sys.exit(main())
