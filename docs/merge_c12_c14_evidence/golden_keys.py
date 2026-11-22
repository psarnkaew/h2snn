"""Merge c12+c14: compare the configuration and tensor key sets of the old (66e9349) and new v3b golden."""
import sys
import torch
sys.path.insert(0, "../docs/legacy_evidence")
import v3b_golden as g  # noqa: E402
old, new = (torch.load(p, weights_only=False) for p in sys.argv[1:3])
ro, rn = {}, {}
g._flatten("", old["records"], ro); g._flatten("", new["records"], rn)
print("configs old/new", len(old["records"]), len(new["records"]), "same names", sorted(old["records"]) == sorted(new["records"]))
print("skipped old/new", sorted(old["skipped"]), sorted(new["skipped"]))
print("tensors old/new", len(ro), len(rn), "only old", sorted(set(ro) - set(rn)), "only new", sorted(set(rn) - set(ro)))
same = [k for k in ro if k in rn and ro[k].shape == rn[k].shape and torch.equal(ro[k], rn[k])]
print("bit-identical", len(same), "changed", len(ro) - len(same))
changed_cfg = sorted({k.split(".")[0] for k in ro if k not in same})
print("configurations with changed tensors", changed_cfg)
print("unchanged configurations", sorted(set(old["records"]) - set(changed_cfg)))
