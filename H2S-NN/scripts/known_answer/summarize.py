"""Card 4 K3-K5: tables of the per-run JSONs against the FIXED criteria of the pre-registration
(`docs/known_answer_preregistration_draft.md`); nothing here is tuned or loosened.

    python scripts/known_answer/summarize.py --k3 ../docs/card4_evidence/k3 --k4 ../docs/card4_evidence/k4 \
        --k5 ../docs/card4_evidence/k5 --out ../docs/card4_evidence/known_answer_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(folder: Path, prefix: str) -> list[dict]:
    if folder is None or not folder.exists():
        return []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob(f"{prefix}_*.json"))]
    return [r for r in records if "arm" in r]          # per-run records only (not the audit)


def k3(rows: list[dict]) -> list[str]:
    if not rows:
        return ["K3: no runs."]
    lines = ["| arm | β | seed | test acc | best ep | rel. share D_P | rel. share cell mass | dispersion | τ final | −∂L/∂β>0 share |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], ["0", "log64", "log512"].index(r["beta"]), r["seed"])):
        if r.get("status") == "failed":
            lines.append(f"| {r['arm']} | {r['beta']} | {r['seed']} | failed: {r['stop']} | — | — | — | — | — | — |")
            continue
        share = r["neg_dL_dbeta_positive_share_before_best"]
        lines.append(f"| {r['arm']} | {r['beta']} | {r['seed']} | {r['test_acc']:.3f} | {r['best_epoch']} | "
                     f"{r['relevant_share_D_median']:.3f} | {r['relevant_share_cell_mass_median']:.3f} | "
                     f"{r['dispersion']:.3f} | {r['tau_final']:.3f} | "
                     f"{'n/a' if share is None else f'{share:.2f}'} |")
    acc = {(r["arm"].split(" ")[0], r["beta"], r["seed"]): r.get("test_acc") for r in rows}
    complete = len(acc) == 18 and all(v is not None for v in acc.values())
    leak = any((acc.get((a, "0", s)) or 0) >= 0.9 for a in ("full", "covering_only") for s in (1, 2, 3))
    full512 = [acc.get(("full", "log512", s)) for s in (1, 2, 3)]
    warning = any(v is not None and v < 0.9 for v in full512)
    cov = [acc.get(("covering_only", b, s)) for b in ("0", "log64", "log512") for s in (1, 2, 3)]
    c12 = (all(v is not None and v >= 0.9 for v in full512)
           and all(v is not None and v <= 0.65 for v in cov))
    cov512_high = all((acc.get(("covering_only", "log512", s)) or 0) >= 0.9 for s in (1, 2, 3))
    failed = [f"{r['arm']} {r['beta']} s{r['seed']}" for r in rows if r.get("status") == "failed"]
    lines += ["", f"- runs: {len(acc)} of 18; failed by a stop condition: {failed or 'none'}",
              f"- leak (β = 0 ≥ 0.9 in either arm): {'YES (no verdict)' if leak else 'no'}",
              f"- warning signal (`full` + oracle < 0.9 on any seed at log 512): "
              f"{'YES: stop the card' if warning else 'no'} ({full512})",
              f"- C12 known-answer part (full ≥ 0.9 on 3/3 at log 512 and covering_only ≤ 0.65 at every β "
              f"and seed): {'passes' if c12 and not leak and complete else ('not decidable: a run stopped' if any(v is None for v in cov) and all((v is None or v <= 0.65) for v in cov) else 'does not pass')}",
              f"- covering_only ≥ 0.9 at log 512 on 3/3: {'yes' if cov512_high else 'no'}"]
    return lines


def k4(rows: list[dict]) -> list[str]:
    if not rows:
        return ["K4: no runs."]
    lines = ["| arm | seed | test acc | best ep | log-w gap len-4 (rel − irr) | log-w gap all cells | τ final |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
        lines.append(f"| {r['arm']} | {r['seed']} | {r['test_acc']:.3f} | {r['best_epoch']} | "
                     f"{r.get('log_w_gap_len4', float('nan')):.3f} | {r['log_w_gap']:.3f} | {r['tau_final']:.3f} |")
    acc = {(r["arm"], r["seed"]): r["test_acc"] for r in rows}
    controls = [v for (a, _), v in acc.items() if a in ("endpoints", "uniform")]
    oracle = [v for (a, _), v in acc.items() if a.startswith("full + oracle")]
    full = [v for (a, _), v in acc.items() if a == "full"]
    if any(v >= 0.9 for v in controls):
        verdict = "leak (a control ≥ 0.9): redesign, no verdict"
    elif any(v < 0.9 for v in oracle):
        verdict = "full + oracle < 0.9: capacity not shown on this graph"
    elif len(full) == 3 and all(v >= 0.9 for v in full):
        verdict = "full ≥ 0.9 on 3/3: passes (g learns instance-level relevance)"
    elif len(full) == 3 and all(v <= 0.65 for v in full):
        verdict = "full ≤ 0.65 on all seeds: not shown at default settings"
    else:
        verdict = "between: reported as is"
    return lines + ["", f"- runs: {len(rows)} of 12", f"- reading (fixed): {verdict}"]


def k5(rows: list[dict]) -> list[str]:
    if not rows:
        return ["K5: no runs."]
    lines = ["| arm | seed | test acc | σ* coverage | strict rank of σ* (mean log w) | σ* − best other | best other |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
        ro = r["readout"]
        lines.append(f"| {r['arm']} | {r['seed']} | {r['test_acc']:.3f} | {ro['planted_coverage']:.4f} | "
                     f"{ro['strict_rank_mean_log_weight']} | {ro['gap_star_minus_best_other']:.4f} | {ro['best_other']} |")
    full = [r for r in rows if r["arm"] == "full"]
    passed = len(full) == 3 and all(r["readout"]["criterion_pass"] for r in full)
    return lines + ["", f"- runs: {len(rows)} of 6",
                    f"- criterion (σ* strictly first on 3/3 `full` seeds): {'passes' if passed else 'does not pass'}",
                    "- `uniform` accuracy: selection not required on this graph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k3", type=Path)
    parser.add_argument("--k4", type=Path)
    parser.add_argument("--k5", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = ["# Card 4 known-answer tables (synthetic, pre-registered; generated by "
            "`scripts/known_answer/summarize.py`)", "", "## K3 S1′-B", *k3(load(args.k3, "s1pb")),
            "", "## K4 S3", *k4(load(args.k4, "s3")), "", "## K5 S2′", *k5(load(args.k5, "s2p")), ""]
    args.out.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
