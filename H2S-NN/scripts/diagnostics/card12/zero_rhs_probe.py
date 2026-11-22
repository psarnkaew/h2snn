"""Card 12 P2/P3: per-PCG-call log of the zero right-hand-side rule (spec 1.30).

Installed after the card 11 solver probe (`card11/solver_probe.py`): it replaces
`src.models.anchored_solve.pcg` by an outer wrapper that, before calling the inner function
unchanged, computes the per-scope ||b_s|| with the module's own `segment_dot` and the rule's own
`zero_rhs_mask`, and writes one TSV row per call: epoch, mode, step, direction (from the card 11
probe's context), number of scopes, triggered scopes, max / median / min ||b_s||, the smallest
ratio ||b_s|| / max, and the ||b_s|| of every triggered scope. Only extra reads and `.item()`
synchronisations are added; the computation is unchanged.

`stop_after=(E, S)` (card 12 amendment 2: P2 stops once PDNS passes epoch 1 step 172): at the first
training solve of epoch E with step > S, i.e. after step S's forward and backward completed, the
wrapper raises the card 11 probe's `WallLimitReached` (a diagnosis stop, reported as such).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import torch

COLUMNS = ("epoch", "mode", "step", "direction", "num_scopes", "triggered", "b_norm_max",
           "b_norm_median", "b_norm_min", "min_ratio", "triggered_scopes", "triggered_b_norms")


class ZeroRHSProbe:
    def __init__(self, path: Path, card11_probe, stop_after: tuple[int, int] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(path, "wt", encoding="utf-8") if path.suffix == ".gz" \
            else open(path, "w", encoding="utf-8")
        self.handle.write("\t".join(COLUMNS) + "\n")
        self.card11 = card11_probe
        self.inner = None
        self.calls = 0
        self.triggered_total = 0
        self.stop_after = stop_after

    def install(self) -> "ZeroRHSProbe":
        import src.models.anchored_solve as AS

        probe = self
        inner = AS.pcg
        self.inner = inner

        def pcg(apply_m, b, y0, jacobi_inverse, tolerance, max_iterations, scope_rows=None,
                num_scopes=1):
            ctx = getattr(probe.card11, "ctx", {}) or {}
            if (probe.stop_after is not None and ctx.get("training")
                    and ctx.get("epoch") == probe.stop_after[0]
                    and (ctx.get("step") or -1) > probe.stop_after[1]):
                from solver_probe import WallLimitReached
                raise WallLimitReached(f"card12 stop: epoch {ctx.get('epoch')} step {ctx.get('step')} "
                                       f"reached (passed step {probe.stop_after[1]})")
            try:
                probe.record(AS, b, scope_rows, num_scopes)
            except Exception as exc:  # the probe must never change the run
                print(f"[zero_rhs_probe] record failed: {exc!r}", flush=True)
            return inner(apply_m, b, y0, jacobi_inverse, tolerance, max_iterations, scope_rows,
                         num_scopes)

        AS.pcg = pcg
        return self

    @torch.no_grad()
    def record(self, AS, b, scope_rows, num_scopes) -> None:
        b_norm = AS.segment_dot(b, b, scope_rows, num_scopes).sqrt()
        mask = AS.zero_rhs_mask(b_norm)
        bmax = float(b_norm.max())
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        ctx = getattr(self.card11, "ctx", {}) or {}
        n = len(idx)
        self.calls += 1
        self.triggered_total += n
        ratio = float(b_norm.min()) / bmax if bmax > 0 else 0.0
        row = (ctx.get("epoch"), "train" if ctx.get("training") else "eval", ctx.get("step"),
               getattr(self.card11, "direction", "?"), num_scopes, n, f"{bmax:.4e}",
               f"{float(b_norm.median()):.4e}", f"{float(b_norm.min()):.4e}", f"{ratio:.4e}",
               ",".join(map(str, idx)), ",".join(f"{float(b_norm[i]):.4e}" for i in idx))
        self.handle.write("\t".join(str(v) for v in row) + "\n")
        if n:
            self.handle.flush()
            print(f"[zero_rhs_probe] triggered epoch={row[0]} step={row[2]} dir={row[3]} "
                  f"scopes={row[10]} b_norm={row[11]} max={row[6]}", flush=True)

    def close(self) -> None:
        import src.models.anchored_solve as AS

        if self.inner is not None:
            AS.pcg = self.inner
            self.inner = None
        self.handle.close()
        print(f"[zero_rhs_probe] calls={self.calls} triggered_scopes={self.triggered_total}",
              flush=True)
