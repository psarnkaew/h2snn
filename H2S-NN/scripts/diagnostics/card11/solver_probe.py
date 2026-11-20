"""Card 11 B1: a run-time wrapper around the anchored solve that logs every PCG call.

No `src/` change: `install()` replaces three names at run time and calls the originals unchanged,
so the computation is the same (only extra reads and `.item()` synchronisations are added; checked
bit-identical by `b1_identity_check.py`):

- `src.models.anchored_solve.pcg` (module global used by `solve_anchored` and by the implicit
  backward of `AnchoredSolve`): one TSV row per call, written after the original returns and
  before `solve_anchored` / the backward can raise `PCGConvergenceError`, so the failing solve is
  logged;
- `src.models.anchored_solve.solve_anchored`: marks the calls it makes as `fwd` (else `bwd`);
- `src.models.h2snn_model.H2SNN._encode_anchored`: keeps the epoch, train/eval mode, message
  graph and scope index of the current forward (for degrees and the step counter).

Row fields: wall time, epoch, mode, step (training forward solves in this epoch), direction, tau,
tolerance, T_max, number of scopes, rows, largest scope (rows), iterations (max / mean over scopes),
unconverged scopes, final relative residual (max / median over scopes), the worst scope (most
iterations, then largest residual) with its rows, its largest node degree and its Jacobi range, the
Jacobi diagonal J = 1 / jacobi_inverse (min / max over all rows and over node rows), the largest node
degree of the message graph in the batch, `path_log_w` min / max / range of this forward (the
model's own `last_path_stats`), non-finite flags of the right-hand side and the solution, and the
wall time of the call.
"""

from __future__ import annotations

import gzip
import math
import time
from pathlib import Path

import torch

COLUMNS = (
    "wall_s", "epoch", "mode", "step", "direction", "tau", "tolerance", "t_max", "num_scopes",
    "rows", "num_nodes", "largest_scope_rows", "iters_max", "iters_mean", "unconverged_scopes",
    "rel_res_max", "rel_res_median", "worst_scope", "worst_scope_rows", "worst_scope_iters",
    "worst_scope_rel_res", "worst_scope_max_degree", "worst_scope_jacobi_min",
    "worst_scope_jacobi_max", "jacobi_min", "jacobi_max", "jacobi_node_min", "jacobi_node_max",
    "max_degree", "path_log_w_min", "path_log_w_max", "path_log_w_range", "rhs_finite",
    "y_finite", "call_s",
)


class WallLimitReached(RuntimeError):
    """The diagnosis run's own wall limit (not a stop condition of the model)."""


class SolverProbe:
    def __init__(self, path: Path, wall_limit_s: float | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        self.handle = opener(path, "wt", encoding="utf-8")
        self.handle.write("\t".join(COLUMNS) + "\n")
        self.t0 = time.time()
        self.wall_limit_s = wall_limit_s
        self.ctx: dict = {}
        self.direction = "bwd"
        self.step_epoch = None
        self.step = -1
        self.rows_written = 0
        self.installed = False
        self._originals: dict = {}

    # ------------------------------------------------------------------ install / remove
    def install(self) -> "SolverProbe":
        import src.models.anchored_solve as AS
        import src.models.h2snn_model as HM

        probe = self
        orig_pcg, orig_solve, orig_encode = AS.pcg, AS.solve_anchored, HM.H2SNN._encode_anchored
        self._originals = {"pcg": orig_pcg, "solve": orig_solve, "encode": orig_encode}

        def encode(model, h0, edge_index, edge_type, face_groups, num_nodes, epoch,
                   scope_index=None, global_node_id=None):
            training = bool(model.training)
            if training:
                if probe.step_epoch != epoch:
                    probe.step_epoch, probe.step = epoch, -1
                probe.step += 1
            probe.ctx = {"epoch": epoch, "training": training, "edge_index": edge_index,
                         "num_nodes": int(num_nodes), "scope_index": scope_index, "model": model,
                         "step": probe.step if training else -1}
            return orig_encode(model, h0, edge_index, edge_type, face_groups, num_nodes, epoch,
                               scope_index, global_node_id)

        def solve_anchored(*args, **kwargs):
            probe.direction = "fwd"
            try:
                return orig_solve(*args, **kwargs)
            finally:
                probe.direction = "bwd"

        def pcg(apply_m, b, y0, jacobi_inverse, tolerance, max_iterations, scope_rows=None,
                num_scopes=1):
            start = time.time()
            out = orig_pcg(apply_m, b, y0, jacobi_inverse, tolerance, max_iterations, scope_rows,
                           num_scopes)
            try:
                probe.record(apply_m, b, jacobi_inverse, tolerance, max_iterations, scope_rows,
                             num_scopes, out, start)
            except WallLimitReached:
                raise
            except Exception as exc:  # the probe must never change the run
                print(f"[solver_probe] record failed: {exc!r}", flush=True)
            return out

        AS.pcg, AS.solve_anchored, HM.H2SNN._encode_anchored = pcg, solve_anchored, encode
        self.installed = True
        return self

    def remove(self) -> None:
        if not self.installed:
            return
        import src.models.anchored_solve as AS
        import src.models.h2snn_model as HM

        AS.pcg = self._originals["pcg"]
        AS.solve_anchored = self._originals["solve"]
        HM.H2SNN._encode_anchored = self._originals["encode"]
        self.installed = False

    def close(self) -> None:
        self.remove()
        self.handle.close()

    # ------------------------------------------------------------------ one row
    @torch.no_grad()
    def record(self, apply_m, b, jacobi_inverse, tolerance, max_iterations, scope_rows,
               num_scopes, out, start) -> None:
        y, iterations, converged, residual = out
        if b.is_cuda:
            torch.cuda.synchronize(b.device)
        call_s = time.time() - start
        system = getattr(apply_m, "__self__", None)
        tau = float(system.tau.detach()) if system is not None else float("nan")
        ctx = self.ctx
        num_nodes = int(getattr(system, "num_nodes", ctx.get("num_nodes", 0)))
        rows = int(b.size(0))
        S = int(num_scopes)
        multi = scope_rows is not None and S > 1
        jac = 1.0 / jacobi_inverse.detach().double()
        iters = iterations.detach().cpu().double()
        res = residual.detach().cpu().double()
        if multi:
            sizes = torch.bincount(scope_rows, minlength=S)
        else:
            sizes = torch.tensor([rows], device=b.device)
        # worst scope: most iterations, then largest residual
        order_key = iters * 1e6 + torch.nan_to_num(res, nan=1e5, posinf=1e5).clamp(max=1e5)
        worst = int(torch.argmax(order_key))
        # degrees of the message graph of this forward (node rows only)
        edge_index = ctx.get("edge_index")
        max_degree = worst_deg = float("nan")
        if edge_index is not None and num_nodes > 0:
            deg = torch.bincount(edge_index.reshape(-1), minlength=num_nodes)[:num_nodes]
            max_degree = float(deg.max()) if deg.numel() else 0.0
            if multi:
                node_scope = scope_rows[:num_nodes]
                sel = deg[node_scope == worst]
                worst_deg = float(sel.max()) if sel.numel() else 0.0
            else:
                worst_deg = max_degree
        if multi:
            wj = jac[scope_rows == worst]
        else:
            wj = jac
        stats = getattr(ctx.get("model"), "last_path_stats", None) or {}
        lw_max = stats.get("path_log_w_max", float("nan"))
        lw_range = stats.get("path_log_w_range", float("nan"))
        lw_min = lw_max - lw_range if not (math.isnan(lw_max) or math.isnan(lw_range)) else float("nan")
        row = {
            "wall_s": f"{time.time() - self.t0:.2f}", "epoch": ctx.get("epoch", ""),
            "mode": "train" if ctx.get("training") else "eval", "step": ctx.get("step", -1),
            "direction": self.direction, "tau": f"{tau:.6g}", "tolerance": f"{tolerance:.1e}",
            "t_max": max_iterations, "num_scopes": S, "rows": rows, "num_nodes": num_nodes,
            "largest_scope_rows": int(sizes.max()), "iters_max": int(iters.max()),
            "iters_mean": f"{float(iters.mean()):.2f}",
            "unconverged_scopes": int((~converged).sum()),
            "rel_res_max": f"{float(res.max()):.3e}", "rel_res_median": f"{float(res.median()):.3e}",
            "worst_scope": worst, "worst_scope_rows": int(sizes[worst]) if multi else rows,
            "worst_scope_iters": int(iters[worst]), "worst_scope_rel_res": f"{float(res[worst]):.3e}",
            "worst_scope_max_degree": worst_deg,
            "worst_scope_jacobi_min": f"{float(wj.min()):.4g}" if wj.numel() else "",
            "worst_scope_jacobi_max": f"{float(wj.max()):.4g}" if wj.numel() else "",
            "jacobi_min": f"{float(jac.min()):.4g}", "jacobi_max": f"{float(jac.max()):.4g}",
            "jacobi_node_min": f"{float(jac[:num_nodes].min()):.4g}" if num_nodes else "",
            "jacobi_node_max": f"{float(jac[:num_nodes].max()):.4g}" if num_nodes else "",
            "max_degree": max_degree, "path_log_w_min": f"{lw_min:.4g}",
            "path_log_w_max": f"{lw_max:.4g}", "path_log_w_range": f"{lw_range:.4g}",
            "rhs_finite": int(bool(torch.isfinite(b).all())),
            "y_finite": int(bool(torch.isfinite(y).all())), "call_s": f"{call_s:.3f}",
        }
        self.handle.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        self.rows_written += 1
        if self.rows_written % 200 == 0:
            self.handle.flush()
        if self.wall_limit_s is not None and time.time() - self.t0 > self.wall_limit_s:
            self.handle.flush()
            raise WallLimitReached(f"wall limit {self.wall_limit_s:.0f} s reached")


def install(path: Path, wall_limit_s: float | None = None) -> SolverProbe:
    return SolverProbe(path, wall_limit_s).install()
