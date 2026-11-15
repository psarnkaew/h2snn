"""Legacy-removal card, LR-1: v3-B golden outputs, captured on the code before any removal and compared
after every unit (same pattern as `docs/partMB_evidence/mb0_reference.py`).

    python ../docs/legacy_evidence/v3b_golden.py capture|compare <file.pt>     (run from H2S-NN/)

For every configuration (synthetic data only, the anchored v3-B model): evaluation logits (or scores), one
training step's loss, every parameter gradient, and the anchored solver's statistics of both forwards
(PCG iterations and residuals of the forward and backward solve, tolerance, system sizes). Configurations:
NC full batch: the 7 values of V3B_ARMS; path_weight_input endpoints / schema / uniform; geometry_context
false; path_weight_context true; direction_blind_maps; map_family dense; h0_only_readout; path_source
schemas; an IMDB-style multilabel head. LP: decoder dot and distmult. GC: each pooling. Mini-batch: one NC
batch of scopes and one LP batch with scopes (as the Part MB tests).

`compare` prints every differing tensor name and exits 1 on any difference; both modes exit 2 unless the
pinned kernel environment E is in effect (CS8 rule R2); the references are computed with 1 thread (set
here and restored). A configuration that cannot be built is listed (`skipped`) and left out.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "H2S-NN"
sys.path.insert(0, str(REPO))
import torch  # noqa: E402

PINNED_MESSAGE = ("bit-exact test needs the pinned kernel environment: "
                  "run through scripts/slurm/run_tests.sbatch")
SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
         "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"}
ARMS = ("full", "covering_only", "hub_A", "closure_only", "no_path_cells", "instance_features",
        "free_cell_maps")


def kernel_path_is_pinned() -> bool:
    return (torch.backends.cpu.get_cpu_capability() == "AVX2"
            and os.environ.get("MKL_CBWR") == "AVX2")


def _config(model: dict | None = None, ablation: dict | None = None, data: dict | None = None,
            dataset: str = "acm"):
    from src.configs.config import TrainConfig

    base_data = {"dataset_name": dataset, "wisp_max_length": 3, "wisp_walks_per_anchor": 5}
    return TrainConfig.from_dict({
        "device": "cpu", "seed": 3,
        "data": {**base_data, **(data or {})},
        "model": {**SMALL, **(model or {})},
        "ablation": {"coupling": "symmetric", **(ablation or {})}})


def _solver(model) -> dict:
    return {k: torch.tensor(float(v), dtype=torch.float64)
            for k, v in sorted(model.last_solver_stats.items())}


def _grads(model) -> dict:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


# ----------------------------------------------------------------------------- node classification
def _nc_graph(cfg, multilabel: bool = False):
    from src.data.dataset import build_dataset
    from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

    hetero = acm_like_hetero(seed=9)
    del hetero["term"].x  # one featureless type (learned per-node rows), as mb0_reference.py
    with patched_acm_loader(hetero):
        data = build_dataset(cfg)
    if multilabel:  # IMDB-style head: multi-hot targets, per-genre BCE (spec §5 task readouts)
        g = torch.Generator().manual_seed(4)
        y = (torch.rand(data.target_y.numel(), 3, generator=g) < 0.4).long()
        data = replace(data, target_y=y, num_classes=3, is_multilabel=True)
    return data


def _nc_record(cfg, multilabel: bool = False) -> dict:
    from src.models.h2snn_model import build_model

    data = _nc_graph(cfg, multilabel)
    torch.manual_seed(0)
    model = build_model(cfg, data)
    out: dict = {}
    model.eval()
    with torch.no_grad():
        out["eval_logits"] = model(data).logits.clone()
    out["eval_solver"] = _solver(model)
    model.train()
    torch.manual_seed(1)
    logits = model(data, epoch=0).logits
    mask = data.target_train_mask
    if multilabel:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], data.target_y[mask].to(logits.dtype))
    else:
        loss = torch.nn.functional.cross_entropy(logits[mask], data.target_y[mask])
    loss.backward()
    out["train_loss"] = loss.detach().clone()
    out["train_solver"] = _solver(model)
    out["grads"] = _grads(model)
    return out


# ----------------------------------------------------------------------------- link prediction
def _write_tiny_lastfm(root: Path) -> None:
    """Copy of `tests/test_task_heads.py::_write_tiny_lastfm` at `a6d6816` (HGB local format)."""
    from src.data.data_layout import lp_root

    folder = lp_root(root) / "LastFM"
    folder.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    users, artists, tags = range(0, 12), range(12, 24), range(24, 30)
    with (folder / "node.dat").open("w", encoding="utf-8") as handle:
        for node_type, ids in enumerate((users, artists, tags)):
            for raw in ids:
                features = ",".join(f"{float(v):.4f}" for v in torch.rand(3, generator=generator))
                handle.write(f"{raw}\tn{raw}\t{node_type}\t{features}\n")
    train_pairs: set[tuple[int, int]] = set()
    test_pairs: set[tuple[int, int]] = set()
    for user in users:
        chosen = torch.randperm(12, generator=generator)[:6].tolist()
        for rank, artist in enumerate(chosen):
            (test_pairs if rank == 5 else train_pairs).add((user, 12 + artist))
    with (folder / "link.dat").open("w", encoding="utf-8") as handle:
        for user, artist in sorted(train_pairs):
            handle.write(f"{user}\t{artist}\t0\t1.0\n")
        for user in users:
            handle.write(f"{user}\t{(user + 1) % 12}\t1\t1.0\n")
        for artist in artists:
            handle.write(f"{artist}\t{24 + artist % 6}\t2\t1.0\n")
    with (folder / "link.dat.test").open("w", encoding="utf-8") as handle:
        for user, artist in sorted(test_pairs):
            handle.write(f"{user}\t{artist}\t0\t1.0\n")


def _lp_record(decoder: str) -> dict:
    from src.data.link_prediction_dataset import build_link_prediction_dataset
    from src.models.h2snn_model import build_link_prediction_model

    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_lastfm(Path(tmp))
        cfg = _config(data={"root": tmp, "decoder_type": decoder, "wisp_walks_per_anchor": 4},
                      dataset="lastfm")
        data = build_link_prediction_dataset(cfg)
    torch.manual_seed(0)
    model = build_link_prediction_model(cfg, data)

    def scores(pos, neg, epoch):
        encoded = model.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                               data.face_groups, data.num_nodes, epoch=epoch)
        return model.link_readout(encoded.representation, torch.cat([pos, neg], dim=1), None)

    out: dict = {}
    model.eval()
    with torch.no_grad():
        out["eval_scores"] = scores(data.val_pos_edge_index, data.val_neg_edge_index, -1).clone()
    out["eval_solver"] = _solver(model)
    model.train()
    torch.manual_seed(1)
    pos, neg = data.train_pos_edge_index, data.val_neg_edge_index
    s = scores(pos, neg, 0)
    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(s.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(s, labels)
    loss.backward()
    out["train_loss"] = loss.detach().clone()
    out["train_solver"] = _solver(model)
    out["grads"] = _grads(model)
    return out


# ----------------------------------------------------------------------------- graph classification
def _typed_graph(seed: int, num_nodes: int = 7):
    """Copy of `tests/test_task_heads.py::_typed_graph` at `a6d6816`."""
    from torch_geometric.data import Data

    generator = torch.Generator().manual_seed(seed)
    node_type = torch.randint(2, (num_nodes,), generator=generator)
    src, dst = [], []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if bool(torch.rand(1, generator=generator) < 0.45) or v == u + 1:
                src += [u, v]
                dst += [v, u]
    edge_index = torch.tensor([src, dst])
    edge_type = 2 * node_type[edge_index[0]] + node_type[edge_index[1]]
    return Data(x=torch.randn(num_nodes, 5, generator=generator), edge_index=edge_index,
                edge_type=edge_type, node_type=node_type,
                y=torch.randint(2, (1,), generator=generator), graph_id=torch.tensor([seed]))


def _gc_record(pooling: str) -> dict:
    from torch_geometric.data import Batch

    from src.builders import build_optimizer
    from src.models.h2snn_model import ModelDims, build_graph_classification_model
    from src.trainers.graph_classification_trainer import GraphClassificationTrainer

    cfg = _config(model={"graph_pooling": pooling}, dataset="mutag",
                  data={"wisp_walks_per_anchor": 3})
    dims = ModelDims(in_channels=5, num_node_types=2, num_relations=4, num_classes=2,
                     face_lengths=(2, 3), path_actions=True)
    torch.manual_seed(0)
    model = build_graph_classification_model(cfg, dims)
    trainer = GraphClassificationTrainer(model, build_optimizer(model, cfg), None, cfg,
                                         logger=None, checkpointer=None)
    batch = Batch.from_data_list([_typed_graph(seed) for seed in range(4)])
    out: dict = {}
    model.eval()
    with torch.no_grad():
        out["eval_logits"] = trainer.forward_batch(batch).clone()
    out["eval_solver"] = _solver(model)
    model.train()
    torch.manual_seed(1)
    logits = trainer.forward_batch(batch, epoch=0)
    loss = torch.nn.functional.cross_entropy(logits, batch.y)
    loss.backward()
    out["train_loss"] = loss.detach().clone()
    out["train_solver"] = _solver(model)
    out["grads"] = _grads(model)
    return out


# ----------------------------------------------------------------------------- mini-batch
def _mb_nc_record() -> dict:
    """One batch of scopes, as tests/test_partMB_model.py (`_setup`, `_batch`)."""
    from src.data.dataset import build_dataset
    from src.data.scopes import build_scope_batch
    from src.data.wisp_sampler import build_wisp_anchor_cells
    from src.models.h2snn_model import build_model
    from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

    cfg = _config(data={"wisp_walks_per_anchor": 4})
    with patched_acm_loader(acm_like_hetero(num_papers=30, num_authors=16, seed=12)):
        graph = build_dataset(cfg)
    torch.manual_seed(0)
    model = build_model(cfg, graph)
    targets = graph.target_indices[:10]
    cells = build_wisp_anchor_cells(graph.edge_index, graph.edge_type, graph.node_type,
                                    graph.fact_layer.self_inverse, targets, 4, 3, seed=21,
                                    chunk_size=4)
    batch = build_scope_batch(graph, targets, [cells.of(int(t)) for t in targets])
    data = batch.data
    out: dict = {}
    model.eval()
    with torch.no_grad():
        out["eval_logits"] = model(data).logits.clone()
    out["eval_solver"] = _solver(model)
    model.train()
    torch.manual_seed(1)
    logits = model(data, epoch=0).logits
    loss = torch.nn.functional.cross_entropy(logits, data.target_y)
    loss.backward()
    out["train_loss"] = loss.detach().clone()
    out["train_solver"] = _solver(model)
    out["grads"] = _grads(model)
    return out


def _mb_lp_record() -> dict:
    """One LP batch with scopes, as tests/test_partMB_link.py (`_graph`, `_model`, LinkScopes)."""
    import random

    from src.configs.config import TrainConfig
    from src.data.structures import PreparedH2SData
    from src.models.h2snn_model import H2SNN, ModelDims
    from src.trainers.minibatch_lp import LinkScopes

    n = 20
    rng = random.Random(0)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < 60:
        u, v = rng.sample(range(n), 2)
        pairs.add((u, v))
    pairs |= {(v, u) for u, v in list(sorted(pairs))[:10]}
    pairs_sorted = sorted(pairs)
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(n, 4, generator=generator)
    edge_index = torch.tensor(pairs_sorted, dtype=torch.long).t()
    mask = torch.zeros(n, dtype=torch.bool)
    graph = PreparedH2SData(
        x=x, y=torch.zeros(n, dtype=torch.long), node_type=torch.zeros(n, dtype=torch.long),
        node_type_names=("user",), node_slices={"user": (0, n)}, feature_dims={"user": 4},
        edge_index=edge_index, edge_type=torch.zeros(edge_index.size(1), dtype=torch.long),
        relation_names=("user__rel_0__user",), relation_tuples=(("user", "rel_0", "user"),),
        train_mask=mask, val_mask=mask, test_mask=mask, target_node_type="user",
        target_indices=torch.arange(n), target_y=torch.zeros(n, dtype=torch.long),
        target_train_mask=mask, target_val_mask=mask, target_test_mask=mask, num_classes=2,
        is_multilabel=False, face_groups=(), metrics=torch.zeros(0))
    cfg = TrainConfig.from_dict({
        "device": "cpu",
        "data": {"dataset_name": "lastfm", "wisp_walks_per_anchor": 3, "wisp_max_length": 3},
        "model": {**SMALL, "solver_tolerance": 1e-4, "solver_max_iterations": 2000},
        "ablation": {"coupling": "symmetric"}})
    dims = ModelDims.from_prepared_data(graph)._replace(face_lengths=(2, 3), path_actions=True)
    torch.manual_seed(0)
    model = H2SNN(cfg, dims)
    scopes = LinkScopes(model, graph, 3, 3)
    links = torch.tensor([[0, 0, 3, 5, 7], [1, 2, 0, 9, 11]])
    negatives = torch.tensor([[2, 4, 6], [13, 15, 17]])
    out: dict = {}
    model.eval()
    with torch.no_grad():
        cells = scopes.cells(links.reshape(-1), seed=8)
        out["eval_scores"] = scopes.score(links, cells).clone()
    out["eval_solver"] = _solver(model)
    model.train()
    torch.manual_seed(1)
    both = torch.cat([links, negatives], dim=1)
    cells = scopes.cells(both.reshape(-1), seed=9)
    s = scopes.score(both, cells, epoch=0)
    labels = torch.cat([torch.ones(links.size(1)), torch.zeros(negatives.size(1))]).to(s.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(s, labels)
    loss.backward()
    out["train_loss"] = loss.detach().clone()
    out["train_solver"] = _solver(model)
    out["grads"] = _grads(model)
    return out


# ----------------------------------------------------------------------------- the cases
def cases() -> dict:
    nc = {f"nc_arm_{arm}": dict(model={"arm": arm}) for arm in ARMS}
    nc.update({f"nc_path_weight_input_{v}": dict(ablation={"path_weight_input": v})
               for v in ("endpoints", "schema", "uniform")})
    nc.update({
        "nc_geometry_context_false": dict(model={"geometry_context": False}),
        "nc_path_weight_context": dict(ablation={"path_weight_context": True}),
        "nc_direction_blind_maps": dict(ablation={"direction_blind_maps": True}),
        "nc_map_family_dense": dict(model={"map_family": "dense"}),
        "nc_h0_only_readout": dict(ablation={"h0_only_readout": True}),
        # the literature row needs r >= the number of listed meta-paths (6 on ACM, spec v1.19 §6)
        "nc_path_source_schemas": dict(data={"path_source": "schemas", "wisp_walks_per_anchor": 12}),
    })
    table = {name: (lambda kw=kw: _nc_record(_config(**kw))) for name, kw in nc.items()}
    table["nc_multilabel_head"] = lambda: _nc_record(_config(), multilabel=True)
    table["lp_dot"] = lambda: _lp_record("dot")
    table["lp_distmult"] = lambda: _lp_record("distmult")
    for pooling in ("mean", "max", "add", "attention"):
        table[f"gc_{pooling}"] = lambda p=pooling: _gc_record(p)
    table["mb_nc_batch_of_scopes"] = _mb_nc_record
    table["mb_lp_batch_with_scopes"] = _mb_lp_record
    return table


def references() -> tuple[dict, dict]:
    """(records, skipped): every case's record; a case that cannot be built is skipped with its error."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if torch.get_num_threads() != 1:
            print("v3b_golden: could not set 1 thread"); sys.exit(2)
        records, skipped = {}, {}
        for name, build in cases().items():
            try:
                records[name] = build()
            except Exception as err:  # reported, never silent: listed in the capture and printed
                skipped[name] = f"{type(err).__name__}: {err}"
        return records, skipped
    finally:
        torch.set_num_threads(threads)


def _flatten(prefix, value, into):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}", v, into)
    else:
        into[prefix] = value


def main() -> None:
    mode, path = sys.argv[1], Path(sys.argv[2])
    if not kernel_path_is_pinned():
        print(f"{PINNED_MESSAGE} (cpu_capability={torch.backends.cpu.get_cpu_capability()}, "
              f"MKL_CBWR={os.environ.get('MKL_CBWR')})")
        sys.exit(2)
    from tests.partMB_fixtures import restore_real_data, unreachable_real_data

    unreachable_real_data()  # synthetic data only: the real data root is unreachable
    try:
        records, skipped = references()
    finally:
        restore_real_data()
    for name, why in skipped.items():
        print(f"  SKIPPED {name}: {why}")
    if mode == "capture":
        torch.save({"records": records, "skipped": skipped}, path)
        total = {}
        _flatten("", records, total)
        print(f"captured {len(records)} configurations, {len(total)} tensors, "
              f"skipped {len(skipped)} -> {path}")
        return
    stored = torch.load(path, weights_only=False)
    ref, now = {}, {}
    _flatten("", stored["records"], ref)
    _flatten("", records, now)
    bad = [k for k in ref if k not in now or ref[k].dtype != now[k].dtype
           or ref[k].shape != now[k].shape or not torch.equal(ref[k], now[k])]
    extra = sorted(set(now) - set(ref))
    print(f"compared {len(ref)} tensors in {len(stored['records'])} configurations; "
          f"differing: {len(bad) or 'none'}{'' if not extra else f'; new {len(extra)}'}")
    for k in bad:
        detail = ""
        if k in now and ref[k].shape == now[k].shape and ref[k].is_floating_point():
            detail = f" max|diff|={float((ref[k].double() - now[k].double()).abs().max()):.3e}"
        print(f"  DIFF {k}{detail}")
    for k in extra:
        print(f"  NEW {k}")
    sys.exit(1 if bad or extra else 0)


if __name__ == "__main__":
    main()
