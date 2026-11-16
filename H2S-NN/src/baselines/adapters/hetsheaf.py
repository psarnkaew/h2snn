"""HetSheaf -- Heterogeneous Sheaf Neural Networks (vendor `Heterogeneous-Sheaf-Neural-Networks` at 6964e1e).

Card 3 Part A (`docs/H2S-NN_hetsheaf_freebase_card.md`), author decisions H-1..H-4 and S-A0-1/S-A0-2
(`docs/hetsheaf_evidence/a0_preconditions.md`):

* **Model code imported, not their training scripts (H-1).** Their sheaf encoders
  (`hetsheaf/models/sheaf_gnn/`) run under the shared protocol; their Lightning/Hydra scripts are not used.
* **Import route (S-A0-1, author).** `import hetsheaf.models.sheaf_gnn` normally runs `hetsheaf/__init__.py:4-10`
  (Lightning data modules and trainers) and `hetsheaf/models/__init__.py:4` (`gnn_baselines/HAN.py`: omegaconf);
  neither `lightning` nor `omegaconf` is installed in h2snn-b. :func:`_vendor` registers `hetsheaf`,
  `hetsheaf.models`, `hetsheaf.datasets` and `hetsheaf.graph_classification` as empty packages whose `__path__`
  is the vendor folder, so the modules we use are imported from their files unchanged while those `__init__`
  files never run. SheafPool's module (`graph_classification/sheaf_graph_classifier.py:10, 14`) and its base
  (`graph_classifier.py:7, 11, 24`) import `lightning.pytorch`; for that import only, `lightning.pytorch` is
  mapped to the installed `pytorch_lightning` (2.6.5; their lock pins `lightning` 2.6.0) and the mapping is
  removed again afterwards (:func:`_lightning_alias`). Nothing is created at registry or package import time.
* **Protocol (H-2).** AdamW on the shared lr / weight decay / dropout axes, no scheduler (their NC
  `CosineAnnealingLR`, `node_classifier.py:178-192`, dropped); their preprocessing on the **shared** splits
  (`SubsetTrainSplit`, `datasets/hgb.py:55`, never applied); NC their head `Linear(hidden_dim, C)`
  (`node_classifier.py:39`, `sheaf_node_classifier.py:37-46`), IMDB `multilabel_soft_margin_loss`
  (`node_classifier.py:66-67`); LP the shared link head of every baseline (spec §6).
* **Search space (H-3).** One row: map family x learner from their sweep lists, `d` 2-5, `layers` 2-8,
  `hidden_channels` {8, 16, 32, 64}, `input_dropout` U[0, 0.9]; lr / weight decay / dropout shared.
* **H2GB NC and large LP (A3; H-4, author 2026-11-16).** Class C (no mini-batch scheme in the source), D1 second
  branch: their HGB NC chain applied **once to the whole H2GB graph** (their code transforms the dataset, and
  `NormalizeFeatures` shifts by each type's global minimum, so a per-subgraph chain would change the features), then
  H2GB's `hgt` sampler with the dataset's H2GB GCN list unchanged (S1) through `minibatch.hgt_loader` /
  `eval_loader` / `train_minibatch_node_classification` (called, not edited); each sampled batch goes through
  `to_homogeneous` (type ids per subgraph, H-4) and their encoder is rebound to it as their inductive classes
  rebind (`inductive/disc_models.py:24-30`; `RebindingEncoder`). Featureless types take `Constant`'s column of ones (their chain), not the
  one-hot rows of the other class-C adapters. Large LP (ogbn-mag P-P only; H-Pokec U-U waits for G-8): D2 on the
  task's message graph (their chain applied to it), a typed copy of the loop of `large_lp_training.py:115-160`
  (author decision: that loop reduces each subgraph to HGB's untyped graph inside itself), calling its helpers.
* **GC (A2; D-P2).** Their inductive encoders (`models/sheaf_gnn/inductive/disc_models.py:18-58`;
  `configs/sheaf_config_gc.yaml:4`, `configs/model/ind_*_sheaf.yaml`: `edge_weights: false`) with their readouts
  as a categorical: mean / max / add (PyG globals), attention (`pooling.py:7-40`, heads 2) and `late_token`
  = SheafPool (`sheaf_graph_classifier.py:33-142`), each followed by their decoder `Linear` (`graph_classifier.py:38`;
  input `hidden_channels` for `late_token`, `hidden_dim` otherwise, `sheaf_graph_classifier.py:161-177`), on the
  shared GC loop (HetSheaf's protocol: constant features, their seed-42 split, batch 1, AdamW on the shared axes
  instead of their hard-coded 1e-3 / 5e-4 and cosine schedule, `graph_classifier.py:155-161`).

Preprocessing per dataset, as the branch of their code that loads it (stated in the report):

* HGB NC (ACM, DBLP, IMDB, Freebase): PyG's `HGBDataset` as their `HGBBaseDataModule.prepare_data`
  (`datasets/hgb.py:50-82`) with `Constant(node_types=None)`, `ToUndirected`, `RemoveSelfLoops`,
  `NormalizeFeatures`, `RemoveDuplicatedEdges` (their order, `SubsetTrainSplit` removed), `coalesce`,
  `to_homogeneous`. The derived LP tasks (graphs of the same HGB datasets; their code has no LP task on them)
  take the same chain on the training message graph.
* LastFM LP: their `LinkPredBase` chain (`datasets/link_pred.py:39-46`): `Constant`, `ToUndirected`,
  `NormalizeFeatures`, `RemoveSelfLoops`; their `RandomLinkSplit` (`:83-92`) not applied.
* PubMed LP: their `HGBLinkPredBase._build_hetero_data` (`datasets/hgb.py:311-357`): one `rev_` relation per
  stored relation (`ei.flip(0)`), `ones(n, 128)` for a type without features, no transform; their
  `RandomLinkSplit` (`:401-410`) not applied.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from src.baselines.budget import budget_kwargs
from src.baselines.bundles import HGB_RAW_DIRNAME, LinkBundle, NodeBundle, bundle_data_root
from src.baselines import search_grids as grids
from src.baselines.heads import SharedLinkHead
from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import vendor_path
from src.baselines.training import train_full_batch_node_classification, train_link_prediction
from src.utils.misc import seed_everything

NAME = "hetsheaf"
VENDOR = "hetsheaf"
SUPPORTED_DATASETS = frozenset({"acm", "dblp", "imdb", "freebase"})
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})
#: A3 (H-4): class C on the H2GB NC datasets.
H2GB_DATASETS = frozenset({"ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"})
READS_BATCH_PLAN = True
#: A3 (D2): the large LP task wired here (H-Pokec U-U: per-type fan-out, decision G-8 of card 1c).
HETSHEAF_LARGE_LP_TASKS = frozenset({"ogbn-mag_pp"})
#: Their GC readouts (`graph_classifier.py:43-54`, `sheaf_graph_classifier.py:184-192`; `scripts/run_pooling_ablation.py:23`).
GC_POOLINGS: tuple[str, ...] = ("mean", "max", "add", "attention", "late_token")
#: Their GC `output_dim` (`run_sheaf_gc.py:40`: `cfg.hidden_dim`, 64 in `configs/sheaf_config_gc.yaml`).
_GC_OUTPUT_DIM = 64
#: LP tasks built on an HGB NC graph: their HGB NC chain (module docstring).
_HGB_NC_LP_DATASETS = frozenset({"acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"})

#: Map family -> their encoder class (`models/sheaf_gnn/__init__.py:4-13`; `configs/model/{diag,bundle,general}_sheaf.yaml:1`).
MAP_FAMILIES: dict[str, str] = {
    "diag": "DiscreteDiagSheafDiffusion",
    "bundle": "DiscreteBundleSheafDiffusion",
    "general": "DiscreteGeneralSheafDiffusion",
}
#: Their learner list per family, from their sweeps for our datasets (e.g. `sweeps/nc/acm/diag_sheaf.yaml:11-21`,
#: `sweeps/nc/acm/bundle_sheaf.yaml`, `general_sheaf.yaml`; the same in every nc/lp/gc sweep of our datasets).
_HETERO_LEARNERS = ("local_concat", "type_concat", "type_ensemble", "edge_type_concat", "node_type_concat",
                    "node_type", "edge_type", "types_only")
_ATTENTION_LEARNERS = ("attention", "attention_type_concat", "attention_type_ensemble", "attention_edge_encoding",
                       "attention_node_encoding", "attention_types_only", "attention_node_type",
                       "attention_edge_type")
FAMILY_LEARNERS: dict[str, tuple[str, ...]] = {
    "diag": _HETERO_LEARNERS,
    "bundle": _HETERO_LEARNERS + _ATTENTION_LEARNERS,
    "general": _HETERO_LEARNERS + _ATTENTION_LEARNERS,
}
#: H-3 as one categorical over the valid (family, learner) pairs (no invalid combination; uniform per pair).
SHEAF_CHOICES: tuple[str, ...] = tuple(f"{family}/{learner}" for family, learners in FAMILY_LEARNERS.items()
                                       for learner in learners)
#: Their LP `output_dim` (`configs/model/{diag,bundle,general}_sheaf.yaml`: 64; not passed by `run_sheaf_lp.py`).
_LP_OUTPUT_DIM = 64
#: Their PubMed fallback width for a type without features (`datasets/hgb.py:337`).
_PUBMED_FALLBACK_DIM = 128


# --------------------------------------------------------------------------------------------------
# Import route (S-A0-1)
# --------------------------------------------------------------------------------------------------

_STUB_PACKAGES = ("hetsheaf", "hetsheaf.models", "hetsheaf.datasets", "hetsheaf.graph_classification")


def _package_dir(name: str) -> Any:
    root = vendor_path(VENDOR)
    return root.joinpath(*name.split("."))


def _register_stub_packages() -> None:
    """Empty package modules over the vendor folders: their `__init__` files are not run."""
    for name in _STUB_PACKAGES:
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        assert spec is not None
        spec.submodule_search_locations = [str(_package_dir(name))]
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)


@contextlib.contextmanager
def _lightning_alias() -> Iterator[None]:
    """`lightning`, `lightning.pytorch` and `lightning.pytorch.utilities(.types)` mapped to the installed
    `pytorch_lightning` for the duration of one import, then removed (S-A0-1)."""
    import pytorch_lightning
    import pytorch_lightning.utilities.types as pl_types

    names = ("lightning", "lightning.pytorch", "lightning.pytorch.utilities", "lightning.pytorch.utilities.types")
    if any(name in sys.modules for name in names):
        raise RuntimeError("a `lightning` module is already loaded; the HetSheaf alias would shadow it")
    top = types.ModuleType("lightning")
    top.pytorch = pytorch_lightning  # type: ignore[attr-defined]
    sys.modules.update({"lightning": top, "lightning.pytorch": pytorch_lightning,
                        "lightning.pytorch.utilities": pytorch_lightning.utilities,
                        "lightning.pytorch.utilities.types": pl_types})
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)


@functools.cache
def _vendor() -> types.SimpleNamespace:
    """Their model code (encoders, model arguments, `RemoveSelfLoops`)."""
    _register_stub_packages()
    sheaf_gnn = importlib.import_module("hetsheaf.models.sheaf_gnn")
    inductive = importlib.import_module("hetsheaf.models.sheaf_gnn.inductive")
    config = importlib.import_module("hetsheaf.models.sheaf_gnn.config")
    utils = importlib.import_module("hetsheaf.datasets.utils")
    return types.SimpleNamespace(
        families={family: getattr(sheaf_gnn, cls) for family, cls in MAP_FAMILIES.items()},
        inductive_families={family: getattr(inductive, f"Inductive{cls}") for family, cls in MAP_FAMILIES.items()},
        SheafModelArguments=config.SheafModelArguments,
        RemoveSelfLoops=utils.RemoveSelfLoops,
        SubsetTrainSplit=utils.SubsetTrainSplit,
    )


@functools.cache
def _vendor_gc() -> types.SimpleNamespace:
    """Their GC readouts: SheafPool (`sheaf_graph_classifier.py:33-142`) and attention pooling (`pooling.py:7-40`)."""
    _vendor()
    with _lightning_alias():
        sheaf_gc = importlib.import_module("hetsheaf.graph_classification.sheaf_graph_classifier")
    pooling = importlib.import_module("hetsheaf.graph_classification.pooling")
    return types.SimpleNamespace(LateTokenSheafReadoutTransport=sheaf_gc.LateTokenSheafReadoutTransport,
                                 AttentionPooling=pooling.AttentionPooling)


# --------------------------------------------------------------------------------------------------
# Their preprocessing
# --------------------------------------------------------------------------------------------------

def hgb_nc_transform() -> Any:
    """Their HGB NC chain (`datasets/hgb.py:51-60`) with `SubsetTrainSplit` removed (H-2)."""
    from torch_geometric import transforms as T

    return T.Compose([
        T.Constant(node_types=None),
        T.ToUndirected(),
        _vendor().RemoveSelfLoops(),
        T.NormalizeFeatures(),
        T.RemoveDuplicatedEdges(),
    ])


def lastfm_lp_transform() -> Any:
    """Their LP chain (`datasets/link_pred.py:39-46`)."""
    from torch_geometric import transforms as T

    return T.Compose([T.Constant(), T.ToUndirected(), T.NormalizeFeatures(), _vendor().RemoveSelfLoops()])


def to_their_homogeneous(hetero: Any) -> Any:
    """`dataset[0].coalesce()` then `to_homogeneous()` (`datasets/hgb.py:63, 74`)."""
    return hetero.coalesce().to_homogeneous()


@dataclass(slots=True)
class SheafGraph:
    """One homogeneous graph in their form: the inputs their encoder reads (`x`, `edge_index`, `node_type`,
    `edge_type`; `disc_models.py:164-200`) and the number of node and edge types."""

    x: torch.Tensor
    edge_index: torch.Tensor
    node_type: torch.Tensor
    edge_type: torch.Tensor
    num_node_types: int
    num_edge_types: int
    node_type_names: tuple[str, ...]

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    def encoder_input(self, device: torch.device) -> Any:
        """A `Data` carrying only what the encoder reads (no label, no mask: label-route hygiene)."""
        from torch_geometric.data import Data

        return Data(x=self.x.to(device), edge_index=self.edge_index.to(device),
                    node_type=self.node_type.to(device), edge_type=self.edge_type.to(device))


def _sheaf_graph(homogeneous: Any) -> SheafGraph:
    return SheafGraph(
        x=homogeneous.x.float(), edge_index=homogeneous.edge_index, node_type=homogeneous.node_type,
        edge_type=homogeneous.edge_type, num_node_types=int(homogeneous.num_node_types),
        num_edge_types=int(homogeneous.num_edge_types), node_type_names=tuple(homogeneous._node_type_names))


@dataclass(slots=True)
class NodeGraph:
    """The HGB NC graph after their chain, with the shared split read back from the chain's output."""

    graph: SheafGraph
    target_rows: torch.Tensor          # homogeneous ids of the target nodes, in target order
    train_mask: torch.Tensor           # over the target nodes, after the chain
    val_mask: torch.Tensor
    test_mask: torch.Tensor


def prepare_node_graph(hetero: Any, target_type: str, *, train_mask: torch.Tensor, val_mask: torch.Tensor,
                       test_mask: torch.Tensor, transform: Any | None = None) -> NodeGraph:
    """Their NC preprocessing on the shared split: the shared masks replace the release's own masks on the target
    type, the chain runs, and the masks are read back from its output (a split-changing transform in the chain
    would change them; `tests/test_baselines_hetsheaf.py` checks they come back unchanged)."""
    hetero = hetero.clone()
    store = hetero[target_type]
    for key in ("train_mask", "val_mask", "test_mask", "y"):
        if key in store:
            del store[key]
    store.train_mask, store.val_mask, store.test_mask = train_mask.bool(), val_mask.bool(), test_mask.bool()
    data = to_their_homogeneous((transform or hgb_nc_transform())(hetero))
    target_index = list(data._node_type_names).index(target_type)
    target_rows = (data.node_type == target_index).nonzero(as_tuple=False).view(-1)
    return NodeGraph(graph=_sheaf_graph(data), target_rows=target_rows, train_mask=data.train_mask[target_rows],
                     val_mask=data.val_mask[target_rows], test_mask=data.test_mask[target_rows])


def _hgb_release(bundle: NodeBundle) -> Any:
    """PyG's `HGBDataset` as their data module loads it (`datasets/hgb.py:61`), never downloading."""
    from src.data.data_layout import nc_root
    from src.data.offline import offline_hgb_dataset

    return offline_hgb_dataset(nc_root(bundle_data_root(bundle)), HGB_RAW_DIRNAME[bundle.dataset])[0]


def _check_target_order(hetero: Any, bundle: NodeBundle) -> None:
    """The release's target rows are the bundle's target rows (same count; same label on every node the release
    labels for training). Guards the mapping from the shared split to their graph."""
    store = hetero[bundle.data.target_node_type]
    count = int(bundle.data.target_y.shape[0])
    if int(store.num_nodes) != count:
        raise ValueError(f"{bundle.dataset}: release has {int(store.num_nodes)} target nodes, bundle {count}")
    if "y" in store and "train_mask" in store:
        rows = store.train_mask.nonzero(as_tuple=False).view(-1)
        if not torch.equal(store.y[rows].to(bundle.data.target_y.dtype), bundle.data.target_y[rows]):
            raise ValueError(f"{bundle.dataset}: release labels differ from the bundle's on the release's training rows")


def link_hetero(data: Any) -> Any:
    """The LP training message graph (`LinkBundle.data`, training positives only) as a `HeteroData`: each type at
    its own width (no `x` for a type without features), each stored relation with its local edge index."""
    from torch_geometric.data import HeteroData

    hetero = HeteroData()
    for name in data.node_type_names:
        start, end = data.node_slices[name]
        dim = int(data.feature_dims[name])
        if dim > 0:
            hetero[name].x = data.x[start:end, :dim].float().cpu()
        hetero[name].num_nodes = end - start
    edge_index, edge_type = data.edge_index.cpu(), data.edge_type.cpu()
    for relation_id, (src, rel, dst) in enumerate(data.relation_tuples):
        mask = edge_type == relation_id
        if not bool(mask.any()):
            continue
        local = edge_index[:, mask] - torch.tensor([[data.node_slices[src][0]], [data.node_slices[dst][0]]])
        hetero[src, rel, dst].edge_index = local
    return hetero


def pubmed_hetero(hetero: Any) -> Any:
    """Their HGB-LP construction (`datasets/hgb.py:331-357`): features as stored, `ones(n, 128)` for a type without
    features, every stored relation followed by its `rev_` relation (`ei.flip(0)`)."""
    from torch_geometric.data import HeteroData

    out = HeteroData()
    for name in hetero.node_types:
        store = hetero[name]
        out[name].x = store.x if "x" in store else torch.ones(int(store.num_nodes), _PUBMED_FALLBACK_DIM)
        out[name].num_nodes = int(store.num_nodes)
    for src, rel, dst in hetero.edge_types:
        ei = hetero[src, rel, dst].edge_index
        out[src, rel, dst].edge_index = ei
        out[dst, f"rev_{rel}", src].edge_index = ei.flip(0)
    return out


def prepare_link_graph(dataset: str, data: Any) -> tuple[SheafGraph, torch.Tensor]:
    """Their preprocessing of the LP message graph for `dataset`, and for every homogeneous node its id in the
    bundle's flattened index space (`global_id[homogeneous id]`)."""
    hetero = link_hetero(data)
    if dataset == "pubmed":
        homogeneous = to_their_homogeneous(pubmed_hetero(hetero))
    elif dataset == "lastfm":
        homogeneous = to_their_homogeneous(lastfm_lp_transform()(hetero))
    elif dataset in _HGB_NC_LP_DATASETS:
        homogeneous = to_their_homogeneous(hgb_nc_transform()(hetero))
    else:
        raise UnsupportedCell(NAME, LINK_PREDICTION, dataset, "no preprocessing branch for this dataset")
    graph = _sheaf_graph(homogeneous)
    global_id = torch.empty(graph.num_nodes, dtype=torch.long)
    for type_index, name in enumerate(graph.node_type_names):
        rows = (graph.node_type == type_index).nonzero(as_tuple=False).view(-1)
        start, end = data.node_slices[name]
        if rows.numel() != end - start:
            raise ValueError(f"{dataset}: type {name} has {rows.numel()} nodes after the chain, {end - start} before")
        global_id[rows] = torch.arange(start, end)
    return graph, global_id


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------

def parse_sheaf_choice(choice: str) -> tuple[str, str]:
    family, _, learner = str(choice).partition("/")
    if family not in FAMILY_LEARNERS or learner not in FAMILY_LEARNERS[family]:
        raise ValueError(f"sheaf must be one of {SHEAF_CHOICES}, got {choice!r}")
    return family, learner


def build_encoder(graph: SheafGraph, params: BaselineParams, device: torch.device, *, output_dim: int,
                  edge_index: torch.Tensor | None = None) -> torch.nn.Module:
    """Their transductive encoder for one map family and learner, on `graph` (their `run_sheaf_nc.py:27-37`
    arguments: graph size, input width, type counts; `configs/model/*_sheaf.yaml` keys `d`, `layers`,
    `hidden_channels`, `input_dropout`, `dropout`; every other argument at their dataclass default,
    `models/sheaf_gnn/config.py:42-91`). `initial_dropout` has no effect in their code and `output_dim` only sizes
    `lin2`, which no forward calls (`disc_models.py:223`; H-2); `output_dim` is set as their scripts set it (NC the
    class count, `run_sheaf_nc.py:33`; LP the model yaml's 64, `configs/model/*_sheaf.yaml`, `run_sheaf_lp.py:46-49`;
    GC the hidden width, `run_sheaf_gc.py:40`)."""
    vendor = _vendor()
    family, learner = parse_sheaf_choice(params.get("sheaf", "diag/type_concat"))
    args = vendor.SheafModelArguments(
        d=int(params.get("d", 2)),
        layers=int(params.get("layers", 2)),
        hidden_channels=int(params.get("hidden_channels", 32)),
        input_dropout=float(params.get("input_dropout", 0.0)),
        dropout=float(params.get("dropout", 0.0)),
        device=str(device),
        graph_size=graph.num_nodes,
        input_dim=int(graph.x.size(1)),
        num_node_types=graph.num_node_types,
        num_edge_types=graph.num_edge_types,
        output_dim=int(output_dim),
    )
    index = graph.edge_index if edge_index is None else edge_index
    return vendor.families[family](index.to(device), args, learner)


def build_inductive_encoder(params: BaselineParams, device: torch.device, *, graph_size: int, input_dim: int,
                            num_node_types: int, num_edge_types: int) -> torch.nn.Module:
    """Their inductive encoder for GC (`run_sheaf_gc.py:35-44`: graph size = the largest training graph, input
    width, type counts, `output_dim` = `cfg.hidden_dim`; `configs/model/ind_*_sheaf.yaml`: `edge_weights: false`)."""
    vendor = _vendor()
    family, learner = parse_sheaf_choice(params.get("sheaf", "diag/type_concat"))
    args = vendor.SheafModelArguments(
        d=int(params.get("d", 2)),
        layers=int(params.get("layers", 2)),
        hidden_channels=int(params.get("hidden_channels", 32)),
        input_dropout=float(params.get("input_dropout", 0.0)),
        dropout=float(params.get("dropout", 0.0)),
        device=str(device),
        graph_size=int(graph_size),
        input_dim=int(input_dim),
        num_node_types=int(num_node_types),
        num_edge_types=int(num_edge_types),
        output_dim=_GC_OUTPUT_DIM,
        edge_weights=False,
    )
    return vendor.inductive_families[family](args, learner)


class HetSheafGraphHead(torch.nn.Module):
    """Their readout and decoder (`SheafGraphClassifier.__init__`, `sheaf_graph_classifier.py:161-192`;
    `GraphClassifier.__init__`, `graph_classifier.py:36-54`) on the encoder's node embeddings."""

    def __init__(self, encoder: torch.nn.Module, pooling: str, num_classes: int) -> None:
        super().__init__()
        from torch_geometric.nn import Linear, global_add_pool, global_max_pool, global_mean_pool

        if pooling not in GC_POOLINGS:
            raise ValueError(f"pooling must be one of {GC_POOLINGS}, got {pooling!r}")
        hidden_dim = int(encoder.hidden_dim)
        stalk_dim = int(getattr(encoder, "final_d", getattr(encoder, "d", 1)))
        channel_dim = int(getattr(encoder, "hidden_channels", hidden_dim // max(stalk_dim, 1)))
        decoder_input_dim = channel_dim if pooling == "late_token" else hidden_dim
        self.pooling = pooling
        self.decoder = Linear(decoder_input_dim, num_classes)
        if pooling == "late_token":
            self.pool: Any = _vendor_gc().LateTokenSheafReadoutTransport(
                stalk_dim=stalk_dim, channel_dim=channel_dim, hidden_dim=None, eps_whiten=1e-4, eps_eigh=1e-6,
                proper_rot=True)
        elif pooling == "attention":
            self.pool = _vendor_gc().AttentionPooling(decoder_input_dim, heads=2)
        else:
            self.pool = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}[pooling]

    def forward(self, embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.pool(embeddings, batch))


class HetSheafNodeClassifier(torch.nn.Module):
    """Their encoder and their NC head `Linear(hidden_dim, C)` (`node_classifier.py:39`; PyG `Linear` as theirs),
    returning the target rows' logits."""

    def __init__(self, encoder: torch.nn.Module, num_classes: int, target_rows: torch.Tensor) -> None:
        super().__init__()
        from torch_geometric.nn import Linear

        self.encoder = encoder
        self.decoder = Linear(encoder.hidden_dim, num_classes)
        self.register_buffer("target_rows", target_rows, persistent=False)

    def forward(self, data: Any) -> torch.Tensor:
        embeddings, _maps = self.encoder(data)
        return self.decoder(embeddings)[self.target_rows]


class HetSheafLinkEncoder(torch.nn.Module):
    """Their encoder; embeddings returned in the bundle's flattened node order (for the shared link head)."""

    def __init__(self, encoder: torch.nn.Module, global_id: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        order = torch.empty_like(global_id)
        order[global_id] = torch.arange(global_id.numel())
        self.register_buffer("order", order, persistent=False)

    @property
    def embed_dim(self) -> int:
        return int(self.encoder.hidden_dim)

    def forward(self, data: Any) -> torch.Tensor:
        embeddings, _maps = self.encoder(data)
        return embeddings[self.order]


def _optimizer(modules: list[torch.nn.Module], params: BaselineParams) -> torch.optim.Optimizer:
    parameters = [p for module in modules for p in module.parameters()]
    return torch.optim.AdamW(parameters, lr=float(params.get("lr", 1e-3)),
                             weight_decay=float(params.get("weight_decay", 0.0)))


# --------------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------------

class HetSheafAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in SUPPORTED_DATASETS or dataset in H2GB_DATASETS
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS or dataset in HETSHEAF_LARGE_LP_TASKS
        if task == GRAPH_CLASSIFICATION:
            return dataset in SUPPORTED_GC_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task not in (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION):
            raise UnsupportedCell(NAME, task, dataset, "unsupported task")
        space: dict[str, Any] = {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "dropout": grids.DROPOUT,
            # their sweeps: input_dropout U[0, 0.9] (`sweeps/nc/acm/diag_sheaf.yaml:35-38`), the shared axis
            "input_dropout": grids.INPUT_DROP,
            "sheaf": {"values": list(SHEAF_CHOICES)},
            "d": {"values": [2, 3, 4, 5]},
            "layers": {"values": [2, 3, 4, 5, 6, 7, 8]},
            "hidden_channels": {"values": [8, 16, 32, 64]},
        }
        if task == LINK_PREDICTION:
            space["decoder_type"] = {"values": ["dot", "distmult"]}  # the shared link head (spec §6)
        if task == GRAPH_CLASSIFICATION:
            space["pooling"] = {"values": list(GC_POOLINGS)}  # their readouts incl. late_token (D-P2)
        if task == NODE_CLASSIFICATION and dataset in H2GB_DATASETS:
            from src.configs.config import MINIBATCH_BATCH_SIZES

            space["batch_size"] = {"values": list(MINIBATCH_BATCH_SIZES)}  # the shared batch range
        return space

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        return load_published_params(NAME, task, dataset)

    def fit_task(self, bundle: Any, params: BaselineParams, *, task: str, seed: int,
                 device: torch.device) -> dict[str, float]:
        if task == NODE_CLASSIFICATION and bundle.dataset in SUPPORTED_DATASETS:
            return self._fit_node_classification(bundle, params, seed=seed, device=device)
        if task == NODE_CLASSIFICATION and bundle.dataset in H2GB_DATASETS:
            return fit_h2gb_node_classification(bundle, params, seed=seed, device=device)
        if task == LINK_PREDICTION and bundle.dataset in HETSHEAF_LARGE_LP_TASKS:
            return fit_large_lp(bundle, params, seed=seed, device=device)
        if task == LINK_PREDICTION and bundle.dataset in SUPPORTED_LP_DATASETS:
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        if task == GRAPH_CLASSIFICATION and bundle.dataset in SUPPORTED_GC_DATASETS:
            return self._fit_graph_classification(bundle, params, seed=seed, device=device)
        raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported cell")

    def _fit_node_classification(self, bundle: NodeBundle, params: BaselineParams, *, seed: int,
                                 device: torch.device) -> dict[str, float]:
        seed_everything(seed, device)
        hetero = _hgb_release(bundle)
        _check_target_order(hetero, bundle)
        node_graph = prepare_node_graph(
            hetero, bundle.data.target_node_type, train_mask=bundle.data.target_train_mask,
            val_mask=bundle.data.target_val_mask, test_mask=bundle.data.target_test_mask)
        return fit_node_graph(node_graph, bundle, params, device=device)

    def _fit_graph_classification(self, bundle: Any, params: BaselineParams, *, seed: int,
                                  device: torch.device) -> dict[str, float]:
        return _fit_graph_classification(bundle, params, seed=seed, device=device)

    def _fit_link_prediction(self, bundle: LinkBundle, params: BaselineParams, *, seed: int,
                             device: torch.device) -> dict[str, float]:
        seed_everything(seed, device)
        graph, global_id = prepare_link_graph(bundle.dataset, bundle.data)
        model = HetSheafLinkEncoder(build_encoder(graph, params, device, output_dim=_LP_OUTPUT_DIM), global_id).to(device)
        head = SharedLinkHead(str(params.get("decoder_type", "dot")),
                              num_relations=max(int(bundle.data.num_target_relations), 1),
                              dim=model.embed_dim).to(device)
        inputs = graph.encoder_input(device)
        record = train_link_prediction(
            model=model, head=head, embed=lambda: model(inputs), data=bundle.data.to(device),
            optimizer=_optimizer([model, head], params), device=device, **budget_kwargs(LINK_PREDICTION, params))
        record["params_source"] = params.source
        record["decoder_type"] = str(params.get("decoder_type", "dot"))
        record["sheaf"] = str(params.get("sheaf", "diag/type_concat"))
        return record


def gc_encoder_input(batch: Any) -> Any:
    """The fields their encoder reads from one GC batch (no label)."""
    from torch_geometric.data import Data

    return Data(x=batch.x.float(), edge_index=batch.edge_index, node_type=batch.node_type.long(),
                edge_type=batch.edge_type.long())


def _fit_graph_classification(bundle: Any, params: BaselineParams, *, seed: int,
                              device: torch.device) -> dict[str, float]:
    from src.baselines.training import train_graph_classification

    seed_everything(seed, device)
    train_loader = bundle.loaders[0]
    graph_size = max(int(graph.num_nodes) for graph in train_loader.dataset)  # their run_sheaf_gc.py:29-32
    encoder = build_inductive_encoder(params, device, graph_size=graph_size, input_dim=bundle.in_channels,
                                      num_node_types=bundle.num_node_types,
                                      num_edge_types=max(int(bundle.fold.dims.num_relations), 1)).to(device)
    pooling = str(params.get("pooling", "late_token"))
    head = HetSheafGraphHead(encoder, pooling, bundle.num_classes).to(device)
    record = train_graph_classification(
        model=encoder, head=head, embed_batch=lambda batch: encoder(gc_encoder_input(batch))[0],
        loaders=bundle.loaders, optimizer=_optimizer([encoder, head], params), device=device,
        **budget_kwargs(GRAPH_CLASSIFICATION, params))
    record["params_source"] = params.source
    record["sheaf"] = str(params.get("sheaf", "diag/type_concat"))
    record["readout"] = pooling
    return record


def fit_node_graph(node_graph: NodeGraph, bundle: NodeBundle, params: BaselineParams, *,
                   device: torch.device) -> dict[str, float]:
    """Full-batch NC on a prepared graph: their encoder and head, the shared loop (AdamW, no scheduler)."""
    model = HetSheafNodeClassifier(build_encoder(node_graph.graph, params, device, output_dim=bundle.num_classes),
                                   bundle.num_classes,
                                   node_graph.target_rows).to(device)
    labels = bundle.data.target_y.to(device)
    if bundle.is_multilabel:
        labels = labels.float()
        criterion = F.multilabel_soft_margin_loss  # their IMDB loss (`node_classifier.py:66-67`)
    else:
        criterion = F.cross_entropy  # theirs (`node_classifier.py:70`)
    inputs = node_graph.graph.encoder_input(device)
    record = train_full_batch_node_classification(
        model=model,
        forward=lambda: model(inputs),
        labels=labels,
        train_idx=node_graph.train_mask.nonzero(as_tuple=False).view(-1).to(device),
        val_idx=node_graph.val_mask.nonzero(as_tuple=False).view(-1).to(device),
        test_idx=node_graph.test_mask.nonzero(as_tuple=False).view(-1).to(device),
        optimizer=_optimizer([model], params),
        criterion=criterion,
        is_multilabel=bundle.is_multilabel,
        **budget_kwargs(NODE_CLASSIFICATION, params),
    )
    record["params_source"] = params.source
    record["sheaf"] = str(params.get("sheaf", "diag/type_concat"))
    return record


# --------------------------------------------------------------------------------------------------
# A3: H2GB NC (class C) and large LP (D2)
# --------------------------------------------------------------------------------------------------

def their_chain_on_graph(data: Any) -> Any:
    """Their HGB NC chain (`hgb_nc_transform`) and `coalesce` on a whole heterogeneous graph (in place)."""
    return hgb_nc_transform()(data).coalesce()


@dataclass(slots=True)
class TypedSchema:
    """Node and edge types of the transformed graph; every sampled batch must reproduce them in this order."""

    node_types: tuple[str, ...]
    edge_types: tuple[tuple[str, str, str], ...]
    input_dim: int

    @classmethod
    def of(cls, data: Any) -> "TypedSchema":
        return cls(tuple(data.node_types), tuple(data.edge_types),
                   max(int(data[t].x.size(1)) for t in data.node_types))


def typed_batch(batch: Any, schema: TypedSchema) -> tuple[SheafGraph, dict[str, torch.Tensor]]:
    """One sampled `HeteroData` batch in their form (`to_homogeneous`: type ids per subgraph, H-4), and for every
    node type the homogeneous rows of its batch nodes in batch order."""
    from torch_geometric.data import HeteroData

    if set(batch.node_types) != set(schema.node_types) or not set(batch.edge_types) <= set(schema.edge_types):
        raise RuntimeError("a sampled batch has a different type schema than the graph")
    # Only what the encoder reads, in the graph's type order (sampler fields, labels and masks left out).
    clean = HeteroData()
    for node_type in schema.node_types:
        clean[node_type].x = batch[node_type].x
    present = set(batch.edge_types)
    for edge_type in schema.edge_types:
        clean[edge_type].edge_index = (batch[edge_type].edge_index if edge_type in present
                                       else torch.empty(2, 0, dtype=torch.long))
    homogeneous = clean.to_homogeneous()
    graph = SheafGraph(x=homogeneous.x.float(), edge_index=homogeneous.edge_index, node_type=homogeneous.node_type,
                       edge_type=homogeneous.edge_type, num_node_types=len(schema.node_types),
                       num_edge_types=len(schema.edge_types), node_type_names=schema.node_types)
    if graph.x.size(1) != schema.input_dim:
        raise RuntimeError(f"batch input width {graph.x.size(1)}, graph {schema.input_dim}")
    rows = {t: (graph.node_type == i).nonzero(as_tuple=False).view(-1) for i, t in enumerate(schema.node_types)}
    return graph, rows


class _SchemaGraph:
    """The type counts and input width the encoder is built for (the graph itself is rebound per batch)."""

    def __init__(self, schema: TypedSchema) -> None:
        self.x = torch.zeros(2, schema.input_dim)
        self.edge_index = torch.tensor([[0, 1], [1, 0]])
        self.num_nodes = 2
        self.num_node_types = len(schema.node_types)
        self.num_edge_types = len(schema.edge_types)


class RebindingEncoder(torch.nn.Module):
    """Their transductive encoder (NC arguments, `edge_weights` True) rebound to each batch as their inductive
    classes rebind (`inductive/disc_models.py:24-30`: graph size, then `update_edge_index`, which also re-indexes the
    edge-weight learners, `transductive/disc_models.py:342-345, 510-513`). Their inductive classes themselves cannot
    be built with `edge_weights` True (their `EdgeWeightLearner` indexes the construction graph, `sheaf_models.py:199-205`;
    their GC configs set it false)."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @property
    def hidden_dim(self) -> int:
        return int(self.encoder.hidden_dim)

    def forward(self, data: Any) -> Any:
        self.encoder.graph_size = int(data.x.size(0))
        self.encoder.update_edge_index(data.edge_index)
        return self.encoder(data)


def build_rebinding_encoder(schema: TypedSchema, params: BaselineParams, device: torch.device, *,
                            output_dim: int) -> torch.nn.Module:
    """`build_encoder`'s model (NC arguments) on a two-node placeholder graph, rebound to every batch."""
    return RebindingEncoder(build_encoder(_SchemaGraph(schema), params, device, output_dim=output_dim))


class HetSheafBatchClassifier(torch.nn.Module):
    """Their rebinding encoder and their NC head on one sampled batch; returns the seed targets' logits."""

    def __init__(self, encoder: torch.nn.Module, num_classes: int) -> None:
        super().__init__()
        from torch_geometric.nn import Linear

        self.encoder = encoder
        self.decoder = Linear(encoder.hidden_dim, num_classes)

    def forward(self, graph: SheafGraph, seed_rows: torch.Tensor, device: torch.device) -> torch.Tensor:
        embeddings, _maps = self.encoder(graph.encoder_input(device))
        return self.decoder(embeddings[seed_rows.to(device)])


def fit_h2gb_node_classification(bundle: Any, params: BaselineParams, *, seed: int,
                                 device: torch.device) -> dict[str, float]:
    """Class C on H2GB's `hgt` batches (module docstring, A3)."""
    from src.baselines.minibatch import (CLASS_C_REFERENCE_METHOD, eval_loader, h2gb_graph, h2gb_sample_sizes,
                                         hgt_loader, minibatch_plan, train_minibatch_node_classification)
    from src.data.h2gb_nc import ALL_H2GB_NC_SPECS

    seed_everything(seed, device)
    data = their_chain_on_graph(h2gb_graph(bundle))
    schema = TypedSchema.of(data)
    target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
    fanout = h2gb_sample_sizes(CLASS_C_REFERENCE_METHOD, bundle.dataset, schema.node_types)
    plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
    labels = data[target].y.long()
    num_classes = int(labels.max()) + 1
    model = HetSheafBatchClassifier(build_rebinding_encoder(schema, params, device, output_dim=num_classes),
                                    num_classes).to(device)

    def forward_seeds(batch: Any) -> torch.Tensor:
        graph, rows = typed_batch(batch, schema)
        return model(graph, rows[target][:int(batch[target].batch_size)], device)

    record = train_minibatch_node_classification(
        model=model, forward_seeds=forward_seeds,
        train_loader=hgt_loader(data, fanout, target, "train", plan.batch_size, shuffle=True,
                                steps=plan.steps_per_epoch),
        val_loader=eval_loader(data, fanout, target, "val", plan.batch_size),
        test_loader=eval_loader(data, fanout, target, "test", plan.batch_size),
        val_ids=data[target].val_mask.nonzero().view(-1), test_ids=data[target].test_mask.nonzero().view(-1),
        target_type=target, optimizer=_optimizer([model], params), criterion=F.cross_entropy, plan=plan,
        report=getattr(params, "report", None), labels=labels)
    record["params_source"] = params.source
    record["sheaf"] = str(params.get("sheaf", "diag/type_concat"))
    record["batch_size"] = float(plan.batch_size)
    record["sampler_depth"] = float(max(len(v) for v in fanout.values()))
    return record


class HetSheafBatchEncoder(torch.nn.Module):
    """Their rebinding encoder on one sampled batch; returns the rows of `node_type`'s batch nodes (batch order)."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @property
    def embed_dim(self) -> int:
        return int(self.encoder.hidden_dim)

    def forward(self, batch: Any, schema: TypedSchema, node_type: str, device: torch.device) -> torch.Tensor:
        graph, rows = typed_batch(batch, schema)
        embeddings, _maps = self.encoder(graph.encoder_input(device))
        return embeddings[rows[node_type].to(device)]


def _replace_in_place(target: Any, source: Any) -> None:
    """Make HeteroData `target` hold `source`'s node and edge stores (same object, so every holder of it sees them)."""
    for edge_type in [t for t in target.edge_types if t not in source.edge_types]:
        del target[edge_type]
    for node_type in [t for t in target.node_types if t not in source.node_types]:
        del target[node_type]
    for stores in (source.node_types, source.edge_types):
        for key in stores:
            for name in [k for k in target[key].keys() if k not in source[key].keys()]:
                del target[key][name]
            for name, value in source[key].items():
                target[key][name] = value


def build_large_lp(d: Any, params: Any, device: torch.device, sampler: Any = None) -> tuple[Any, Any]:
    """The shared large-LP loop's `build` (card 1c F7 convention, `large_lp_training.fit_large_lp`): their chain applied
    to the task's message graph `d.graph` (the loop samples from it), HetSheaf's typed per-batch embedding (type ids
    per subgraph, the rebinding encoder) and H2GB's sampler of the class-C reference method (`h2gb_sampler`: per-hop
    `link` on ogbn-mag P-P); `sampler` replaces it in the tests."""
    from src.baselines import large_lp_training as llp
    from src.baselines.minibatch import CLASS_C_REFERENCE_METHOD

    _replace_in_place(d.graph, their_chain_on_graph(d.graph))  # whoever samples `d.graph` sees their chain
    schema = TypedSchema.of(d.graph)
    encoder = HetSheafBatchEncoder(build_rebinding_encoder(schema, params, device, output_dim=_LP_OUTPUT_DIM))
    model = llp.LargeLPModel(modules=encoder, embed=lambda batch, train: encoder(batch, schema, d.node_type, device),
                             embed_dim=encoder.embed_dim)
    return model, sampler or llp.h2gb_sampler(CLASS_C_REFERENCE_METHOD, d.task, schema.node_types)


def fit_large_lp(bundle: Any, params: BaselineParams, *, seed: int, device: torch.device) -> dict[str, float]:
    """D2 on the task's message graph through the shared large-LP loop (`large_lp_training.fit_large_lp`, card 1c F7:
    steps of `LP_POSITIVES_PER_STEP` positives in a seeded continued order, one HGB tail negative per positive per
    epoch, evaluation endpoints computed once, best validation AUROC, the LP budget) with `build_large_lp`."""
    from src.baselines import large_lp_training as llp

    record = llp.fit_large_lp(bundle, params, seed=seed, device=device,
                              build=lambda d: build_large_lp(d, params, device))
    record["params_source"] = params.source
    record["sheaf"] = str(params.get("sheaf", "diag/type_concat"))
    return record


ADAPTER = HetSheafAdapter()
