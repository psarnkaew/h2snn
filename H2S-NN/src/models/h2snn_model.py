"""H2S-NN v3-B (spec v1.11 R13; plan §4, decision D6): one anchored quadratic solve on the three-level
sheaf energy (`src/models/sheaf_energy.py`),

    h0 = TypeEncoder(x),   (P^T P + tau L_hat) Y* = P^T h0,   Y = (x, s, z),   tau = exp(theta_tau),
    representation = x*            (spec 1.25 Z3; [x*, h0] under the arm `ablation.ego_head`),

solved by PCG with an implicit backward (`src/models/anchored_solve.py`). `Delta_0` runs over the
observed facts with the D1 map operators (one conformal scale per fact, D1a); every path 2-cell
couples to every node on it through ordered parallel transport, with weight `beta_f = w_f m_f`,
`w_f = exp(g(path_latent_f))`. The arms of spec §6 are configurations of this one path (`model.arm`).

One encoder, three task heads (plan §4.8), selected by `data.task_type`:
- node classification: `MLP(x*)` on the target nodes;
- link prediction: the `dot` / `distmult` decoders on `x*` (spec v1.22 W2);
- graph classification: pooling of `x*` per graph -> one Linear layer
(spec 1.25 Z3; `[x*, h0]` under the arm `ablation.ego_head`, the 1.24 readout).
The v2 propagation (Chebyshev filter bank / first-order update, listen/broadcast gates, pair
attention, the v2 path switches, identity maps, the dense map family) was removed by the
legacy-removal card (v1/v2).
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple

import torch
from torch import nn

from src.configs.config import TrainConfig, is_literature_row, literature_lengths
from src.data.link_pred_structures import PreparedLinkPredData
from src.data.structures import FaceGroup, PreparedH2SData, ScopeIndex, face_lengths_of
from src.models.graph_readout import GraphReadout
from src.models.link_readout import LinkPredictionReadout
from src.models.map_operators import (
    EdgeMapPair,
    IdentityMapOperator,
    PathTransportOperator,
    TD1MapOperator,
    checkpoint_if_grad,
    orthonormal_subspaces,
    skew_from_coefficients,
)
from src.models.anchored_solve import (LogWeightRangeStopError, NonFiniteLogWeightError,
                                       PCGConvergenceError, SolveStats, TauStopError,
                                       anchored_solve, solver_tolerance)
from src.models.sheaf_energy import (
    EliminatedOperator,
    IncidenceComplex,
    SheafEnergyOperator,
    build_cell_group,
)
from src.models.node_encoder import TypeSpecificNodeEncoder
from src.models.orthogonal import Orthogonal
from src.models.sheaf_maps import TD1Factors, TD1NodeToEdgeSheaf, validate_map_rank
from src.models.vertex_path_laplacian import HolonomyEdgeTerm


def _per_type_layout(
    node_type_names: tuple[str, ...],
    feature_dims: "Mapping[str, int]",
    node_slices: "Mapping[str, tuple[int, int]]",
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Raw feature width, node count, and flat-id start offset for each node type.

    Ordered by node-type id: `stack_node_features` assigns ids with
    `enumerate(node_types)` and returns that same tuple as `node_type_names`, so
    position `i` here is the type whose nodes carry `node_type == i`.
    """
    dims: list[int] = []
    counts: list[int] = []
    offsets: list[int] = []
    for name in node_type_names:
        start, end = node_slices[name]
        dims.append(int(feature_dims[name]))
        counts.append(int(end - start))
        offsets.append(int(start))
    return tuple(dims), tuple(counts), tuple(offsets)


class ModelDims(NamedTuple):
    """Task-agnostic sizing info H2SNN needs to build its layers.

    Decouples H2SNN.__init__ from PreparedH2SData's concrete shape so the same
    model class can be constructed from link-prediction/graph-classification data
    containers too — each task's data builder produces a ModelDims of its own.
    """

    in_channels: int
    num_node_types: int
    num_relations: int
    num_classes: int
    num_nodes: int | None = None  # required only when in_channels == 0 (feature-less graphs)
    num_target_relations: int = 1  # > 1 only for link-prediction datasets pooling relations (Amazon)
    # Per-node-type raw feature width / node count / start index in the flattened node
    # id space, ordered to match `node_type_names`. Supplied by the heterogeneous data
    # builders so the encoder can size one linear map per type at its *own* width, as
    # HGB does. Left None by graph classification, whose types share a single width.
    type_input_dims: tuple[int, ...] | None = None
    type_num_nodes: tuple[int, ...] | None = None
    type_offsets: tuple[int, ...] | None = None
    # Distinct path lengths L present in the face groups, ascending. The "concat" face
    # encoder sizes one input projection per length, and they must be built in __init__:
    # a projection created lazily on first forward would be registered after
    # build_optimizer() has already captured model.parameters(), so it would never train.
    face_lengths: tuple[int, ...] = ()
    # Relation ids (compact, after the fact layer) with rho(r) = r: their source and target
    # roles coincide (D7). Empty under `fact_layer: untied` and when no layer was built.
    self_inverse_relations: tuple[int, ...] = ()
    # Part H (WISP): cells of one length group differ in relation sequence, so the path encoders
    # read a per-cell action embedding (relation embedding + one-hot hop direction -1/0/+1) of
    # width d_r + 3 per hop; False = one relation token per group (schema and DFS builders).
    path_actions: bool = False

    @classmethod
    def from_prepared_data(cls, data: PreparedH2SData) -> "ModelDims":
        type_dims, type_counts, type_offsets = _per_type_layout(
            node_type_names=data.node_type_names,
            feature_dims=data.feature_dims,
            node_slices=data.node_slices,
        )
        return cls(
            in_channels=data.x.size(1),
            num_node_types=len(data.node_type_names),
            num_relations=len(data.relation_names),
            num_classes=data.num_classes,
            num_nodes=data.x.size(0),
            type_input_dims=type_dims,
            type_num_nodes=type_counts,
            type_offsets=type_offsets,
            face_lengths=face_lengths_of(data.face_groups),
            self_inverse_relations=_self_inverse_of(data),
        )

    @classmethod
    def from_link_pred_data(cls, data: PreparedLinkPredData) -> "ModelDims":
        type_dims, type_counts, type_offsets = _per_type_layout(
            node_type_names=data.node_type_names,
            feature_dims=data.feature_dims,
            node_slices=data.node_slices,
        )
        return cls(
            in_channels=data.x.size(1),
            num_node_types=len(data.node_type_names),
            num_relations=len(data.relation_names),
            num_classes=0,  # unused — link prediction builds link_readout, not NodeReadout
            num_nodes=data.x.size(0),
            num_target_relations=data.num_target_relations,
            type_input_dims=type_dims,
            type_num_nodes=type_counts,
            type_offsets=type_offsets,
            face_lengths=face_lengths_of(data.face_groups),
            self_inverse_relations=_self_inverse_of(data),
        )


def _self_inverse_of(data: object) -> tuple[int, ...]:
    """Self-inverse relation ids of the data's fact layer (empty without one)."""
    layer = getattr(data, "fact_layer", None)
    return () if layer is None else tuple(layer.self_inverse)


class AnchoredState(NamedTuple):
    """What one anchored forward solved with (v3-B)."""

    maps: EdgeMapPair
    path_weights: tuple[torch.Tensor, ...]  # w_f per face group (before m_f)
    tau: torch.Tensor                        # detached
    stats: SolveStats
    layout_rows: tuple[int, int, int]        # (N, |fact states|, F) of the solved system


class EncodedState(NamedTuple):
    """Output of the shared encoder, before any task head."""

    h0: torch.Tensor
    x_star: torch.Tensor  # x* (h0 under ablation.h0_only_readout)
    # x* (spec 1.25 Z3); [x*, h0] under ablation.ego_head; h0 under ablation.h0_only_readout;
    # the instance feature appended under the arm instance_features
    representation: torch.Tensor
    operator: "AnchoredState | None"  # None under ablation.h0_only_readout


class H2SNNOutput(NamedTuple):
    """Node-classification output."""

    logits: torch.Tensor
    x_star: torch.Tensor
    h0: torch.Tensor
    # Detached diagnostics for logging (no effect on optimization).
    path_gate_mean: torch.Tensor | None
    path_gate_max: torch.Tensor | None
    max_edge_scale: torch.Tensor | None


class PathEncoder(nn.Module):
    """Positional `concat` encoder over the ordered `h0` of a path and its hop tokens (the schema's
    relation embeddings, or per-cell action embeddings under WISP): one `Linear + GELU` per path
    length; its output feeds the path gate."""

    def __init__(self, d: int, d_r: int, hidden_dim: int, face_lengths: tuple[int, ...]) -> None:
        """`d`: width of the per-node input (`h0`; spec 1.25 Z1: no relational context).
        `d_r`: width of one hop token (d_r, or d_r + 3 with per-cell actions)."""
        super().__init__()
        self.input_proj = nn.ModuleDict(
            {
                str(length): nn.Linear((length + 1) * d + length * d_r, hidden_dim)
                for length in face_lengths
            }
        )
        self.activation = nn.GELU()
        self.latent_dim = hidden_dim

    def forward(
        self,
        h0: torch.Tensor,
        face_node_index: torch.Tensor,
        hop_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """`[P, L+1]` paths with hop tokens `[1 or P, L * d_r]` -> `[P, hidden]` latents."""
        length = int(face_node_index.size(1)) - 1
        key = str(length)
        if key not in self.input_proj:
            available = ", ".join(sorted(self.input_proj)) or "none"
            raise KeyError(
                f"PathEncoder has no projection for path length {length} (built for: "
                f"{available}). ModelDims.face_lengths must list every length in the data."
            )
        num_paths = face_node_index.size(0)
        inputs = torch.cat(
            [h0[face_node_index].reshape(num_paths, -1), hop_tokens.expand(num_paths, -1)],
            dim=-1,
        )
        return self.activation(self.input_proj[key](inputs))


class EndpointEncoder(nn.Module):
    """Latent of a path from its endpoints and hop tokens only, `[h_{v_0}, h_{v_L}, tokens]`
    (ablation `path_mode: endpoint_edges`, and the `endpoints` arm of the C2 weight ladder: no
    intermediate node enters)."""

    def __init__(self, d: int, d_r: int, hidden_dim: int, face_lengths: tuple[int, ...]) -> None:
        super().__init__()
        self.input_proj = nn.ModuleDict(
            {str(length): nn.Linear(2 * d + length * d_r, hidden_dim) for length in face_lengths}
        )
        self.activation = nn.GELU()
        self.latent_dim = hidden_dim

    def forward(
        self,
        h0: torch.Tensor,
        face_node_index: torch.Tensor,
        hop_tokens: torch.Tensor,
    ) -> torch.Tensor:
        length = int(face_node_index.size(1)) - 1
        num_paths = face_node_index.size(0)
        endpoints = face_node_index[:, [0, length]]
        inputs = torch.cat(
            [h0[endpoints].reshape(num_paths, -1), hop_tokens.expand(num_paths, -1)], dim=-1
        )
        return self.activation(self.input_proj[str(length)](inputs))


class SchemaEncoder(nn.Module):
    """`schema` arm of the C2 weight ladder: the latent depends on the hop tokens (relation
    sequence, and directions under WISP) and the length only, so cells with the same sequence
    get the same weight whatever their nodes."""

    def __init__(self, d_r: int, hidden_dim: int, face_lengths: tuple[int, ...]) -> None:
        super().__init__()
        self.input_proj = nn.ModuleDict(
            {str(length): nn.Linear(length * d_r, hidden_dim) for length in face_lengths}
        )
        self.activation = nn.GELU()
        self.latent_dim = hidden_dim

    def forward(self, h0: torch.Tensor, face_node_index: torch.Tensor,
                hop_tokens: torch.Tensor) -> torch.Tensor:
        length = int(face_node_index.size(1)) - 1
        latent = self.activation(self.input_proj[str(length)](hop_tokens))
        return latent.expand(face_node_index.size(0), -1)


def endpoint_hidden_width(full_hidden: int, node_dim: int, hop_dim: int,
                          face_lengths: tuple[int, ...]) -> int:
    """Hidden width of the `endpoints` arm whose g (per-length first layer + gate) has the
    parameter count closest to `full`'s (Part H capacity rule; both counts are linear in the
    width, so rounding gives the minimiser)."""
    lengths = tuple(face_lengths) or (2,)
    full_inputs = sum((length + 1) * node_dim + length * hop_dim + 1 for length in lengths)
    endpoint_inputs = sum(2 * node_dim + length * hop_dim + 1 for length in lengths)
    full_params = full_hidden * (full_inputs + 1) + 1
    return max(1, round((full_params - 1) / (endpoint_inputs + 1)))


def _rotation_parameters(count: int, orthogonal: Orthogonal, d: int) -> nn.Parameter:
    """FastHPP parameters of `count` orthogonal frames, drawn N(0, 1/d) as in D1."""
    parameters = nn.Parameter(torch.empty(count, orthogonal.parameter_count))
    nn.init.normal_(parameters, mean=0.0, std=1.0 / math.sqrt(d))
    return parameters


class FreeIncidenceMaps(nn.Module):
    """Arm `free_cell_maps` (C17; Thm 3 control): one free map per fact-cell incidence from the TD-1
    family of the fact maps (spec 1.25; author 2026-11-18, option A),
    `R_position exp(U_position C_{v,f} U_position^T)`, with one FastHPP frame and one `U` per hop
    position and `C_{v,f}` in so(m) from an MLP over `[h_v, path_latent_f, onehot(position)]` whose
    output layer is zero-initialised (`C = 0` at init). Not forced by functoriality."""

    def __init__(self, d: int, latent_dim: int, hidden_dim: int, map_rank: int,
                 max_positions: int, orthogonal_map: str) -> None:
        super().__init__()
        validate_map_rank(map_rank, d)
        self.d = d
        self.map_rank = map_rank
        self.num_skew = map_rank * (map_rank - 1) // 2
        self.max_positions = max_positions
        self.orthogonal = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        self.frame_params = _rotation_parameters(max_positions, self.orthogonal, d)
        self.subspace_params = nn.Parameter(torch.empty(max_positions, d, map_rank))
        nn.init.normal_(self.subspace_params, mean=0.0, std=1.0)
        self.skew_mlp = nn.Sequential(
            nn.Linear(d + latent_dim + max_positions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_skew),
        )
        nn.init.zeros_(self.skew_mlp[-1].weight)
        nn.init.zeros_(self.skew_mlp[-1].bias)

    def factors(self, h0: torch.Tensor, face_node_index: torch.Tensor, latent: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(exp(C) - I [P * size, m, m], U [positions, d, m], frames [positions, d, d])`; row
        `p * size + i` is incidence `i` of cell `p`."""
        num_paths, size = face_node_index.shape
        positions = torch.arange(size, device=h0.device)
        onehot = nn.functional.one_hot(positions, self.max_positions).to(h0.dtype)
        inputs = torch.cat(
            [
                h0[face_node_index],
                latent.unsqueeze(1).expand(num_paths, size, -1),
                onehot.unsqueeze(0).expand(num_paths, size, -1),
            ],
            dim=-1,
        ).reshape(num_paths * size, -1)
        generator = skew_from_coefficients(self.skew_mlp(inputs), self.map_rank)
        eye = torch.eye(self.map_rank, dtype=generator.dtype, device=generator.device)
        return (torch.linalg.matrix_exp(generator) - eye,
                orthonormal_subspaces(self.subspace_params), self.orthogonal(self.frame_params))

    def operator(self, rotation: torch.Tensor, subspaces: torch.Tensor, frames: torch.Tensor,
                 num_paths: int, length: int) -> TD1MapOperator:
        """The free maps of `num_paths` cells of `length` hops (scale 1: free maps are orthogonal)."""
        positions = int(subspaces.size(0))
        return TD1MapOperator(
            scale=rotation.new_ones(num_paths * length), rotation=rotation,
            group_ids=torch.arange(length, device=rotation.device).repeat(num_paths),
            subspaces=subspaces, frames=frames,
            instance_groups=tuple(True for _ in range(positions)),
            frame_index=tuple(range(positions)))

    def forward(self, h0: torch.Tensor, face_node_index: torch.Tensor,
                latent: torch.Tensor) -> TD1MapOperator:
        num_paths, size = face_node_index.shape
        return self.operator(*self.factors(h0, face_node_index, latent), num_paths, size)


_LINK_PREDICTION = "link_prediction"
_GRAPH_CLASSIFICATION = "graph_classification"


class H2SNN(nn.Module):
    def __init__(self, config: TrainConfig, dims: ModelDims) -> None:
        super().__init__()
        model_config = config.model
        ablation = config.ablation
        self.config = config
        self.model_config = model_config
        self.task_type = config.data.task_type
        d = model_config.d
        self.d = d
        self.checkpointing = model_config.checkpointing
        # D1a: one conformal scale per fact (the per-side scales were removed with v2).
        self.map_scale = "per_edge"
        self.h0_only = ablation.h0_only_readout
        self.propagation = model_config.propagation
        self.arm = model_config.arm
        anchored = self.propagation == "anchored" and not self.h0_only
        self.anchored = anchored
        # Arm "maps = I" (spec 1.23 X-11; S2 with arm full, S0 with arm no_path_cells): every fact
        # map's orthogonal factor is I, so every transport is I; c_e, g, beta_f, tau, solver and head
        # unchanged; the frames, W_r and the skew rows of the edge MLP are not parameters.
        self.identity_maps = ablation.fact_maps_identity

        self.node_encoder: TypeSpecificNodeEncoder | None = None
        self.identity_embedding: nn.Embedding | None = None
        if dims.in_channels == 0:
            if dims.num_nodes is None:
                raise ValueError("ModelDims.num_nodes is required when in_channels == 0.")
            self.identity_embedding = nn.Embedding(dims.num_nodes, d)
        else:
            self.node_encoder = TypeSpecificNodeEncoder(
                input_dim=dims.in_channels,
                hidden_dim=d,
                num_node_types=dims.num_node_types,
                dropout=model_config.encoder_dropout,
                type_input_dims=dims.type_input_dims,
                type_num_nodes=dims.type_num_nodes,
                type_offsets=dims.type_offsets,
            )
        self.relation_embedding = nn.Embedding(dims.num_relations, model_config.d_r)
        # rho(r) = r relations of the data's fact layer (the GC path builder needs them).
        self.self_inverse_relations = tuple(dims.self_inverse_relations)
        # Shared roles (D7): relations with rho(r) = r, or every relation under the
        # direction-blind ablation.
        self.shared_role_relations = (
            tuple(range(dims.num_relations)) if ablation.direction_blind_maps
            else tuple(dims.self_inverse_relations)
        )
        # Spec 1.25 Z1: the geometry (maps, fact weights, g) reads each endpoint's h0 only; the
        # relational context c is removed.
        # Spec 1.25 Z2: TD-1 maps, one frame per relation with rho(r) != r, rank m = map_rank.
        self.node_to_edge = TD1NodeToEdgeSheaf(
            d=d,
            d_r=model_config.d_r,
            num_relations=dims.num_relations,
            hidden_dim=model_config.edge_mlp_hidden_dim,
            map_rank=model_config.map_rank,
            scale_eps=model_config.map_scale_eps,
            orthogonal_map=model_config.orthogonal_map,
            checkpoint=model_config.checkpointing,
            shared_role_relations=self.shared_role_relations,
            identity_maps=self.identity_maps,
        )
        # N2c (pre-registration §4.3): C_e = 0 in evaluation forwards only (a training forward raises).
        self.type_only_maps = config.evaluation.type_only_maps
        self.path_encoder: PathEncoder | EndpointEncoder | SchemaEncoder | None = None
        self.path_gate: nn.Linear | None = None
        self.path_log_weight: nn.Parameter | None = None
        # Part H: w_f = exp(g) without gamma; the C2 ladder restricts the inputs of g on identical
        # cells.
        self.path_weight_input = ablation.path_weight_input
        # Spec v1.4 (R12): g, i.e. the path encoder and its gate, reads each instance node's h0.
        path_input_dim = d
        self.path_actions = dims.path_actions
        hop_dim = model_config.d_r + (3 if dims.path_actions else 0)
        hidden = model_config.face_mlp_hidden_dim
        if self.path_weight_input == "full":
            self.path_encoder = PathEncoder(path_input_dim, hop_dim, hidden, dims.face_lengths)
        elif self.path_weight_input == "endpoints":
            hidden = endpoint_hidden_width(hidden, path_input_dim, hop_dim, dims.face_lengths)
            self.path_encoder = EndpointEncoder(path_input_dim, hop_dim, hidden, dims.face_lengths)
        elif self.path_weight_input == "schema":
            self.path_encoder = SchemaEncoder(hop_dim, hidden, dims.face_lengths)
        if self.path_weight_input != "uniform":
            # Gate g, default weight init so the path encoder gets gradients from step 0; bias
            # b = log(path_weight_init), w_f = exp(g) starts near path_weight_init.
            self.path_gate = nn.Linear(hidden, 1)
            nn.init.constant_(self.path_gate.bias, math.log(model_config.path_weight_init))
        else:
            # uniform: w_f = exp(b) for every cell (b alone; no g input).
            self.path_log_weight = nn.Parameter(
                torch.tensor(math.log(model_config.path_weight_init)))
        self.last_path_stats: dict[str, float] = {}
        self.last_path_weights: tuple[torch.Tensor, ...] = ()

        # Task head on the node representation x* (spec 1.25 Z3; h0 under h0_only_readout; [x*, h0]
        # under the arm ego_head, the 1.24 readout); the v3-B arm `instance_features` adds the anchor's
        # cell feature sum_f beta_f mean_v T_{v,f} h_v (C14).
        self.ego_head = ablation.ego_head
        representation_dim = d if self.h0_only else (2 * d if self.ego_head else d)
        if anchored and self.arm == "instance_features":
            representation_dim += d
        self.readout: nn.Sequential | None = None
        self.link_readout: LinkPredictionReadout | None = None
        self.graph_readout: GraphReadout | None = None
        if self.task_type == _LINK_PREDICTION:
            self.link_readout = LinkPredictionReadout(
                d=d,
                input_dim=representation_dim,
                decoder_type=config.data.decoder_type,
                num_relations=dims.num_target_relations,
            )
        elif self.task_type == _GRAPH_CLASSIFICATION:
            self.graph_readout = GraphReadout(
                input_dim=representation_dim,
                num_classes=dims.num_classes,
                pooling=model_config.graph_pooling,
            )
        else:
            self.readout = nn.Sequential(
                nn.Linear(representation_dim, model_config.classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(model_config.classifier_dropout),
                nn.Linear(model_config.classifier_hidden_dim, dims.num_classes),
            )
        # v3-B (created last, as before the legacy removal). tau = exp(theta_tau),
        # theta_tau = log(tau_init), own optimizer group (D-B4). The arm `free_cell_maps` has one
        # free orthogonal map per fact-cell incidence (hop positions 0..L_max-1; Thm 3 control).
        self.theta_tau: nn.Parameter | None = None
        self.free_cell_maps: FreeIncidenceMaps | None = None
        self.stop_limits = (model_config.tau_stop, model_config.log_weight_range_stop)
        self._current_epoch = -1
        self.solver_settings = (model_config.solver_warmup_epochs,
                                model_config.solver_warmup_tolerance,
                                model_config.solver_tolerance,
                                model_config.solver_max_iterations)
        # Statistics of the last anchored solve: the PCG counts live in `_last_solve` (the backward
        # fills its half after `forward` returns), the layout numbers in `_last_solve_info`.
        self._last_solve: SolveStats | None = None
        self._last_solve_info: dict[str, float] = {}
        self._warm_start: torch.Tensor | None = None
        self._warm_start_key: tuple = ()
        if anchored:
            self.theta_tau = nn.Parameter(torch.tensor(math.log(model_config.tau_init)))
            if self.arm == "free_cell_maps":
                latent_dim = (self.path_encoder.latent_dim if self.path_encoder is not None
                              else model_config.face_mlp_hidden_dim)
                self.free_cell_maps = FreeIncidenceMaps(
                    d=d, latent_dim=latent_dim, hidden_dim=model_config.edge_mlp_hidden_dim,
                    map_rank=model_config.map_rank,
                    max_positions=max(dims.face_lengths, default=1),
                    orthogonal_map=model_config.orthogonal_map,
                )

    def _type_only(self) -> bool:
        """N2c type-only read-out (`evaluation.type_only_maps`): evaluation forwards only."""
        if not self.type_only_maps:
            return False
        if self.training:
            raise ValueError("evaluation.type_only_maps (N2c, C_e = 0) is an evaluation-only read-out "
                             "of a trained checkpoint; a training forward with it is refused.")
        return True

    # ------------------------------------------------------------------ scalars
    def tau(self) -> torch.Tensor | None:
        """`exp(theta_tau)` of the anchored solve; None under `h0_only_readout`."""
        if self.theta_tau is None:
            return None
        return torch.exp(self.theta_tau)

    def diffusion_scalars(self) -> dict[str, float]:
        """tau and the last solve's PCG statistics, and log w_f of the last forward, for per-epoch
        logging."""
        with torch.no_grad():
            scalars: dict[str, float] = {}
            tau = self.tau()
            if tau is not None:
                scalars["tau"] = float(tau)
                scalars.update(self.last_solver_stats)
        scalars.update(self.last_path_stats)      # log w_f of the last forward (Part H)
        return scalars

    @property
    def last_solver_stats(self) -> dict[str, float]:
        """PCG iterations and residuals of the last solve (forward and, once run, backward), the
        tolerance used, and the sizes of the solved system (`|E_P|`, `|E_shared|`, states, cells,
        eliminated share)."""
        if self._last_solve is None:
            return {}
        return {**self._last_solve.as_dict(), **self._last_solve_info}

    def scalar_parameters(self) -> list[nn.Parameter]:
        """`theta_tau` (v3-B, D-B4) and the bias b of g (or the uniform arm's b): the optimizer gives
        them their own group (plan §12.4)."""
        scalars = [p for p in (self.theta_tau,) if p is not None]
        # b, the bias of g (gamma's role under exp, spec v1.3 §3.2c), or the uniform arm's b.
        if self.path_gate is not None:
            scalars.append(self.path_gate.bias)
        if self.path_log_weight is not None:
            scalars.append(self.path_log_weight)
        return scalars

    # ------------------------------------------------------------------ inputs
    def encode_nodes(self, x: torch.Tensor, node_type: torch.Tensor,
                     global_node_id: torch.Tensor | None = None) -> torch.Tensor:
        """`h0`. `global_node_id` (Part MB): rows of a batch of scopes -> node ids of the whole
        graph, for the per-node embeddings of featureless types; None in full batch."""
        if self.node_encoder is not None:
            if global_node_id is None:
                return self.node_encoder(x, node_type)
            return self.node_encoder(x, node_type, global_node_id)
        if global_node_id is None:
            return self.identity_embedding.weight  # type: ignore[union-attr]
        return self.identity_embedding.weight[global_node_id]  # type: ignore[union-attr]

    def geometry_inputs(self, h0: torch.Tensor, edge_index: torch.Tensor,
                        edge_type: torch.Tensor) -> torch.Tensor:
        """The input of the map MLPs: `h0` itself (spec 1.25 Z1, the geometry is a function of the
        incidence alone; the relational context `c` is removed). Kept as a method so that analysis
        and known-answer code reads the geometry the way the model does."""
        return h0

    def path_inputs(self, h0: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """The per-node input of g (spec v1.4 R12): each node's own h0."""
        return h0

    def _path_latent(self, h0: torch.Tensor, group: FaceGroup) -> torch.Tensor | None:
        """`[P, hidden]` path latent of g, or None (`path_weight_input: uniform`)."""
        if self.path_encoder is None:
            return None
        encoder = self.path_encoder
        return checkpoint_if_grad(
            lambda h_in, rel_in: encoder(h_in, group.face_node_index,
                                         self._hop_tokens(group, rel_in)),
            h0,
            self.relation_embedding.weight,
            enabled=self.checkpointing,
        )

    def _hop_tokens(self, group: FaceGroup, relation_weight: torch.Tensor) -> torch.Tensor:
        """`[1, L d_r]` relation token of a one-schema group (schema and DFS builders), or, for a
        WISP group, `[P, L (d_r + 3)]` per-cell action embeddings: the relation embedding and a
        one-hot of the hop direction (-1 / 0 for a self-inverse relation / +1)."""
        if group.signature_index is None:
            return relation_weight[group.relation_index[0]].reshape(1, -1)
        if not self.path_actions:
            raise ValueError("a WISP face group needs ModelDims.path_actions (build the model "
                             "with data.path_source: wisp).")
        relations = group.relation_index
        shared = torch.zeros(relation_weight.size(0), dtype=torch.bool, device=relations.device)
        if self.self_inverse_relations:
            shared[list(self.self_inverse_relations)] = True
        direction = torch.where(shared[relations], torch.zeros_like(relations),
                                group.face_edge_sign.sign().long())
        onehot = nn.functional.one_hot(direction + 1, 3).to(relation_weight.dtype)
        tokens = torch.cat([relation_weight[relations], onehot], dim=-1)
        return tokens.reshape(relations.size(0), -1)

    def _path_weights(self, latent: torch.Tensor | None, group: FaceGroup,
                      like: torch.Tensor) -> torch.Tensor:
        """`w_f = exp(g(latent_f))`; under `path_weight_input: uniform`, `exp(b)`."""
        if self.path_gate is None:
            return torch.exp(self.path_log_weight).expand(group.num_faces)  # type: ignore[arg-type]
        logits = self.path_gate(latent).squeeze(-1)
        if self.anchored and not bool(torch.isfinite(logits).all()):  # spec v1.16 K5, before exp
            bad = logits[~torch.isfinite(logits)][0]
            raise NonFiniteLogWeightError(float(bad), self._current_epoch)
        return torch.exp(logits)

    # ------------------------------------------------------------------ encoder
    def encode(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        num_nodes: int,
        epoch: int = -1,
        global_node_id: torch.Tensor | None = None,
        scope_index: ScopeIndex | None = None,
    ) -> EncodedState:
        """Shared encoder for all three tasks (a graph batch is one disjoint union). `epoch` sets
        the anchored solver's tolerance (fixed schedule; -1 = the final tolerance);
        `global_node_id` maps a batch of scopes to the whole graph's node ids (Part MB)."""
        h0 = self.encode_nodes(x, node_type, global_node_id)
        if self.h0_only:
            return EncodedState(h0=h0, x_star=h0, representation=h0, operator=None)
        return self._encode_anchored(h0, edge_index, edge_type, face_groups, num_nodes, epoch,
                                     scope_index)

    # ------------------------------------------------------------------ v3-B anchored solve
    def _cell_weights(self, h0: torch.Tensor, geometry: torch.Tensor,
                      groups: tuple[FaceGroup, ...]
                      ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
        """Per group: `beta_f = w_f m_f`, `w_f = exp(g(xi_f))` and g's latent (spec §3.2c)."""
        path_input = self.path_inputs(h0, geometry)
        betas, weights, latents = [], [], []
        for group in groups:
            latent = self._path_latent(path_input, group)
            w = self._path_weights(latent, group, h0)
            beta = w * group.path_weight.to(w.dtype) if group.path_weight is not None else w
            betas.append(beta)
            weights.append(w)
            latents.append(latent)
        return betas, weights, latents

    def _instance_features(self, h0: torch.Tensor, maps: EdgeMapPair, edge_index: torch.Tensor,
                           groups: tuple[FaceGroup, ...], betas: list[torch.Tensor],
                           num_nodes: int) -> torch.Tensor:
        """Arm `instance_features` (C14, MAGNN-type): `sum_f beta_f mean_{u in f} T_{u,f} h0_u` at
        both endpoints of every cell, each in its own frame (`T_{v_0} = I`; `T_{v_L}^T` for the
        far end), concatenated to the head input; cells enter the solve with beta = 0."""
        feature = h0.new_zeros((num_nodes, h0.size(-1)))
        for group, beta in zip(groups, betas):
            transport = PathTransportOperator(
                maps.src, maps.tgt, group.face_node_index, group.face_edge_index, edge_index,
                shared_scale=True, face_edge_sign=group.face_edge_sign)
            mean = transport.transport_sum(h0) / float(group.length + 1)
            feature.index_add_(0, group.face_node_index[:, 0], beta.unsqueeze(-1) * mean)
            feature.index_add_(0, group.face_node_index[:, -1],
                               beta.unsqueeze(-1) * transport.apply_holonomy_T(mean))
        return feature

    def _encode_anchored(self, h0: torch.Tensor, edge_index: torch.Tensor,
                         edge_type: torch.Tensor, face_groups: tuple[FaceGroup, ...],
                         num_nodes: int, epoch: int,
                         scope_index: ScopeIndex | None = None) -> EncodedState:
        """`Y* = argmin 1/2||x - h0||^2 + tau/2 Y^T L_hat Y` on the three-level energy (R13):
        geometry once from `h0`, cell weights from g, then `AnchoredSolve` on the eliminated
        operator (fact states only on `E_shared`; `hub_A`: none). Arms: see `model.arm`."""
        tau_stop, range_stop = self.stop_limits
        self._current_epoch = epoch  # for the stop conditions raised inside g (K5)
        tau_value = float(self.tau().detach())  # type: ignore[union-attr]
        if tau_value > tau_stop:  # spec §6 stop condition (J10)
            raise TauStopError(tau_value, tau_stop, epoch)
        geometry = self.geometry_inputs(h0, edge_index, edge_type)
        type_only = self._type_only()
        sheaf = self.node_to_edge(
            h=geometry, edge_index=edge_index, edge_type=edge_type,
            relation_embeddings=self.relation_embedding.weight, type_only=type_only,
        )
        maps, factors = sheaf.maps, sheaf.factors
        groups = tuple(group for group in face_groups if group.num_faces > 0)
        arm = self.arm
        cells_in_solve = arm not in ("closure_only", "no_path_cells", "instance_features")
        betas, weights, latents = self._cell_weights(h0, geometry, groups)
        if weights:
            if self.training:  # the adaptive proposal's input for the next epoch (Part H3)
                self.last_path_weights = tuple(w.detach() for w in weights)
            with torch.no_grad():
                log_w = torch.cat(weights).log()
                log_w_range = float(log_w.max() - log_w.min())
                self.last_path_stats = {"path_log_w_max": float(log_w.max()),
                                        "path_log_w_mean": float(log_w.mean()),
                                        "path_log_w_range": log_w_range}
            if log_w_range > range_stop:  # spec §6 stop condition (J10)
                raise LogWeightRangeStopError(log_w_range, range_stop, epoch)
        free_params: list[torch.Tensor] = []
        if arm == "free_cell_maps" and self.free_cell_maps is not None:
            for group, latent in zip(groups, latents):
                # one free map per fact-cell incidence, indexed by the hop's near node and position
                # (its latent is g's; the v2 map encoder never existed under anchored)
                free_params.extend(self.free_cell_maps.factors(
                    geometry, group.face_node_index[:, :-1], latent))
        num_groups = len(groups)
        device = h0.device

        identity_maps = self.identity_maps
        d_stalk = self.d
        node_to_edge = self.node_to_edge
        free_cell_maps = self.free_cell_maps

        def build(scale: torch.Tensor, *factors_and_rest: torch.Tensor) -> EliminatedOperator:
            if identity_maps:  # arm maps = I (S2/S0): only the per-edge scale (a_e = c_e^2)
                rest = factors_and_rest
                pair = EdgeMapPair(src=IdentityMapOperator(scale, d_stalk),
                                   tgt=IdentityMapOperator(scale, d_stalk))
            else:
                rotation, subspaces, frames = factors_and_rest[:3]
                rest = factors_and_rest[3:]
                pair = node_to_edge.operators(TD1Factors(scale, rotation, subspaces, frames),
                                              edge_type, type_only)
            group_betas, free_tensors = rest[:num_groups], rest[num_groups:]
            cell_groups, node_terms = [], []
            for i, (group, beta) in enumerate(zip(groups, group_betas)):
                if cells_in_solve:
                    free_maps = None
                    if arm == "free_cell_maps":
                        num_paths, length = group.face_edge_index.shape
                        free_maps = free_cell_maps.operator(
                            *free_tensors[3 * i:3 * i + 3], num_paths, length)
                    cell_groups.append(build_cell_group(pair, group, edge_index, beta, free_maps))
                elif arm == "closure_only":
                    transport = PathTransportOperator(
                        pair.src, pair.tgt, group.face_node_index, group.face_edge_index,
                        edge_index, shared_scale=True, face_edge_sign=group.face_edge_sign)
                    node_terms.append(HolonomyEdgeTerm(transport, beta))
            operator = SheafEnergyOperator(
                pair, edge_index, num_nodes, cell_groups,
                node_cell=arm != "covering_only", fact_cell=arm != "hub_A",
                node_terms=node_terms,
            )
            kept = (torch.zeros(0, dtype=torch.long, device=device) if arm == "hub_A" else None)
            return EliminatedOperator(operator, kept)

        if identity_maps:
            params = (maps.src.scale, *betas, *free_params)
        else:
            params = (factors.scale, factors.rotation, factors.subspaces, factors.frames, *betas,
                      *free_params)
        warmup_epochs, warmup_tolerance, final_tolerance, max_iterations = self.solver_settings
        tolerance = solver_tolerance(epoch, warmup_epochs, warmup_tolerance, final_tolerance)
        with torch.no_grad():  # layout of the solved system (for the warm start and the log)
            complex_ = IncidenceComplex.from_face_groups(
                edge_index, num_nodes, groups if cells_in_solve else ())
            num_states = 0 if arm == "hub_A" else complex_.num_shared_facts
            rows = num_nodes + num_states + complex_.num_cells
        # Part MB (MB3): in a batch of scopes every row of the solved system carries its scope
        # (per-scope inner products, stopping and T_max); None in full batch.
        scope_rows = None
        num_scopes = 1
        if scope_index is not None:
            scope_rows = solved_system_scope_rows(scope_index, complex_, face_groups,
                                                  num_states > 0, cells_in_solve)
            num_scopes = scope_index.num_scopes
        # Evaluation warm start (D-B7 otherwise): full batch only (MB6a), and only when the
        # evaluation draw is the same one as the stored solution's, not only the same shape.
        draw_key = _draw_key(edge_index, groups)
        y0 = None
        if (not self.training and scope_index is None and self._warm_start is not None
                and self._warm_start.shape == (rows, h0.size(-1))
                and self._warm_start.device == device
                and _same_draw(self._warm_start_key, draw_key)):
            y0 = self._warm_start  # the evaluation draw is fixed: warm start from the last epoch
        stats = SolveStats()
        try:
            y_star = anchored_solve(build, h0, self.theta_tau, params, tolerance, max_iterations,
                                    stats, y0=y0, scope_rows=scope_rows, num_scopes=num_scopes)
        except PCGConvergenceError as exc:  # T_max: name the epoch and the limit (K5), re-raise
            exc.condition, exc.value, exc.limit, exc.epoch = (
                "T_max", float(max_iterations), float(max_iterations), epoch)
            raise
        if not self.training and scope_index is None:
            self._warm_start = y_star.detach()
            self._warm_start_key = draw_key
        x_star = y_star[:num_nodes]
        parts = [x_star, h0] if self.ego_head else [x_star]
        if arm == "instance_features":
            parts.append(self._instance_features(h0, maps, edge_index, groups, betas, num_nodes))
        tau = self.tau().detach()  # type: ignore[union-attr]
        self._last_solve = stats
        self._last_solve_info = {
            "solver_tolerance": tolerance,
            "num_path_facts": float(complex_.num_path_facts),
            "num_shared_facts": float(complex_.num_shared_facts),
            "num_fact_states": float(num_states),
            "num_cells": float(complex_.num_cells),
            "eliminated_share": (1.0 - num_states / complex_.num_facts) if complex_.num_facts else 0.0,
        }
        state = AnchoredState(maps=maps, path_weights=tuple(weights), tau=tau, stats=stats,
                              layout_rows=(num_nodes, num_states, complex_.num_cells))
        return EncodedState(h0=h0, x_star=x_star, representation=torch.cat(parts, dim=-1),
                            operator=state)

    # ------------------------------------------------------------------ node classification
    def forward(self, data: PreparedH2SData, epoch: int = -1) -> H2SNNOutput:
        """Node classification. `epoch` sets the anchored solver's tolerance schedule."""
        if self.readout is None:
            raise RuntimeError(
                f"forward() is the node-classification head; this model was built for "
                f"'{self.task_type}' (use encode() with link_readout / graph_readout)."
            )
        encoded = self.encode(
            data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
            data.num_nodes, epoch=epoch, global_node_id=data.global_node_id,
            scope_index=data.scope_index,
        )
        logits = self.readout(encoded.representation[data.target_indices])
        operator = encoded.operator
        if operator is None:
            return H2SNNOutput(logits=logits, x_star=encoded.x_star, h0=encoded.h0,
                               path_gate_mean=None, path_gate_max=None, max_edge_scale=None)
        gate_mean = gate_max = None
        if operator.path_weights:
            all_weights = torch.cat(operator.path_weights).detach()
            gate_mean, gate_max = all_weights.mean(), all_weights.max()
        max_scale = torch.maximum(operator.maps.src.scale.max(), operator.maps.tgt.scale.max())
        return H2SNNOutput(
            logits=logits,
            x_star=encoded.x_star,
            h0=encoded.h0,
            path_gate_mean=gate_mean,
            path_gate_max=gate_max,
            max_edge_scale=max_scale.detach(),
        )


def solved_system_scope_rows(scope_index: ScopeIndex, complex_: "IncidenceComplex",
                             face_groups: tuple[FaceGroup, ...], fact_states: bool,
                             cells_in_solve: bool) -> torch.Tensor:
    """`[rows]` scope of every row of the anchored system (Part MB, MB3): nodes, then the kept
    fact states (facts on >= 2 cells, ascending id, `EliminatedOperator`), then the cells of the
    non-empty groups in group order (`IncidenceComplex.cell_offsets`)."""
    parts = [scope_index.node]
    if fact_states:
        kept = (complex_.cells_per_fact >= 2).nonzero(as_tuple=True)[0]
        parts.append(scope_index.fact[kept])
    if cells_in_solve:
        parts.extend(cells for group, cells in zip(face_groups, scope_index.cell)
                     if group.num_faces > 0)
    return torch.cat(parts)


def _draw_key(edge_index: torch.Tensor, groups: tuple[FaceGroup, ...]) -> tuple:
    """The tensors that identify an evaluation draw (for the warm start)."""
    return (edge_index, *[(g.face_node_index, g.face_edge_index, g.face_edge_sign, g.path_weight)
                          for g in groups])


def _same_tensor(a, b) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    return a.shape == b.shape and a.device == b.device and bool(torch.equal(a, b))


def _same_draw(stored: tuple, current: tuple) -> bool:
    if len(stored) != len(current) or not stored:
        return False
    if not _same_tensor(stored[0], current[0]):
        return False
    return all(all(_same_tensor(x, y) for x, y in zip(a, b))
               for a, b in zip(stored[1:], current[1:]))


def build_model(config: TrainConfig, data: PreparedH2SData) -> H2SNN:
    if config.data.path_source not in ("schemas", "wisp"):
        raise ValueError(
            f"data.path_source {config.data.path_source!r} is unresolved; build the config with "
            "TrainConfig.from_dict or DataConfig.resolved() (a directly built DataConfig resolves "
            "nothing).")
    dims = ModelDims.from_prepared_data(data)
    if config.data.path_source == "wisp":
        # Every length of the support gets a projection, not only those in this draw: a later
        # training draw may contain a length the evaluation draw lacks.
        dims = dims._replace(face_lengths=tuple(range(2, config.data.wisp_max_length + 1)),
                             path_actions=True)
    elif is_literature_row(config.data):
        # Part O (O3): the fixed-literature-schema row reads WISP-format cells (per-cell actions);
        # one projection of g per length of the listed meta-paths (every training draw has them).
        dims = dims._replace(face_lengths=literature_lengths(config.data.dataset_name),
                             path_actions=True)
    return H2SNN(config=config, dims=dims)


def build_link_prediction_model(config: TrainConfig, data: PreparedLinkPredData) -> H2SNN:
    dims = ModelDims.from_link_pred_data(data)
    if config.data.path_source == "wisp":
        # spec v1.15 J1: every length of the support (a training draw may hold a length the
        # evaluation draw lacks), as for node classification.
        dims = dims._replace(face_lengths=tuple(range(2, config.data.wisp_max_length + 1)),
                             path_actions=True)
    return H2SNN(config=config, dims=dims)


def build_graph_classification_model(config: TrainConfig, dims: ModelDims) -> H2SNN:
    return H2SNN(config=config, dims=dims)
