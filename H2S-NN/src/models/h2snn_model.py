"""H2S-NN: node diffusion with the vertex-2-cell sheaf Laplacian (plan §4, decision D6).

    h0 = TypeEncoder(x)
    repeat K steps:
        x <- x - eta D^{-1/2} (Delta_0 + gamma L_P) D^{-1/2} x       # then layer dropout
    representation = [x_K, h0]

`Delta_0` runs over the observed edges with the D1 map operators (one conformal scale per
edge, D1a); `L_P` couples every path 2-cell to every node on it through ordered parallel
transport (`src/models/vertex_path_laplacian.py`). `eta = sigmoid(theta_eta)`,
`gamma = softplus(theta_gamma)`, `w_f = sigmoid(g(path_latent_f))`.

One encoder, three task heads (plan §4.8), selected by `data.task_type`:
- node classification: `MLP([x_K, h0])` on the target nodes;
- link prediction: the `dot` / `distmult` / `concat_mlp` decoders on `[x_K, h0]`;
- graph classification: sum of `[x_K, h0]` per graph -> BatchNorm -> MLP.
Ablation switches (plan §4.6) come from `config.ablation`.

v3-B (spec v1.11 R13; `model.propagation: anchored`): the propagation above is replaced by one
anchored quadratic solve on the three-level sheaf energy (`src/models/sheaf_energy.py`),

    (P^T P + tau L_hat) Y* = P^T h0,   Y = (x, s, z),   tau = exp(theta_tau),

solved by PCG with an implicit backward (`src/models/anchored_solve.py`); the arms of spec §6 are
configurations of this one path (`model.arm`). The legacy propagation stays reachable under
`model.propagation: chebyshev` (the default) and is unchanged.
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple

import torch
from torch import nn

from src.configs.config import TrainConfig
from src.data.link_pred_structures import PreparedLinkPredData
from src.data.structures import FaceGroup, PreparedH2SData, face_lengths_of
from src.models.graph_readout import GraphReadout
from src.models.link_readout import LinkPredictionReadout
from src.models.pair_attention import (
    AttentionOperator,
    PairAttention,
    build_attention_operator,
)
from src.models.map_operators import (
    EdgeMapPair,
    FactorizedMapOperator,
    IdentityMapOperator,
    PathTransportOperator,
    checkpoint_if_grad,
    normalize_reflectors,
)
from src.models.filter_bank import ChebyshevFilterBank
from src.models.anchored_solve import SolveStats, anchored_solve, solver_tolerance
from src.models.sheaf_energy import (
    EliminatedOperator,
    IncidenceComplex,
    SheafEnergyOperator,
    build_cell_group,
)
from src.models.node_encoder import TypeSpecificNodeEncoder
from src.models.relational_context import RelationalContext
from src.models.orthogonal import Orthogonal
from src.models.sheaf_maps import (
    REFLECTOR_NORM_EPS,
    DenseConformalNodeToEdgeSheaf,
    FactorizedNodeToEdgeSheaf,
    gathered_linear,
)
from src.models.vertex_path_laplacian import (
    EdgePairTerm,
    FreeIncidenceOperator,
    HolonomyEdgeTerm,
    PathCellTerm,
    PathTerm,
    VertexPathSheafLaplacian,
)


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


class DiffusionOperator(NamedTuple):
    """Everything one forward pass diffuses with; `encode` and the tests share it."""

    laplacian: VertexPathSheafLaplacian
    eta: torch.Tensor | None  # None under encoder: filter_bank (eta exists only under first_order)
    gamma: torch.Tensor | None
    path_weights: tuple[torch.Tensor, ...]
    maps: EdgeMapPair
    # D10 listen / broadcast gates g, b in (0, 1), shape [N]; None = 1 (symmetric, or the
    # absent half under `listen_only`). The update is x <- x - eta G Delta B x.
    listen: torch.Tensor | None = None
    broadcast: torch.Tensor | None = None


class AnchoredState(NamedTuple):
    """What one anchored forward solved with (v3-B); `forward` reads the same four attributes as
    from `DiffusionOperator` (eta and gamma do not exist: None)."""

    maps: EdgeMapPair
    path_weights: tuple[torch.Tensor, ...]  # w_f per face group (before m_f)
    tau: torch.Tensor                        # detached
    stats: SolveStats
    layout_rows: tuple[int, int, int]        # (N, |fact states|, F) of the solved system
    eta: torch.Tensor | None = None
    gamma: torch.Tensor | None = None


class EncodedState(NamedTuple):
    """Output of the shared encoder, before any task head."""

    h0: torch.Tensor
    x_star: torch.Tensor  # x_K (h0 under ablation.h0_only_readout)
    representation: torch.Tensor  # [x_K, h0] (h0 under ablation.h0_only_readout)
    operator: "DiffusionOperator | AnchoredState | None"  # None under ablation.h0_only_readout


class H2SNNOutput(NamedTuple):
    """Node-classification output."""

    logits: torch.Tensor
    x_star: torch.Tensor
    h0: torch.Tensor
    # Detached diagnostics for logging (no effect on optimization).
    eta: torch.Tensor | None
    gamma: torch.Tensor | None
    path_gate_mean: torch.Tensor | None
    path_gate_max: torch.Tensor | None
    max_edge_scale: torch.Tensor | None


def _hop_block(hop_tokens: torch.Tensor, num_paths: int, reference: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """`(tokens, index)` for `gathered_linear`: one shared row (schema groups, index 0) or one
    row per cell (WISP groups)."""
    if hop_tokens.size(0) == 1:
        return hop_tokens, reference.new_zeros(num_paths)
    return hop_tokens, torch.arange(num_paths, device=reference.device)


class PathEncoder(nn.Module):
    """Positional `concat` encoder over the ordered `h0` of a path and its hop tokens (the schema's
    relation embeddings, or per-cell action embeddings under WISP): one `Linear + GELU` per path
    length; its output feeds the path gate."""

    def __init__(self, d: int, d_r: int, hidden_dim: int, face_lengths: tuple[int, ...],
                 factored_inputs: bool = False) -> None:
        """`d`: width of the per-node input (`h0`; `[h0, c]` only under the spec v1.4 appendix
        switch `ablation.path_weight_context`).
        `d_r`: width of one hop token (d_r, or d_r + 3 with per-cell actions).
        `factored_inputs`: project per node, then gather (`gathered_linear`; D12, R7)."""
        super().__init__()
        self.input_proj = nn.ModuleDict(
            {
                str(length): nn.Linear((length + 1) * d + length * d_r, hidden_dim)
                for length in face_lengths
            }
        )
        self.activation = nn.GELU()
        self.latent_dim = hidden_dim
        self.factored_inputs = factored_inputs

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
        if self.factored_inputs:
            blocks = [(h0, face_node_index[:, j]) for j in range(length + 1)]
            blocks.append(_hop_block(hop_tokens, num_paths, face_node_index))
            return self.activation(gathered_linear(self.input_proj[key], blocks))
        inputs = torch.cat(
            [h0[face_node_index].reshape(num_paths, -1), hop_tokens.expand(num_paths, -1)],
            dim=-1,
        )
        return self.activation(self.input_proj[key](inputs))


class EndpointEncoder(nn.Module):
    """Latent of a path from its endpoints and hop tokens only, `[h_{v_0}, h_{v_L}, tokens]`
    (ablation `path_mode: endpoint_edges`, and the `endpoints` arm of the C2 weight ladder: no
    intermediate node enters)."""

    def __init__(self, d: int, d_r: int, hidden_dim: int, face_lengths: tuple[int, ...],
                 factored_inputs: bool = False) -> None:
        super().__init__()
        self.input_proj = nn.ModuleDict(
            {str(length): nn.Linear(2 * d + length * d_r, hidden_dim) for length in face_lengths}
        )
        self.activation = nn.GELU()
        self.latent_dim = hidden_dim
        self.factored_inputs = factored_inputs

    def forward(
        self,
        h0: torch.Tensor,
        face_node_index: torch.Tensor,
        hop_tokens: torch.Tensor,
    ) -> torch.Tensor:
        length = int(face_node_index.size(1)) - 1
        num_paths = face_node_index.size(0)
        endpoints = face_node_index[:, [0, length]]
        if self.factored_inputs:
            blocks = [(h0, endpoints[:, 0]), (h0, endpoints[:, 1]),
                      _hop_block(hop_tokens, num_paths, face_node_index)]
            return self.activation(gathered_linear(self.input_proj[str(length)], blocks))
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


class EndpointEdgeMaps(nn.Module):
    """Factorized maps of the endpoint edge `(v_0 -> v_L)` from the endpoint latent:
    `F_side = c R_side H_k(u_side)`, one rotation per side, `c` per `map_scale`."""

    def __init__(self, d: int, latent_dim: int, householder_k: int, map_scale: str,
                 scale_eps: float, orthogonal_map: str) -> None:
        super().__init__()
        self.d = d
        self.householder_k = householder_k
        self.map_scale = map_scale
        self.scale_eps = scale_eps
        self.orthogonal = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        self.rotation_params = _rotation_parameters(2, self.orthogonal, d)
        self.reflector_head = nn.Linear(latent_dim, 2 * householder_k * d)
        self.scale_head = nn.Linear(latent_dim, 1 if map_scale == "per_edge" else 2)

    def forward(self, latent: torch.Tensor) -> EdgeMapPair:
        num_paths = latent.size(0)
        reflectors = normalize_reflectors(
            self.reflector_head(latent).view(num_paths, 2, self.householder_k, self.d),
            REFLECTOR_NORM_EPS,
        )
        scales = nn.functional.softplus(self.scale_head(latent)) + self.scale_eps
        if self.map_scale == "per_edge":
            side_scales = (scales[:, 0], scales[:, 0])  # one c_e, shared (D1a)
        else:
            side_scales = (scales[:, 0], scales[:, 1])
        rotations = self.orthogonal(self.rotation_params)
        group_ids = torch.zeros(num_paths, dtype=torch.long, device=latent.device)
        sides = [
            FactorizedMapOperator(
                scale=side_scales[side],
                reflectors=reflectors[:, side],
                rotations=rotations[side:side + 1],
                group_ids=group_ids,
            )
            for side in range(2)
        ]
        return EdgeMapPair(src=sides[0], tgt=sides[1])


class FreeIncidenceMaps(nn.Module):
    """Ablation `path_maps: free`: one orthogonal map `R_{position} H_k(u_{v,f})` per
    incidence, `u` from `[h_v, path_latent_f, onehot(position)]` (same factorized family as
    the node->edge maps, but not forced by functoriality)."""

    def __init__(self, d: int, latent_dim: int, hidden_dim: int, householder_k: int,
                 max_positions: int, orthogonal_map: str, node_dim: int | None = None,
                 factored_inputs: bool = False) -> None:
        super().__init__()
        self.d = d
        self.factored_inputs = factored_inputs
        node_dim = d if node_dim is None else node_dim
        self.householder_k = householder_k
        self.max_positions = max_positions
        self.orthogonal = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        self.rotation_params = _rotation_parameters(max_positions, self.orthogonal, d)
        self.reflector_mlp = nn.Sequential(
            nn.Linear(node_dim + latent_dim + max_positions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, householder_k * d),
        )

    def forward(self, h0: torch.Tensor, face_node_index: torch.Tensor,
                latent: torch.Tensor) -> FactorizedMapOperator:
        num_paths, size = face_node_index.shape
        positions = torch.arange(size, device=h0.device)
        onehot = nn.functional.one_hot(positions, self.max_positions).to(h0.dtype)
        if self.factored_inputs:
            blocks = [(h0, face_node_index.reshape(-1)),
                      (latent, torch.arange(num_paths, device=h0.device).repeat_interleave(size)),
                      (onehot, positions.repeat(num_paths))]
            raw = self.reflector_mlp[1:](gathered_linear(self.reflector_mlp[0], blocks))
        else:
            inputs = torch.cat(
                [
                    h0[face_node_index],
                    latent.unsqueeze(1).expand(num_paths, size, -1),
                    onehot.unsqueeze(0).expand(num_paths, size, -1),
                ],
                dim=-1,
            ).reshape(num_paths * size, -1)
            raw = self.reflector_mlp(inputs)
        reflectors = normalize_reflectors(
            raw.view(-1, self.householder_k, self.d), REFLECTOR_NORM_EPS
        )
        return FactorizedMapOperator(
            scale=h0.new_ones(num_paths * size),  # unused: free maps are orthogonal
            reflectors=reflectors,
            rotations=self.orthogonal(self.rotation_params),
            group_ids=positions.repeat(num_paths),
        )


def _gate_head(d: int, gate_init: float) -> nn.Linear:
    """`g = sigmoid(a^T h0 + beta)` as a linear head: bias = logit(gate_init) so every gate
    starts at `coupling_gate_init`, weights N(0, 0.01^2) (an initialisation choice, like the
    frame init in sheaf_maps.py, not a config key) so training starts near the symmetric model."""
    head = nn.Linear(d, 1)
    nn.init.normal_(head.weight, mean=0.0, std=0.01)
    nn.init.constant_(head.bias, _logit(gate_init))
    return head


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _softplus_inverse(y: float) -> float:
    return y + math.log(-math.expm1(-y))


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
        self.num_steps = model_config.K
        self.checkpointing = model_config.checkpointing
        self.map_scale = ablation.map_scale
        self.h0_only = ablation.h0_only_readout
        # D12 G0b sheaf-free diagnostic: node_to_edge is still built (same state dict) but not called.
        self.identity_maps = ablation.identity_maps
        # v3-B: the anchored solve replaces K, eta, gamma, the filter bank and the gates.
        self.propagation = model_config.propagation
        self.arm = model_config.arm
        anchored = self.propagation == "anchored" and not self.h0_only
        self.anchored = anchored

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
        # D12 G1: geometry inputs [h0, c] (parameter-free context; false = the D11 model).
        self.relational_context: RelationalContext | None = None
        geometry_dim = d
        if model_config.geometry_context:
            self.relational_context = RelationalContext(dims.num_relations,
                                                        self.shared_role_relations)
            geometry_dim = d + self.relational_context.output_dim(d)
        self.geometry_dim = geometry_dim
        self.node_to_edge: FactorizedNodeToEdgeSheaf | DenseConformalNodeToEdgeSheaf
        if model_config.map_family == "factorized":
            self.node_to_edge = FactorizedNodeToEdgeSheaf(
                d=d,
                d_r=model_config.d_r,
                num_relations=dims.num_relations,
                hidden_dim=model_config.edge_mlp_hidden_dim,
                householder_k=model_config.householder_k,
                scale_eps=model_config.map_scale_eps,
                orthogonal_map=model_config.orthogonal_map,
                map_scale=self.map_scale,
                checkpoint=model_config.checkpointing,
                shared_role_relations=self.shared_role_relations,
                input_dim=geometry_dim,
                factored_inputs=model_config.geometry_context,
            )
        else:
            self.node_to_edge = DenseConformalNodeToEdgeSheaf(
                d=d,
                d_r=model_config.d_r,
                num_relations=dims.num_relations,
                hidden_dim=model_config.edge_mlp_hidden_dim,
                scale_eps=model_config.map_scale_eps,
                orthogonal_map=model_config.orthogonal_map,
                map_scale=self.map_scale,
                memory_guard_fraction=model_config.node_edge_memory_guard_fraction,
                shared_role_relations=self.shared_role_relations,
                input_dim=geometry_dim,
            )
        # eta exists only under encoder: first_order (D12); the filter bank absorbs the step size.
        self.encoder = model_config.encoder
        self.theta_eta: nn.Parameter | None = (
            nn.Parameter(torch.tensor(_logit(model_config.eta_init)))
            if model_config.encoder == "first_order" and not anchored else None
        )
        self.use_path_cells = not ablation.no_path_cells
        self.path_mode = ablation.path_mode
        self.free_path_maps = ablation.free_path_maps
        self.theta_gamma: nn.Parameter | None = None
        self.path_encoder: PathEncoder | EndpointEncoder | SchemaEncoder | None = None
        self.path_gate: nn.Linear | None = None
        self.path_log_weight: nn.Parameter | None = None
        self.free_maps: FreeIncidenceMaps | None = None
        self.endpoint_maps: EndpointEdgeMaps | None = None
        # Part H: w_f = exp(g) without gamma ("exp"), or gamma * sigmoid(g) ("sigmoid_gamma");
        # the C2 ladder restricts the inputs of g on identical cells.
        self.path_weight = model_config.path_weight
        self.path_weight_input = ablation.path_weight_input
        # Spec v1.4 (R12): g, i.e. the path encoder and its gate, reads each instance node's h0;
        # [h0, c] only under the appendix switch. Maps and gates keep geometry_dim.
        self.path_weight_context = ablation.path_weight_context
        # Spec v1.6 E10: pair attention has no g; its path latents are scores and read [h0, c].
        path_input_dim = (geometry_dim if self.path_weight_context
                          or ablation.coupling == "pair_attention" else d)
        self.path_input_dim = path_input_dim
        self.path_actions = dims.path_actions
        hop_dim = model_config.d_r + (3 if dims.path_actions else 0)
        if self.use_path_cells:
            if self.path_weight == "sigmoid_gamma":
                self.theta_gamma = nn.Parameter(
                    torch.tensor(_softplus_inverse(model_config.gamma_init))
                )
            hidden = model_config.face_mlp_hidden_dim
            if self.path_mode == "endpoint_edges":
                # No intermediate node enters: neither the maps nor the gate see them.
                self.path_encoder = EndpointEncoder(path_input_dim, hop_dim, hidden,
                                                    dims.face_lengths,
                                                    factored_inputs=model_config.geometry_context)
                self.endpoint_maps = EndpointEdgeMaps(
                    d=d, latent_dim=hidden, householder_k=model_config.householder_k,
                    map_scale=self.map_scale, scale_eps=model_config.map_scale_eps,
                    orthogonal_map=model_config.orthogonal_map,
                )
            elif self.free_path_maps or self.path_weight_input == "full":
                self.path_encoder = PathEncoder(path_input_dim, hop_dim, hidden,
                                                dims.face_lengths,
                                                factored_inputs=model_config.geometry_context)
            elif self.path_weight_input == "endpoints":
                hidden = endpoint_hidden_width(hidden, path_input_dim, hop_dim, dims.face_lengths)
                self.path_encoder = EndpointEncoder(path_input_dim, hop_dim, hidden,
                                                    dims.face_lengths,
                                                    factored_inputs=model_config.geometry_context)
            elif self.path_weight_input == "schema":
                self.path_encoder = SchemaEncoder(hop_dim, hidden, dims.face_lengths)
            if self.free_path_maps:
                self.free_maps = FreeIncidenceMaps(
                    d=d, latent_dim=hidden, hidden_dim=model_config.edge_mlp_hidden_dim,
                    householder_k=model_config.householder_k,
                    max_positions=max(dims.face_lengths, default=1) + 1,
                    orthogonal_map=model_config.orthogonal_map,
                    node_dim=geometry_dim,
                    factored_inputs=model_config.geometry_context,
                )
            if self.path_weight_input != "uniform":
                # Gate g, default weight init so the path encoder gets gradients from step 0.
                # sigmoid_gamma: zero bias, w_f = sigmoid(g) starts near 0.5 with a small spread.
                # exp: bias b = log(path_weight_init), w_f = exp(g) starts near path_weight_init.
                self.path_gate = nn.Linear(hidden, 1)
                if self.path_weight == "exp":
                    nn.init.constant_(self.path_gate.bias, math.log(model_config.path_weight_init))
                else:
                    nn.init.zeros_(self.path_gate.bias)
            elif self.path_weight == "exp":
                # uniform under exp: w_f = exp(b) for every cell (b alone; no g input).
                self.path_log_weight = nn.Parameter(
                    torch.tensor(math.log(model_config.path_weight_init)))
        self.layer_dropout = nn.Dropout(model_config.layer_dropout)
        self.coupling = ablation.coupling
        self.listen_gate: nn.Linear | None = None
        self.broadcast_gate: nn.Linear | None = None
        self.last_coupling_stats: dict[str, float] = {}
        self.last_path_stats: dict[str, float] = {}
        self.last_path_weights: tuple[torch.Tensor, ...] = ()
        # Gate heads are created last, so the other parameters initialise exactly as before
        # (`symmetric` creates none: the pre-D10 model, bit for bit). No heads without
        # diffusion (`h0_only_readout`).
        if not self.h0_only and self.coupling in ("listen_only", "cooperative"):
            self.listen_gate = _gate_head(geometry_dim, model_config.coupling_gate_init)
            if self.coupling == "cooperative":
                self.broadcast_gate = _gate_head(geometry_dim, model_config.coupling_gate_init)

        # Task head on the node representation [x_K, h0] (h0 under h0_only_readout); the v3-B arm
        # `instance_features` adds the anchor's cell feature sum_f beta_f mean_v T_{v,f} h_v (C14).
        representation_dim = d if self.h0_only else 2 * d
        if anchored and self.arm == "instance_features":
            representation_dim = 3 * d
        self.readout: nn.Sequential | None = None
        self.link_readout: LinkPredictionReadout | None = None
        self.graph_readout: GraphReadout | None = None
        if self.task_type == _LINK_PREDICTION:
            self.link_readout = LinkPredictionReadout(
                d=d,
                input_dim=representation_dim,
                decoder_type=config.data.decoder_type,
                hidden_dim=model_config.classifier_hidden_dim,
                dropout=model_config.classifier_dropout,
                num_relations=dims.num_target_relations,
            )
        elif self.task_type == _GRAPH_CLASSIFICATION:
            self.graph_readout = GraphReadout(
                input_dim=representation_dim,
                hidden_dim=model_config.classifier_hidden_dim,
                num_classes=dims.num_classes,
                dropout=model_config.classifier_dropout,
            )
        else:
            self.readout = nn.Sequential(
                nn.Linear(representation_dim, model_config.classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(model_config.classifier_dropout),
                nn.Linear(model_config.classifier_hidden_dim, dims.num_classes),
            )
        # A4 test variant (plan §12.3): created last, so every other parameter initialises as in
        # the gated model; gate heads are not instantiated for it.
        self.pair_attention: PairAttention | None = None
        if self.coupling == "pair_attention" and not self.h0_only:
            self.pair_attention = PairAttention(
                d=d, d_r=model_config.d_r, path_hidden=model_config.face_mlp_hidden_dim,
                num_relations=dims.num_relations,
                max_positions=max(dims.face_lengths, default=1) + 1,
            )
        # D12 G2: the Chebyshev filter bank (spec v1.2 §3). Created last; its initialization is
        # deterministic (V_k = theta_k I, no RNG), so every other parameter is as before.
        self.filter_bank: ChebyshevFilterBank | None = None
        if model_config.encoder == "filter_bank" and not self.h0_only and not anchored:
            self.filter_bank = ChebyshevFilterBank(d, model_config.K, model_config.eta_init)
        # Spec v1.6 E10: the path-specific maps of endpoint_edges and free_path_maps read the
        # geometry [h0, c]; g reads h0. A second encoder (same class) exists only when the two
        # inputs differ (geometry_context on, path_weight_context off), so the context-off model
        # is unchanged; it is created last, so every other parameter initialises as before.
        self.map_encoder: PathEncoder | EndpointEncoder | None = None
        if (self.use_path_cells and (self.path_mode == "endpoint_edges" or self.free_path_maps)
                and path_input_dim != geometry_dim):
            encoder_class = EndpointEncoder if self.path_mode == "endpoint_edges" else PathEncoder
            self.map_encoder = encoder_class(geometry_dim, hop_dim,
                                             model_config.face_mlp_hidden_dim, dims.face_lengths,
                                             factored_inputs=model_config.geometry_context)
        # v3-B (created last: every legacy parameter initialises as before). tau = exp(theta_tau),
        # theta_tau = log(tau_init), own optimizer group (D-B4). The arm `free_cell_maps` has one
        # free orthogonal map per fact-cell incidence (hop positions 0..L_max-1; Thm 3 control).
        self.theta_tau: nn.Parameter | None = None
        self.free_cell_maps: FreeIncidenceMaps | None = None
        self.solver_settings = (model_config.solver_warmup_epochs,
                                model_config.solver_warmup_tolerance,
                                model_config.solver_tolerance,
                                model_config.solver_max_iterations)
        # Statistics of the last anchored solve: the PCG counts live in `_last_solve` (the backward
        # fills its half after `forward` returns), the layout numbers in `_last_solve_info`.
        self._last_solve: SolveStats | None = None
        self._last_solve_info: dict[str, float] = {}
        self._warm_start: torch.Tensor | None = None
        if anchored:
            self.theta_tau = nn.Parameter(torch.tensor(math.log(model_config.tau_init)))
            if self.arm == "free_cell_maps":
                latent_dim = (self.path_encoder.latent_dim if self.path_encoder is not None
                              else model_config.face_mlp_hidden_dim)
                self.free_cell_maps = FreeIncidenceMaps(
                    d=d, latent_dim=latent_dim, hidden_dim=model_config.edge_mlp_hidden_dim,
                    householder_k=model_config.householder_k,
                    max_positions=max(dims.face_lengths, default=1),
                    orthogonal_map=model_config.orthogonal_map,
                    node_dim=geometry_dim,
                    factored_inputs=model_config.geometry_context,
                )

    # ------------------------------------------------------------------ scalars
    def eta(self) -> torch.Tensor | None:
        """`sigmoid(theta_eta)` under `encoder: first_order`; None under `filter_bank`."""
        if self.theta_eta is None:
            return None
        return torch.sigmoid(self.theta_eta)

    def gamma(self) -> torch.Tensor | None:
        if self.theta_gamma is None:
            return None
        return nn.functional.softplus(self.theta_gamma)

    def tau(self) -> torch.Tensor | None:
        """`exp(theta_tau)` under `propagation: anchored`; None otherwise."""
        if self.theta_tau is None:
            return None
        return torch.exp(self.theta_tau)

    def diffusion_scalars(self) -> dict[str, float]:
        """Current eta and gamma (legacy) or tau and the last solve's PCG statistics (anchored),
        for per-epoch logging."""
        with torch.no_grad():
            scalars: dict[str, float] = {}
            eta = self.eta()
            if eta is not None:
                scalars["eta"] = float(eta)
            tau = self.tau()
            if tau is not None:
                scalars["tau"] = float(tau)
                scalars.update(self.last_solver_stats)
            if self.filter_bank is not None:  # sum_k ||W_k||_2 (Prop. 5 bound), float64 norms
                scalars["filter_norm_sum"] = self.filter_bank.diagnostic_norm_sum()
            gamma = self.gamma()
            if gamma is not None:
                scalars["gamma"] = float(gamma)
        scalars.update(self.last_coupling_stats)  # gates of the last training forward
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
        """`theta_eta`, `theta_gamma`, `theta_tau` (v3-B, D-B4) and the gate heads (weights and
        biases): the optimizer gives them their own group (plan §12.4). Sigmoid-saturated scalars
        move too little per full-batch step at the base rate."""
        scalars = [p for p in (self.theta_eta, self.theta_gamma, self.theta_tau) if p is not None]
        for head in (self.listen_gate, self.broadcast_gate):
            if head is not None:
                scalars.extend(head.parameters())
        if self.path_weight == "exp":
            # b, the bias of g (gamma's role under exp, spec v1.3 §3.2c), or the uniform arm's b.
            if self.path_gate is not None:
                scalars.append(self.path_gate.bias)
            if self.path_log_weight is not None:
                scalars.append(self.path_log_weight)
        if self.pair_attention is not None:
            # The strength-forming parameters of the A4 test get the same optimizer treatment as
            # the gate heads they replace (a test of A4 must not handicap one arm).
            scalars.extend(self.pair_attention.parameters())
        return scalars

    # ------------------------------------------------------------------ operator
    def encode_nodes(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        if self.node_encoder is not None:
            return self.node_encoder(x, node_type)
        return self.identity_embedding.weight  # type: ignore[union-attr]

    def geometry_inputs(self, h0: torch.Tensor, edge_index: torch.Tensor,
                        edge_type: torch.Tensor) -> torch.Tensor:
        """[h0, c] under geometry_context (D12 G1), else h0: the input of the map MLPs, the gate
        heads and the path encoder. Computed once per forward, before any propagation."""
        if self.relational_context is None:
            return h0
        return torch.cat([h0, self.relational_context(h0, edge_index, edge_type)], dim=-1)

    def path_inputs(self, h0: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """The per-node input of g (spec v1.4 R12): each node's own h0, or the v1.3 geometry
        [h0, c] under `ablation.path_weight_context` (appendix pair)."""
        return geometry if self.path_weight_context else h0

    def _map_latent(self, geometry: torch.Tensor, group: FaceGroup) -> torch.Tensor:
        """Latent of the path-specific maps (endpoint_edges, free_path_maps) from the geometry
        `[h0, c]` (spec v1.6 E10), through the map encoder."""
        encoder = self.map_encoder
        return checkpoint_if_grad(
            lambda h_in, rel_in: encoder(h_in, group.face_node_index,
                                         self._hop_tokens(group, rel_in)),
            geometry,
            self.relation_embedding.weight,
            enabled=self.checkpointing,
        )

    def _path_latent(self, h0: torch.Tensor, group: FaceGroup) -> torch.Tensor | None:
        """`[P, hidden]` path latent (endpoint-only under `endpoint_edges`), or None."""
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
        """`w_f = exp(g(latent_f))` (`path_weight: exp`) or `sigmoid(g(latent_f))`
        (`sigmoid_gamma`); under `path_weight_input: uniform`, `exp(b)` or 1."""
        if self.path_gate is None:
            if self.path_log_weight is not None:
                return torch.exp(self.path_log_weight).expand(group.num_faces)
            return like.new_ones(group.num_faces)
        logits = self.path_gate(latent).squeeze(-1)
        return torch.exp(logits) if self.path_weight == "exp" else torch.sigmoid(logits)

    def path_terms(
        self,
        h0: torch.Tensor,
        maps: EdgeMapPair,
        edge_index: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        *,
        path_input: torch.Tensor,
    ) -> tuple[tuple[PathTerm, ...], tuple[torch.Tensor, ...]]:
        """One coupling term per face group, per the ablation switches, and its `w_f`. `h0` is the
        per-node geometry input (`geometry_inputs`: `[h0, c]` under `geometry_context`; the free
        maps' node input); `path_input` is g's per-node input (`path_inputs`, spec v1.4 R12)."""
        terms: list[PathTerm] = []
        weights: list[torch.Tensor] = []
        shared = self.map_scale == "per_edge"
        for group in face_groups:
            if group.num_faces == 0:
                continue
            latent = self._path_latent(path_input, group)
            # one latent when g's input is the geometry (context off, or the appendix switch)
            map_latent = latent if self.map_encoder is None else self._map_latent(h0, group)
            group_weights = self._path_weights(latent, group, h0)
            gate_weights = group_weights
            if group.path_weight is not None:  # merged weight m_f, in both L_P and D (D8)
                group_weights = group_weights * group.path_weight.to(group_weights.dtype)
            nodes = group.face_node_index
            if self.path_mode == "endpoint_edges":
                endpoints = torch.stack([nodes[:, 0], nodes[:, -1]])
                term: PathTerm = EdgePairTerm(self.endpoint_maps(map_latent), endpoints,
                                              group_weights)
            elif self.free_path_maps:
                incidence = FreeIncidenceOperator(self.free_maps(h0, nodes, map_latent), nodes)
                term = PathCellTerm(incidence, group_weights)
            else:
                transport = PathTransportOperator(
                    maps.src, maps.tgt, nodes, group.face_edge_index, edge_index,
                    shared_scale=shared, face_edge_sign=group.face_edge_sign,
                )
                if self.path_mode == "holonomy_edges":
                    term = HolonomyEdgeTerm(transport, group_weights)
                else:
                    term = PathCellTerm(transport, group_weights)
            terms.append(term)
            weights.append(gate_weights)
        return tuple(terms), tuple(weights)

    def _build_attention_operator(
        self,
        h0: torch.Tensor,
        maps: EdgeMapPair,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        geometry: torch.Tensor | None = None,
    ) -> AttentionOperator:
        """Pair-attention operator on the same maps, facts and cells (A4 test, plan §12.3)."""
        transports: list[PathTransportOperator] = []
        latents: list[torch.Tensor] = []
        log_merged: list[torch.Tensor] = []
        if self.use_path_cells:
            for group in face_groups:
                if group.num_faces == 0:
                    continue
                transports.append(PathTransportOperator(
                    maps.src, maps.tgt, group.face_node_index, group.face_edge_index, edge_index,
                    shared_scale=True, face_edge_sign=group.face_edge_sign,
                ))
                # spec v1.6 E10: the cell scores' latents read the geometry [h0, c]
                latents.append(self._path_latent(h0 if geometry is None else geometry, group))
                merged = group.path_weight
                log_merged.append(h0.new_zeros(group.num_faces) if merged is None
                                  else merged.to(h0.dtype).clamp_min(1e-30).log())
        shared_roles = torch.zeros(self.relation_embedding.num_embeddings, dtype=torch.bool,
                                   device=h0.device)
        if self.shared_role_relations:
            shared_roles[list(self.shared_role_relations)] = True
        return build_attention_operator(
            self.pair_attention, maps, self.eta(), h0, edge_index, edge_type, shared_roles,
            transports, latents, log_merged, self.checkpointing,
        )

    def coupling_gates(self, h0: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """`(g, b)`, each `[N]` in (0, 1) or None when absent: `sigmoid(a^T h0 + beta)`,
        computed once from h0 like the maps (plan §10.4)."""
        listen = broadcast = None
        if self.listen_gate is not None:
            listen = torch.sigmoid(self.listen_gate(h0).squeeze(-1))
        if self.broadcast_gate is not None:
            broadcast = torch.sigmoid(self.broadcast_gate(h0).squeeze(-1))
        return listen, broadcast

    def build_operator(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        num_nodes: int,
    ) -> DiffusionOperator:
        geometry = self.geometry_inputs(h0, edge_index, edge_type)
        if self.identity_maps:
            identity = IdentityMapOperator.ones(edge_index.size(1), self.d, h0)
            maps = EdgeMapPair(src=identity, tgt=identity)
        else:
            maps = self.node_to_edge(
                h=geometry,
                edge_index=edge_index,
                edge_type=edge_type,
                relation_embeddings=self.relation_embedding.weight,
            ).maps
        if self.pair_attention is not None:
            # C4b arm unchanged: attention scores from h0; path latents from the shared encoder.
            return self._build_attention_operator(h0, maps, edge_index, edge_type, face_groups,
                                                  geometry)
        gamma = self.gamma()
        terms: tuple[PathTerm, ...] = ()
        weights: tuple[torch.Tensor, ...] = ()
        if self.use_path_cells:
            terms, weights = self.path_terms(geometry, maps, edge_index, face_groups,
                                             path_input=self.path_inputs(h0, geometry))
        if weights:
            if self.training:  # the adaptive proposal's input for the next epoch (Part H3)
                self.last_path_weights = tuple(w.detach() for w in weights)
            with torch.no_grad():
                log_w = torch.cat(weights).log()
                self.last_path_stats = {"path_log_w_max": float(log_w.max()),
                                        "path_log_w_mean": float(log_w.mean())}
        laplacian = VertexPathSheafLaplacian(
            edge_maps=maps,
            edge_index=edge_index,
            num_nodes=num_nodes,
            path_terms=terms,
            gamma=gamma if terms else None,
            checkpoint=self.checkpointing,
        )
        listen, broadcast = self.coupling_gates(geometry)
        return DiffusionOperator(laplacian=laplacian, eta=self.eta(), gamma=gamma,
                                 path_weights=weights, maps=maps, listen=listen,
                                 broadcast=broadcast)

    def diffuse(self, x0: torch.Tensor, operator: DiffusionOperator,
                trace: list[torch.Tensor] | None = None) -> torch.Tensor:
        """`encoder: first_order`: K steps of `x <- x - eta G Delta B x` (then layer dropout). G and
        B are skipped, not multiplied by ones, when absent, so `symmetric` runs the pre-D10
        operations. `trace`, if given, receives the state after every step (diagnostics only)."""
        if operator.eta is None:
            raise ValueError("diffuse is the first-order update; eta exists only under "
                             "encoder: first_order (use propagate).")
        x = x0
        for _ in range(self.num_steps):
            x = x - operator.eta * apply_operator(operator, x)
            x = self.layer_dropout(x)
            if trace is not None:
                trace.append(x)
        return x

    def propagate(self, h0: torch.Tensor, operator: DiffusionOperator) -> torch.Tensor:
        """`x_K`: the filter bank `sum_k T_k(A - I)(h0 W_k)` (then layer dropout, once), or the
        first-order update under `encoder: first_order`."""
        if self.filter_bank is None:
            return self.diffuse(h0, operator)
        return self.layer_dropout(self.filter_bank(lambda x: apply_operator(operator, x), h0))

    @property
    def diagnostics_available(self) -> bool:
        """Whether `eta_lambda_max` and `energy_trace` are defined (a Laplacian operator exists).
        The anchored path logs tau and the PCG statistics through `diffusion_scalars` instead."""
        return not self.h0_only and self.pair_attention is None and not self.anchored

    # ------------------------------------------------------------------ D10 diagnostics
    @staticmethod
    def coupling_scale(operator: DiffusionOperator, like: torch.Tensor) -> torch.Tensor:
        """`[N]` per-node `sqrt(g b)`: `G Delta B` is similar to `S Delta S` with this `S`."""
        scale = torch.ones(like.size(0), dtype=like.dtype, device=like.device)
        if operator.listen is not None:
            scale = scale * operator.listen
        if operator.broadcast is not None:
            scale = scale * operator.broadcast
        return scale.sqrt()

    def coupling_stats(self, operator: DiffusionOperator) -> dict[str, float]:
        """Mean and min of the gates present (detached floats, logging only)."""
        stats: dict[str, float] = {}
        if isinstance(operator, AttentionOperator):  # self-attention mass of every node
            stats["alpha_self_mean"] = float(operator.alpha_self.detach().mean())
            stats["alpha_self_min"] = float(operator.alpha_self.detach().min())
            return stats
        for name, gate in (("g", operator.listen), ("b", operator.broadcast)):
            if gate is not None and gate.numel():
                values = gate.detach()
                stats[f"{name}_mean"] = float(values.mean())
                stats[f"{name}_min"] = float(values.min())
        return stats

    @torch.no_grad()
    def lambda_max(self, operator: DiffusionOperator, num_nodes: int, steps: int = 30,
                   seed: int = 0) -> float:
        """Largest eigenvalue of `G Delta B` by power iteration on its symmetric similar form
        `(GB)^{1/2} Delta (GB)^{1/2}` (plan §10.3). Returns the Rayleigh quotient after
        `steps` steps: a lower bound of lambda_max that converges from below."""
        if isinstance(operator, AttentionOperator):
            raise ValueError("pair_attention has no Laplacian operator (use `diagnostics_available`).")
        probe = operator.maps.src.scale  # device and dtype only
        scale = self.coupling_scale(operator, probe.new_zeros(num_nodes)).unsqueeze(-1)
        generator = torch.Generator(device=probe.device).manual_seed(seed)
        x = torch.randn((num_nodes, self.d), generator=generator, device=probe.device,
                        dtype=probe.dtype)
        x = x / x.norm()

        def apply(v: torch.Tensor) -> torch.Tensor:
            return scale * operator.laplacian.apply_normalized(scale * v)

        for _ in range(steps):
            y = apply(x)
            norm = y.norm()
            if float(norm) == 0.0:
                return 0.0
            x = y / norm
        return float((x * apply(x)).sum())

    @torch.no_grad()
    def energy_trace(self, x: torch.Tensor, node_type: torch.Tensor,
                     edge_index: torch.Tensor, edge_type: torch.Tensor,
                     face_groups: tuple[FaceGroup, ...], num_nodes: int) -> dict[str, float]:
        """Normalized energy `<x_t, Delta_hat x_t> / ||x_t||^2` after each diffusion step
        t = 0..K, split into `energy_edge_t` and `energy_path_t` (plan §12.6 diagnostics; a fresh
        eval-mode operator, no grad; the training state is restored). Empty under
        `h0_only_readout`."""
        if self.h0_only:
            return {}
        was_training = self.training
        self.eval()
        try:
            h0 = self.encode_nodes(x, node_type)
            operator = self.build_operator(h0, edge_index, edge_type, face_groups, num_nodes)
            if self.filter_bank is not None:  # no intermediate steps: input and filter output
                states = {"0": h0, "out": self.propagate(h0, operator)}
            else:
                trajectory = [h0]
                self.diffuse(h0, operator, trajectory)
                states = {str(step): state for step, state in enumerate(trajectory)}
            trace: dict[str, float] = {}
            for step, state in states.items():
                edge, path = operator.laplacian.energy_parts(state)
                trace[f"energy_edge_{step}"] = float(edge)
                trace[f"energy_path_{step}"] = float(path)
            return trace
        finally:
            self.train(was_training)

    @torch.no_grad()
    def eta_lambda_max(self, x: torch.Tensor, node_type: torch.Tensor,
                       edge_index: torch.Tensor, edge_type: torch.Tensor,
                       face_groups: tuple[FaceGroup, ...], num_nodes: int,
                       steps: int = 30) -> float:
        """`eta * lambda_max(G Delta B)` of a fresh eval-mode operator (diagnostic, no grad);
        the training state of the module is restored."""
        was_training = self.training
        self.eval()
        try:
            h0 = self.encode_nodes(x, node_type)
            operator = self.build_operator(h0, edge_index, edge_type, face_groups, num_nodes)
            if operator.eta is None:
                raise ValueError("eta exists only under encoder: first_order; use "
                                 "operator_diagnostics.")
            return float(operator.eta) * self.lambda_max(operator, num_nodes, steps)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def operator_diagnostics(self, x: torch.Tensor, node_type: torch.Tensor,
                             edge_index: torch.Tensor, edge_type: torch.Tensor,
                             face_groups: tuple[FaceGroup, ...], num_nodes: int,
                             steps: int = 30) -> dict[str, float]:
        """Spectral diagnostics of a fresh eval-mode operator (no grad; training state restored).
        first_order: `{"eta_lambda_max"}` (D10). filter_bank: `lambda_max_A` (power iteration, a
        lower bound), `filter_norm_sum = sum_k ||W_k||_2` (Prop. 5's bound when the spectrum of A
        is in [0, 2]) and `filter_gain` (power iteration of F*F in M = B G^-1, a lower bound)."""
        if self.filter_bank is None:
            return {"eta_lambda_max": self.eta_lambda_max(x, node_type, edge_index, edge_type,
                                                          face_groups, num_nodes, steps)}
        was_training = self.training
        self.eval()
        try:
            h0 = self.encode_nodes(x, node_type)
            operator = self.build_operator(h0, edge_index, edge_type, face_groups, num_nodes)
            ones = h0.new_ones(num_nodes)
            listen = ones if operator.listen is None else operator.listen
            broadcast = ones if operator.broadcast is None else operator.broadcast
            gain = self.filter_bank.gain(lambda v: apply_operator(operator, v),
                                         broadcast / listen, (num_nodes, self.d), steps)
            return {
                "lambda_max_A": self.lambda_max(operator, num_nodes, steps),
                "filter_norm_sum": self.filter_bank.diagnostic_norm_sum(),
                "filter_gain": gain,
            }
        finally:
            self.train(was_training)

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
    ) -> EncodedState:
        """Shared encoder for all three tasks (a graph batch is one disjoint union). `epoch` sets
        the anchored solver's tolerance (fixed schedule; -1 = the final tolerance)."""
        h0 = self.encode_nodes(x, node_type)
        if self.h0_only:
            return EncodedState(h0=h0, x_star=h0, representation=h0, operator=None)
        if self.anchored:
            return self._encode_anchored(h0, edge_index, edge_type, face_groups, num_nodes, epoch)
        operator = self.build_operator(h0, edge_index, edge_type, face_groups, num_nodes)
        if self.training:
            self.last_coupling_stats = self.coupling_stats(operator)  # logging only
        x_star = self.propagate(h0, operator)
        return EncodedState(
            h0=h0,
            x_star=x_star,
            representation=torch.cat([x_star, h0], dim=-1),
            operator=operator,
        )

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
                         num_nodes: int, epoch: int) -> EncodedState:
        """`Y* = argmin 1/2||x - h0||^2 + tau/2 Y^T L_hat Y` on the three-level energy (R13):
        geometry once from `[h0, c]`, cell weights from g, then `AnchoredSolve` on the eliminated
        operator (fact states only on `E_shared`; `hub_A`: none). Arms: see `model.arm`."""
        geometry = self.geometry_inputs(h0, edge_index, edge_type)
        maps = self.node_to_edge(
            h=geometry, edge_index=edge_index, edge_type=edge_type,
            relation_embeddings=self.relation_embedding.weight,
        ).maps
        groups = tuple(group for group in face_groups if group.num_faces > 0)
        arm = self.arm
        cells_in_solve = arm not in ("closure_only", "no_path_cells", "instance_features")
        betas, weights, latents = self._cell_weights(h0, geometry, groups)
        if weights:
            if self.training:  # the adaptive proposal's input for the next epoch (Part H3)
                self.last_path_weights = tuple(w.detach() for w in weights)
            with torch.no_grad():
                log_w = torch.cat(weights).log()
                self.last_path_stats = {"path_log_w_max": float(log_w.max()),
                                        "path_log_w_mean": float(log_w.mean())}
        free_params: list[torch.Tensor] = []
        if arm == "free_cell_maps" and self.free_cell_maps is not None:
            for group, latent in zip(groups, latents):
                map_latent = latent if self.map_encoder is None else self._map_latent(geometry, group)
                # one free map per fact-cell incidence, indexed by the hop's near node and position
                free = self.free_cell_maps(geometry, group.face_node_index[:, :-1], map_latent)
                free_params.extend([free.reflectors, free.rotations])
        num_groups = len(groups)
        device = h0.device

        def build(scale: torch.Tensor, refl_src: torch.Tensor, rot_src: torch.Tensor,
                  refl_tgt: torch.Tensor, rot_tgt: torch.Tensor, *rest: torch.Tensor
                  ) -> EliminatedOperator:
            pair = EdgeMapPair(
                src=FactorizedMapOperator(scale, refl_src, rot_src, edge_type),
                tgt=FactorizedMapOperator(scale, refl_tgt, rot_tgt, edge_type),
            )
            group_betas, free_tensors = rest[:num_groups], rest[num_groups:]
            cell_groups, node_terms = [], []
            for i, (group, beta) in enumerate(zip(groups, group_betas)):
                if cells_in_solve:
                    free_maps = None
                    if arm == "free_cell_maps":
                        num_paths, length = group.face_edge_index.shape
                        free_maps = FactorizedMapOperator(
                            scale=beta.new_ones(num_paths * length),
                            reflectors=free_tensors[2 * i], rotations=free_tensors[2 * i + 1],
                            group_ids=torch.arange(length, device=device).repeat(num_paths),
                        )
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

        params = (maps.src.scale, maps.src.reflectors, maps.src.rotations,
                  maps.tgt.reflectors, maps.tgt.rotations, *betas, *free_params)
        warmup_epochs, warmup_tolerance, final_tolerance, max_iterations = self.solver_settings
        tolerance = solver_tolerance(epoch, warmup_epochs, warmup_tolerance, final_tolerance)
        with torch.no_grad():  # layout of the solved system (for the warm start and the log)
            complex_ = IncidenceComplex.from_face_groups(
                edge_index, num_nodes, groups if cells_in_solve else ())
            num_states = 0 if arm == "hub_A" else complex_.num_shared_facts
            rows = num_nodes + num_states + complex_.num_cells
        y0 = None
        if (not self.training and self._warm_start is not None
                and self._warm_start.shape == (rows, h0.size(-1))
                and self._warm_start.device == device):
            y0 = self._warm_start  # the evaluation draw is fixed: warm start from the last epoch
        stats = SolveStats()
        y_star = anchored_solve(build, h0, self.theta_tau, params, tolerance, max_iterations,
                                stats, y0=y0)
        if not self.training:
            self._warm_start = y_star.detach()
        x_star = y_star[:num_nodes]
        parts = [x_star, h0]
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
        """Node classification. `epoch` is accepted for trainer compatibility only."""
        if self.readout is None:
            raise RuntimeError(
                f"forward() is the node-classification head; this model was built for "
                f"'{self.task_type}' (use encode() with link_readout / graph_readout)."
            )
        encoded = self.encode(
            data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
            data.num_nodes, epoch=epoch,
        )
        logits = self.readout(encoded.representation[data.target_indices])
        operator = encoded.operator
        if operator is None:
            eta = self.eta()
            return H2SNNOutput(logits=logits, x_star=encoded.x_star, h0=encoded.h0,
                               eta=None if eta is None else eta.detach(), gamma=None,
                               path_gate_mean=None,
                               path_gate_max=None, max_edge_scale=None)
        gate_mean = gate_max = None
        if operator.path_weights:
            all_weights = torch.cat(operator.path_weights).detach()
            gate_mean, gate_max = all_weights.mean(), all_weights.max()
        max_scale = torch.maximum(operator.maps.src.scale.max(), operator.maps.tgt.scale.max())
        return H2SNNOutput(
            logits=logits,
            x_star=encoded.x_star,
            h0=encoded.h0,
            eta=None if operator.eta is None else operator.eta.detach(),
            gamma=None if operator.gamma is None else operator.gamma.detach(),
            path_gate_mean=gate_mean,
            path_gate_max=gate_max,
            max_edge_scale=max_scale.detach(),
        )


def apply_operator(operator: "DiffusionOperator | AttentionOperator", x: torch.Tensor) -> torch.Tensor:
    """`A x` with `A = G Delta_hat B` (gates skipped, not multiplied by ones, when absent), or
    `A = I - P` for the pair-attention arm (its update `x - eta (x - P x)`). Shared by the
    first-order update and the filter bank, so `first_order` is the D11 computation bit for bit."""
    if isinstance(operator, AttentionOperator):
        return x - operator.propagate(x)
    y = x if operator.broadcast is None else operator.broadcast.unsqueeze(-1) * x
    y = operator.laplacian.apply_normalized(y)
    if operator.listen is not None:
        y = operator.listen.unsqueeze(-1) * y
    return y


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
    return H2SNN(config=config, dims=dims)


def build_link_prediction_model(config: TrainConfig, data: PreparedLinkPredData) -> H2SNN:
    return H2SNN(config=config, dims=ModelDims.from_link_pred_data(data))


def build_graph_classification_model(config: TrainConfig, dims: ModelDims) -> H2SNN:
    return H2SNN(config=config, dims=dims)
