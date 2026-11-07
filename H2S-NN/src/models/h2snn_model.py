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


class EncodedState(NamedTuple):
    """Output of the shared encoder, before any task head."""

    h0: torch.Tensor
    x_star: torch.Tensor  # x_K (h0 under ablation.h0_only_readout)
    representation: torch.Tensor  # [x_K, h0] (h0 under ablation.h0_only_readout)
    operator: DiffusionOperator | None  # None under ablation.h0_only_readout


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


class PathEncoder(nn.Module):
    """Positional `concat` encoder over the ordered `h0` of a path and its schema's relation
    embeddings: one `Linear + GELU` per path length; its output feeds the path gate."""

    def __init__(self, d: int, d_r: int, hidden_dim: int, face_lengths: tuple[int, ...],
                 factored_inputs: bool = False) -> None:
        """`d`: width of the per-node input (`h0`, or `[h0, c]` under `geometry_context`).
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
        schema_relations: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """`[P, L+1]` paths with relation sequence `[L]` -> `[P, hidden]` latents."""
        length = int(face_node_index.size(1)) - 1
        key = str(length)
        if key not in self.input_proj:
            available = ", ".join(sorted(self.input_proj)) or "none"
            raise KeyError(
                f"PathEncoder has no projection for path length {length} (built for: "
                f"{available}). ModelDims.face_lengths must list every length in the data."
            )
        num_paths = face_node_index.size(0)
        schema_token = relation_embeddings[schema_relations].reshape(1, -1)
        if self.factored_inputs:
            blocks = [(h0, face_node_index[:, j]) for j in range(length + 1)]
            blocks.append((schema_token, face_node_index.new_zeros(num_paths)))
            return self.activation(gathered_linear(self.input_proj[key], blocks))
        inputs = torch.cat(
            [h0[face_node_index].reshape(num_paths, -1), schema_token.expand(num_paths, -1)],
            dim=-1,
        )
        return self.activation(self.input_proj[key](inputs))


class EndpointEncoder(nn.Module):
    """Latent of a path from its endpoints and schema only, `[h_{v_0}, h_{v_L}, schema]`
    (ablation `path_mode: endpoint_edges`: no intermediate node enters)."""

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
        schema_relations: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        length = int(face_node_index.size(1)) - 1
        num_paths = face_node_index.size(0)
        schema_token = relation_embeddings[schema_relations].reshape(1, -1)
        endpoints = face_node_index[:, [0, length]]
        if self.factored_inputs:
            blocks = [(h0, endpoints[:, 0]), (h0, endpoints[:, 1]),
                      (schema_token, face_node_index.new_zeros(num_paths))]
            return self.activation(gathered_linear(self.input_proj[str(length)], blocks))
        inputs = torch.cat(
            [h0[endpoints].reshape(num_paths, -1), schema_token.expand(num_paths, -1)], dim=-1
        )
        return self.activation(self.input_proj[str(length)](inputs))


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
            if model_config.encoder == "first_order" else None
        )
        self.use_path_cells = not ablation.no_path_cells
        self.path_mode = ablation.path_mode
        self.free_path_maps = ablation.free_path_maps
        self.theta_gamma: nn.Parameter | None = None
        self.path_encoder: PathEncoder | EndpointEncoder | None = None
        self.path_gate: nn.Linear | None = None
        self.free_maps: FreeIncidenceMaps | None = None
        self.endpoint_maps: EndpointEdgeMaps | None = None
        if self.use_path_cells:
            self.theta_gamma = nn.Parameter(
                torch.tensor(_softplus_inverse(model_config.gamma_init))
            )
            hidden = model_config.face_mlp_hidden_dim
            if self.path_mode == "endpoint_edges":
                # No intermediate node enters: neither the maps nor the gate see them.
                self.path_encoder = EndpointEncoder(geometry_dim, model_config.d_r, hidden,
                                                    dims.face_lengths,
                                                    factored_inputs=model_config.geometry_context)
                self.endpoint_maps = EndpointEdgeMaps(
                    d=d, latent_dim=hidden, householder_k=model_config.householder_k,
                    map_scale=self.map_scale, scale_eps=model_config.map_scale_eps,
                    orthogonal_map=model_config.orthogonal_map,
                )
            elif self.free_path_maps or not ablation.uniform_path_gate:
                self.path_encoder = PathEncoder(geometry_dim, model_config.d_r, hidden,
                                                dims.face_lengths,
                                                factored_inputs=model_config.geometry_context)
            if self.free_path_maps:
                self.free_maps = FreeIncidenceMaps(
                    d=d, latent_dim=hidden, hidden_dim=model_config.edge_mlp_hidden_dim,
                    householder_k=model_config.householder_k,
                    max_positions=max(dims.face_lengths, default=1) + 1,
                    orthogonal_map=model_config.orthogonal_map,
                    node_dim=geometry_dim,
                    factored_inputs=model_config.geometry_context,
                )
            if not ablation.uniform_path_gate:
                # Gate g: zero bias, default weight init, so w_f = sigmoid(g(.)) starts near
                # 0.5 with a small spread and the path encoder gets gradients from step 0.
                self.path_gate = nn.Linear(hidden, 1)
                nn.init.zeros_(self.path_gate.bias)
        self.layer_dropout = nn.Dropout(model_config.layer_dropout)
        self.coupling = ablation.coupling
        self.listen_gate: nn.Linear | None = None
        self.broadcast_gate: nn.Linear | None = None
        self.last_coupling_stats: dict[str, float] = {}
        # Gate heads are created last, so the other parameters initialise exactly as before
        # (`symmetric` creates none: the pre-D10 model, bit for bit). No heads without
        # diffusion (`h0_only_readout`).
        if not self.h0_only and self.coupling in ("listen_only", "cooperative"):
            self.listen_gate = _gate_head(geometry_dim, model_config.coupling_gate_init)
            if self.coupling == "cooperative":
                self.broadcast_gate = _gate_head(geometry_dim, model_config.coupling_gate_init)

        # Task head on the node representation [x_K, h0] (h0 under h0_only_readout).
        representation_dim = d if self.h0_only else 2 * d
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
        if model_config.encoder == "filter_bank" and not self.h0_only:
            self.filter_bank = ChebyshevFilterBank(d, model_config.K, model_config.eta_init)

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

    def diffusion_scalars(self) -> dict[str, float]:
        """Current eta and gamma, for per-epoch logging."""
        with torch.no_grad():
            scalars: dict[str, float] = {}
            eta = self.eta()
            if eta is not None:
                scalars["eta"] = float(eta)
            if self.filter_bank is not None:  # sum_k ||W_k||_2 (Prop. 5 bound), cheap
                scalars["filter_norm_sum"] = float(
                    torch.linalg.matrix_norm(self.filter_bank.weights(), ord=2).sum())
            gamma = self.gamma()
            if gamma is not None:
                scalars["gamma"] = float(gamma)
        scalars.update(self.last_coupling_stats)  # gates of the last training forward
        return scalars

    def scalar_parameters(self) -> list[nn.Parameter]:
        """`theta_eta`, `theta_gamma` and the gate heads (weights and biases): the optimizer gives
        them their own group (plan §12.4). Sigmoid-saturated scalars move too little per
        full-batch step at the base rate."""
        scalars = [p for p in (self.theta_eta, self.theta_gamma) if p is not None]
        for head in (self.listen_gate, self.broadcast_gate):
            if head is not None:
                scalars.extend(head.parameters())
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

    def _path_latent(self, h0: torch.Tensor, group: FaceGroup) -> torch.Tensor | None:
        """`[P, hidden]` path latent (endpoint-only under `endpoint_edges`), or None."""
        if self.path_encoder is None:
            return None
        encoder = self.path_encoder
        return checkpoint_if_grad(
            lambda h_in, rel_in: encoder(
                h_in, group.face_node_index, group.relation_index[0], rel_in
            ),
            h0,
            self.relation_embedding.weight,
            enabled=self.checkpointing,
        )

    def _path_weights(self, latent: torch.Tensor | None, group: FaceGroup,
                      like: torch.Tensor) -> torch.Tensor:
        """`w_f = sigmoid(g(latent_f))`, or 1 under `uniform_path_gate`."""
        if self.path_gate is None:
            return like.new_ones(group.num_faces)
        return torch.sigmoid(self.path_gate(latent).squeeze(-1))

    def path_terms(
        self,
        h0: torch.Tensor,
        maps: EdgeMapPair,
        edge_index: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
    ) -> tuple[tuple[PathTerm, ...], tuple[torch.Tensor, ...]]:
        """One coupling term per face group, per the ablation switches, and its `w_f`. `h0` is the
        per-node geometry input (`geometry_inputs`: `[h0, c]` under `geometry_context`)."""
        terms: list[PathTerm] = []
        weights: list[torch.Tensor] = []
        shared = self.map_scale == "per_edge"
        for group in face_groups:
            if group.num_faces == 0:
                continue
            latent = self._path_latent(h0, group)
            group_weights = self._path_weights(latent, group, h0)
            gate_weights = group_weights
            if group.path_weight is not None:  # merged weight m_f, in both L_P and D (D8)
                group_weights = group_weights * group.path_weight.to(group_weights.dtype)
            nodes = group.face_node_index
            if self.path_mode == "endpoint_edges":
                endpoints = torch.stack([nodes[:, 0], nodes[:, -1]])
                term: PathTerm = EdgePairTerm(self.endpoint_maps(latent), endpoints, group_weights)
            elif self.free_path_maps:
                incidence = FreeIncidenceOperator(self.free_maps(h0, nodes, latent), nodes)
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
            terms, weights = self.path_terms(geometry, maps, edge_index, face_groups)
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
        """Whether `eta_lambda_max` and `energy_trace` are defined (a Laplacian operator exists)."""
        return not self.h0_only and self.pair_attention is None

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
                "filter_norm_sum": float(
                    torch.linalg.matrix_norm(self.filter_bank.weights(), ord=2).sum()),
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
    ) -> EncodedState:
        """Shared encoder for all three tasks (a graph batch is one disjoint union)."""
        h0 = self.encode_nodes(x, node_type)
        if self.h0_only:
            return EncodedState(h0=h0, x_star=h0, representation=h0, operator=None)
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
            data.num_nodes,
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
    return H2SNN(config=config, dims=ModelDims.from_prepared_data(data))


def build_link_prediction_model(config: TrainConfig, data: PreparedLinkPredData) -> H2SNN:
    return H2SNN(config=config, dims=ModelDims.from_link_pred_data(data))


def build_graph_classification_model(config: TrainConfig, dims: ModelDims) -> H2SNN:
    return H2SNN(config=config, dims=dims)
