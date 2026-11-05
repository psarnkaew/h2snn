"""H2S-NN v2: node diffusion with the vertex-2-cell sheaf Laplacian (v2 decision D4, plan §4).

    h0 = TypeEncoder(x)
    repeat K steps:
        x <- x - eta D^{-1/2} (Delta_0 + gamma L_P) D^{-1/2} x       # then layer dropout
    logits = MLP([x_K, h0])

`Delta_0` runs over the observed edges only (no closure edges) with the D1 map operators;
`L_P` couples every path 2-cell to every node on it through ordered parallel transport
(`src/models/vertex_path_laplacian.py`). `eta = sigmoid(theta_eta)`,
`gamma = softplus(theta_gamma)`, `w_f = sigmoid(g(path_latent_f))`.

v1's edge cochain, `L_1`, `EdgeInitializer`, the `delta_1` face term, the `A_f` head, face
caches, closure edges, edge LayerNorm, edge pooling and the operator ramp do not exist here.
Node classification only.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from src.configs.config import TrainConfig
from src.data.structures import FaceGroup, PreparedH2SData
from src.models.map_operators import EdgeMapPair, PathTransportOperator, checkpoint_if_grad
from src.models.node_encoder import TypeSpecificNodeEncoder
from src.models.sheaf_maps import DenseConformalNodeToEdgeSheaf, FactorizedNodeToEdgeSheaf
from src.models.vertex_path_laplacian import PathCellTerm, PathTerm, VertexPathSheafLaplacian

_V2_UNSUPPORTED_TASKS = ("link_prediction", "graph_classification")


class H2SNNv2Output(NamedTuple):
    logits: torch.Tensor
    x_star: torch.Tensor  # x_K (h0 under readout "h0_only")
    h0: torch.Tensor
    # Detached diagnostics for logging (no effect on optimization).
    eta: torch.Tensor
    gamma: torch.Tensor | None
    path_gate_mean: torch.Tensor | None
    path_gate_max: torch.Tensor | None
    max_edge_scale: torch.Tensor | None


class V2Operator(NamedTuple):
    """Everything one forward pass diffuses with; `forward` and the tests share it."""

    laplacian: VertexPathSheafLaplacian
    eta: torch.Tensor
    gamma: torch.Tensor | None
    path_weights: tuple[torch.Tensor, ...]
    maps: EdgeMapPair


class PathEncoder(nn.Module):
    """Positional `concat` encoder over the ordered `h0` of a path and its schema's relation
    embeddings: one `Linear + GELU` per path length (v1's `SequenceEdgeToFaceSheaf` encoder
    without its `A_f` map head)."""

    def __init__(self, d: int, d_r: int, hidden_dim: int, face_lengths: tuple[int, ...]) -> None:
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
        inputs = torch.cat(
            [h0[face_node_index].reshape(num_paths, -1), schema_token.expand(num_paths, -1)],
            dim=-1,
        )
        return self.activation(self.input_proj[key](inputs))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _softplus_inverse(y: float) -> float:
    return y + math.log(-math.expm1(-y))


class H2SNNv2(nn.Module):
    def __init__(self, config: TrainConfig, dims: "ModelDims") -> None:  # noqa: F821
        super().__init__()
        model_config = config.model
        if model_config.architecture != "v2":
            raise ValueError("H2SNNv2 needs model.architecture='v2'.")
        if config.data.task_type in _V2_UNSUPPORTED_TASKS:
            raise NotImplementedError(
                f"architecture 'v2' is not implemented for task '{config.data.task_type}' "
                "(D4 covers node classification only)."
            )
        if dims.discovery is not None:
            raise NotImplementedError(
                "architecture 'v2' does not support AMPSS auto-discovery: v1's Stage-II "
                "weights, warm-up and operator ramp are deleted, and the replacement (D2) is "
                "not decided. Use fixed schema lists (discovery disabled)."
            )
        unsupported = []
        if model_config.path_maps != "transport":
            unsupported.append(f"path_maps={model_config.path_maps}")
        if model_config.path_mode != "cells":
            unsupported.append(f"path_mode={model_config.path_mode}")
        if unsupported:
            raise NotImplementedError(f"{', '.join(unsupported)} lands in D4 unit 5.")
        self.config = config
        self.model_config = model_config
        d = model_config.d
        self.d = d
        self.num_steps = model_config.K
        self.checkpointing = model_config.checkpointing

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
                map_scale=model_config.map_scale,
                checkpoint=model_config.checkpointing,
            )
        else:
            self.node_to_edge = DenseConformalNodeToEdgeSheaf(
                d=d,
                d_r=model_config.d_r,
                num_relations=dims.num_relations,
                hidden_dim=model_config.edge_mlp_hidden_dim,
                scale_eps=model_config.map_scale_eps,
                orthogonal_map=model_config.orthogonal_map,
                map_scale=model_config.map_scale,
                memory_guard_fraction=model_config.node_edge_memory_guard_fraction,
            )
        self.theta_eta = nn.Parameter(torch.tensor(_logit(model_config.eta_init)))
        self.use_path_cells = not model_config.ablate_no_path_cells
        self.theta_gamma: nn.Parameter | None = None
        self.path_encoder: PathEncoder | None = None
        self.path_gate: nn.Linear | None = None
        if self.use_path_cells:
            self.theta_gamma = nn.Parameter(
                torch.tensor(_softplus_inverse(model_config.gamma_init))
            )
            if not model_config.uniform_path_gate:
                self.path_encoder = PathEncoder(
                    d=d,
                    d_r=model_config.d_r,
                    hidden_dim=model_config.face_mlp_hidden_dim,
                    face_lengths=dims.face_lengths,
                )
                # Gate g: zero bias, default weight init, so w_f = sigmoid(g(.)) starts near
                # 0.5 with a small spread and the path encoder gets gradients from step 0.
                self.path_gate = nn.Linear(self.path_encoder.latent_dim, 1)
                nn.init.zeros_(self.path_gate.bias)
        self.layer_dropout = nn.Dropout(model_config.layer_dropout)
        readout_in = d if model_config.readout == "h0_only" else 2 * d
        self.readout = nn.Sequential(
            nn.Linear(readout_in, model_config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.classifier_dropout),
            nn.Linear(model_config.classifier_hidden_dim, dims.num_classes),
        )

    # ------------------------------------------------------------------ scalars
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_eta)

    def gamma(self) -> torch.Tensor | None:
        if self.theta_gamma is None:
            return None
        return nn.functional.softplus(self.theta_gamma)

    # ------------------------------------------------------------------ operator
    def encode_nodes(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        if self.node_encoder is not None:
            return self.node_encoder(x, node_type)
        return self.identity_embedding.weight  # type: ignore[union-attr]

    def _path_weights(self, h0: torch.Tensor, group: FaceGroup) -> torch.Tensor:
        if self.path_encoder is None:
            return h0.new_ones(group.num_faces)
        encoder = self.path_encoder
        gate = self.path_gate
        relation_embeddings = self.relation_embedding.weight

        def weights(h_in: torch.Tensor, rel_in: torch.Tensor) -> torch.Tensor:
            latent = encoder(h_in, group.face_node_index, group.relation_index[0], rel_in)
            return torch.sigmoid(gate(latent).squeeze(-1))

        return checkpoint_if_grad(weights, h0, relation_embeddings, enabled=self.checkpointing)

    def build_operator(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        num_nodes: int,
    ) -> V2Operator:
        maps = self.node_to_edge(
            h=h0,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_embeddings=self.relation_embedding.weight,
        ).maps
        gamma = self.gamma()
        terms: list[PathTerm] = []
        weights: list[torch.Tensor] = []
        if self.use_path_cells:
            shared = self.model_config.map_scale == "per_edge"
            for group in face_groups:
                if group.num_faces == 0:
                    continue
                transport = PathTransportOperator(
                    maps.src, maps.tgt, group.face_node_index, group.face_edge_index,
                    edge_index, shared_scale=shared,
                )
                group_weights = self._path_weights(h0, group)
                terms.append(PathCellTerm(transport, group_weights))
                weights.append(group_weights)
        laplacian = VertexPathSheafLaplacian(
            edge_maps=maps,
            edge_index=edge_index,
            num_nodes=num_nodes,
            path_terms=tuple(terms),
            gamma=gamma if terms else None,
            checkpoint=self.checkpointing,
        )
        return V2Operator(laplacian=laplacian, eta=self.eta(), gamma=gamma,
                          path_weights=tuple(weights), maps=maps)

    def diffuse(self, x0: torch.Tensor, operator: V2Operator) -> torch.Tensor:
        x = x0
        for _ in range(self.num_steps):
            x = x - operator.eta * operator.laplacian.apply_normalized(x)
            x = self.layer_dropout(x)
        return x

    # ------------------------------------------------------------------ forward
    def forward(self, data: PreparedH2SData, epoch: int = -1) -> H2SNNv2Output:
        """`epoch` is accepted for trainer compatibility; v2 has no epoch-dependent ramp."""
        h0 = self.encode_nodes(data.x, data.node_type)
        if self.model_config.readout == "h0_only":
            logits = self.readout(h0[data.target_indices])
            return H2SNNv2Output(logits=logits, x_star=h0, h0=h0, eta=self.eta().detach(),
                                 gamma=None, path_gate_mean=None, path_gate_max=None,
                                 max_edge_scale=None)
        operator = self.build_operator(
            h0, data.edge_index, data.edge_type, data.face_groups, data.num_nodes
        )
        x_star = self.diffuse(h0, operator)
        logits = self.readout(torch.cat([x_star, h0], dim=-1)[data.target_indices])
        gate_mean = gate_max = None
        if operator.path_weights:
            all_weights = torch.cat(operator.path_weights).detach()
            gate_mean, gate_max = all_weights.mean(), all_weights.max()
        max_scale = torch.maximum(operator.maps.src.scale.max(), operator.maps.tgt.scale.max())
        return H2SNNv2Output(
            logits=logits,
            x_star=x_star,
            h0=h0,
            eta=operator.eta.detach(),
            gamma=None if operator.gamma is None else operator.gamma.detach(),
            path_gate_mean=gate_mean,
            path_gate_max=gate_max,
            max_edge_scale=max_scale.detach(),
        )
