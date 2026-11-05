from __future__ import annotations

from collections.abc import Callable
from typing import Mapping, NamedTuple

import torch
from torch import nn

from src.configs.config import TrainConfig
from src.data.link_pred_structures import PreparedLinkPredData
from src.data.schema_structures import DiscoveryMetadata
from src.data.structures import FaceGroup, PreparedH2SData, face_lengths_of
from src.models.coboundary_ops import CoboundaryOps
from src.models.diffusion import HodgeDiffusion
from src.models.graph_pooling import GraphPooling, build_graph_pooling
from src.models.h2snn_v2 import H2SNNv2
from src.models.s2pool_h2 import S2PoolH2, GalerkinSheafPool, SheafSpectralPool
from src.models.link_readout import LinkPredictionReadout
from src.models.node_encoder import TypeSpecificNodeEncoder
from src.models.readout import NodeReadout
from src.models.instance_weights import InstanceWeightNetwork
from src.models.map_operators import DenseMapOperator, EdgeMapPair, StackedMapOperator
from src.models.sheaf_maps import (
    EdgeInitializer,
    FactorizedNodeToEdgeSheaf,
    NodeToEdgeSheaf,
    SequenceEdgeToFaceSheaf,
)

# Tasks whose readouts consume dense [E, d, d] maps from `EncodedState.F_src/F_tgt`.
_DENSE_MAP_ONLY_TASKS = ("link_prediction", "graph_classification")


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
    discovery: DiscoveryMetadata | None = None
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
            discovery=data.discovery,
            num_nodes=data.x.size(0),
            type_input_dims=type_dims,
            type_num_nodes=type_counts,
            type_offsets=type_offsets,
            face_lengths=face_lengths_of(data.face_groups),
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
            discovery=data.discovery,
            num_nodes=data.x.size(0),
            num_target_relations=data.num_target_relations,
            type_input_dims=type_dims,
            type_num_nodes=type_counts,
            type_offsets=type_offsets,
            face_lengths=face_lengths_of(data.face_groups),
        )


class EncodedState(NamedTuple):
    """Everything the shared sheaf-diffusion backbone produces, before any
    task-specific readout is applied."""

    h0: torch.Tensor
    x_star: torch.Tensor
    s_star: torch.Tensor
    delta0_h0: torch.Tensor
    delta0_x_star: torch.Tensor
    raw_face_curl: torch.Tensor
    delta1_delta0_h0: torch.Tensor
    # Dense [E_ext, d, d] maps under map_family="dense"; None under "factorized", whose
    # maps are the operators in `edge_maps` (never materialized for the observed edges).
    F_src: torch.Tensor | None
    F_tgt: torch.Tensor | None
    # Edge list *extended* with the synthetic closure 1-cells, in the order the edge
    # cochain uses. Every consumer of `s_star`/`delta0_*` must index with this rather than
    # with `data.edge_index`, which covers real edges only.
    edge_index: torch.Tensor
    num_real_edges: int
    instance_weight_means: torch.Tensor | None
    instance_weight_stds: torch.Tensor | None
    instance_weight_nonempty_mask: torch.Tensor | None
    operator_ramp: float
    x_trajectory: list[torch.Tensor] | None = None
    # Extended-edge map operators under map_family="factorized"; None under "dense".
    edge_maps: EdgeMapPair | None = None


class H2SNNOutput(NamedTuple):
    logits: torch.Tensor
    delta0_x_star: torch.Tensor
    raw_face_curl: torch.Tensor
    delta1_delta0_h0: torch.Tensor
    delta0_h0: torch.Tensor
    x_star: torch.Tensor
    s_star: torch.Tensor
    instance_weight_means: torch.Tensor | None
    instance_weight_stds: torch.Tensor | None
    instance_weight_nonempty_mask: torch.Tensor | None
    operator_ramp: float


class H2SNN(nn.Module):
    def __init__(self, config: TrainConfig, dims: ModelDims) -> None:
        super().__init__()
        if config.model.architecture != "v1":
            raise NotImplementedError(
                f"H2SNN is the v1 model; model.architecture='{config.model.architecture}' "
                "(v2 decision D4) is node classification only and is built by build_model. "
                "Link prediction and graph classification are not implemented for v2 (D4 scope)."
            )
        self.config = config
        self.model_config = config.model
        self.data_config = config.data
        self.map_family = self.model_config.map_family
        if self.map_family == "factorized" and config.data.task_type in _DENSE_MAP_ONLY_TASKS:
            raise NotImplementedError(
                f"model.map_family='factorized' is not implemented for task "
                f"'{config.data.task_type}': v2 decision D1 covers node classification only. "
                "Graph classification, link prediction, SheafCovPool, S2Pool and THS-Pool "
                "consume dense [E, d, d] maps; use map_family='dense' there."
            )

        # Feature-less graphs (e.g. LastFM/AmazonBook link prediction, which ship no
        # node features at all) would collapse every node of a type to an identical
        # vector through nn.Linear(0, d)'s bias-only output — use a learned per-node
        # "free" embedding table instead in that case.
        self.node_encoder: TypeSpecificNodeEncoder | None = None
        self.identity_embedding: nn.Embedding | None = None
        if dims.in_channels == 0:
            if dims.num_nodes is None:
                raise ValueError("ModelDims.num_nodes is required when in_channels == 0.")
            self.identity_embedding = nn.Embedding(dims.num_nodes, self.model_config.d)
        else:
            self.node_encoder = TypeSpecificNodeEncoder(
                input_dim=dims.in_channels,
                hidden_dim=self.model_config.d,
                num_node_types=dims.num_node_types,
                dropout=self.model_config.encoder_dropout,
                type_input_dims=dims.type_input_dims,
                type_num_nodes=dims.type_num_nodes,
                type_offsets=dims.type_offsets,
            )
        # One extra row reserved for synthetic closure edges, which carry no data relation.
        # Its index is `dims.num_relations` (see `closure_relation_id`).
        self.closure_relation_id = dims.num_relations
        self.relation_embedding = nn.Embedding(
            dims.num_relations + 1, self.model_config.d_r
        )
        self.node_to_edge: NodeToEdgeSheaf | FactorizedNodeToEdgeSheaf
        if self.map_family == "dense":
            self.node_to_edge = NodeToEdgeSheaf(
                d=self.model_config.d,
                d_r=self.model_config.d_r,
                num_relations=dims.num_relations,
                hidden_dim=self.model_config.edge_mlp_hidden_dim,
                use_orthogonal=self.model_config.use_orthogonal,
                orthogonal_map=self.model_config.orthogonal_map,
                memory_guard_fraction=self.model_config.node_edge_memory_guard_fraction,
            )
        else:
            self.node_to_edge = FactorizedNodeToEdgeSheaf(
                d=self.model_config.d,
                d_r=self.model_config.d_r,
                num_relations=dims.num_relations,
                hidden_dim=self.model_config.edge_mlp_hidden_dim,
                householder_k=self.model_config.householder_k,
                scale_eps=self.model_config.map_scale_eps,
                orthogonal_map=self.model_config.orthogonal_map,
                map_scale=self.model_config.map_scale,
            )
        self.edge_initializer = EdgeInitializer(
            d=self.model_config.d,
            d_r=self.model_config.d_r,
            hidden_dim=self.model_config.edge_init_hidden_dim,
            use_delta0_init=self.model_config.use_delta0_init,
        )
        self.edge_to_face = SequenceEdgeToFaceSheaf(
            d=self.model_config.d,
            d_r=self.model_config.d_r,
            d_h=self.model_config.d_h,
            hidden_dim=self.model_config.face_mlp_hidden_dim,
            face_chunk_size=self.model_config.face_chunk_size,
            face_encoder=self.model_config.face_encoder,
            face_map_param=self.model_config.face_map_param,
            face_lengths=dims.face_lengths,
        )
        self.diffusion = HodgeDiffusion(
            num_steps=self.model_config.K,
            eta_0=self.model_config.eta_0,
            eta_1=self.model_config.eta_1,
            learnable_eta=self.model_config.learnable_eta,
            layer_dropout=self.model_config.layer_dropout,
        )
        self.node_norm = nn.LayerNorm(self.model_config.d)
        self.edge_norm = nn.LayerNorm(self.model_config.d)

        # Task-specific readout head. Registered as a submodule (not passed around
        # separately) so its parameters are picked up by build_optimizer(model) for
        # free.
        self.readout: NodeReadout | None = None
        self.link_readout: LinkPredictionReadout | None = None
        self.graph_pooling: GraphPooling | None = None
        if config.data.task_type == "link_prediction":
            self.link_readout = LinkPredictionReadout(
                d=self.model_config.d,
                decoder_type=config.data.decoder_type,
                hidden_dim=self.model_config.classifier_hidden_dim,
                dropout=self.model_config.classifier_dropout,
                num_relations=dims.num_target_relations,
            )
        elif config.data.task_type == "graph_classification":
            if config.data.pooling_type == "ths_spectral":
                # THS-Pool (spectral): gauge+perm-invariant heat-kernel descriptor of L0 + local energies.
                self.graph_pooling = SheafSpectralPool(
                    d=self.model_config.d,
                    num_node_types=dims.num_node_types,
                    num_relations=dims.num_relations,
                    num_classes=dims.num_classes,
                    hidden_dim=self.model_config.classifier_hidden_dim,
                    dropout=self.model_config.classifier_dropout,
                    use_spectral=self.model_config.pool_use_spectral,
                )
            elif config.data.pooling_type == "ths_galerkin":
                # THS-Pool (Galerkin): energy-preserving coarse operator L0_c = S^T L0 (S z),
                # matrix-free over the fine maps. Called directly (no backbone re-run/callbacks).
                self.graph_pooling = GalerkinSheafPool(
                    d=self.model_config.d,
                    num_node_types=dims.num_node_types,
                    num_relations=dims.num_relations,
                    num_classes=dims.num_classes,
                    hidden_dim=self.model_config.classifier_hidden_dim,
                    dropout=self.model_config.classifier_dropout,
                    clusters_per_type=self.model_config.pool_clusters_per_type,
                    coarse_steps=self.model_config.pool_coarse_steps,
                    use_coarse=self.model_config.pool_use_coarse,
                )
            elif config.data.pooling_type == "s2pool_h2":
                # THS-Pool (superseded relearn/inject): re-enters the backbone per level via callbacks.
                self.graph_pooling = S2PoolH2(
                    d=self.model_config.d,
                    num_node_types=dims.num_node_types,
                    num_relations=dims.num_relations,
                    num_classes=dims.num_classes,
                    hidden_dim=self.model_config.classifier_hidden_dim,
                    dropout=self.model_config.classifier_dropout,
                    num_levels=self.model_config.pool_num_levels,
                    ratio=self.model_config.pool_ratio,
                    inject_maps=self.model_config.pool_inject_maps,
                )
            else:
                self.graph_pooling = build_graph_pooling(
                    pooling_type=config.data.pooling_type,
                    d=self.model_config.d,
                    hidden_dim=self.model_config.classifier_hidden_dim,
                    num_classes=dims.num_classes,
                    dropout=self.model_config.classifier_dropout,
                    sheafcov_feat_dim=self.model_config.sheafcov_feat_dim,
                )
        else:
            self.readout = NodeReadout(
                d=self.model_config.d,
                num_classes=dims.num_classes,
                hidden_dim=self.model_config.classifier_hidden_dim,
                dropout=self.model_config.classifier_dropout,
            )

        # Stage-II weighting is built only for an auto-discovered schema pool.
        self.instance_weight_network: InstanceWeightNetwork | None = None
        self.schema_embeddings: nn.Embedding | None = None

        if dims.discovery is not None:
            num_schemas = dims.discovery.num_schemas
            mode = config.discovery.instance_weight_mode
            if mode == "schema_conditioned":
                self.schema_embeddings = nn.Embedding(
                    num_schemas, self.model_config.d_r
                )
            if mode != "uniform":
                input_dim = self.edge_to_face.face_latent_dim
                if mode == "schema_conditioned":
                    input_dim += self.model_config.d_r
                self.instance_weight_network = InstanceWeightNetwork(
                    input_dim=input_dim,
                    hidden_dim=self.model_config.face_mlp_hidden_dim,
                )

    def _build_instance_weight_scorer(
        self,
        discovery: DiscoveryMetadata | None,
        epoch: int,
    ) -> Callable[[int, torch.Tensor], torch.Tensor] | None:
        """Build a chunk-local scorer over the ordered face-encoder latent."""
        if (
            discovery is None
            or self.instance_weight_network is None
            or self.config.discovery.instance_weight_mode == "uniform"
            or (epoch >= 0 and epoch < self.config.discovery.warmup_epochs)
        ):
            return None

        def score(group_id: int, instance_latent: torch.Tensor) -> torch.Tensor:
            weight_input = instance_latent
            if self.config.discovery.instance_weight_mode == "schema_conditioned":
                if self.schema_embeddings is None:
                    raise RuntimeError(
                        "Schema-conditioned instance weights require schema embeddings."
                    )
                schema_idx = int(discovery.schema_ids[group_id].item())
                schema_emb = self.schema_embeddings(
                    torch.tensor(schema_idx, device=instance_latent.device)
                ).unsqueeze(0).expand(instance_latent.size(0), -1)
                weight_input = torch.cat([instance_latent, schema_emb], dim=-1)
            return self.instance_weight_network(weight_input)

        return score

    def _operator_ramp(self, epoch: int) -> float:
        if epoch < 0:
            return 1.0
        anneal_epochs = self.config.discovery.operator_anneal_epochs
        if anneal_epochs <= 0:
            return 1.0
        epochs_since_warmup = epoch - self.config.discovery.warmup_epochs
        return min(1.0, max(0.0, epochs_since_warmup / anneal_epochs))

    def _build_face_weights(
        self,
        face_groups: tuple[FaceGroup, ...],
        discovery: DiscoveryMetadata | None,
        h0: torch.Tensor,
        epoch: int = -1,
        learned_instance_weights: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[
        tuple[torch.Tensor, ...] | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return effective upper-Hodge weights and per-schema diagnostics."""
        if discovery is None:
            return None, None, None, None

        use_uniform = (
            self.config.discovery.instance_weight_mode == "uniform"
            or (epoch >= 0 and epoch < self.config.discovery.warmup_epochs)
        )
        weights: list[torch.Tensor] = []
        operator_ramp = self._operator_ramp(epoch)
        effective_by_schema: list[list[torch.Tensor]] = [
            [] for _ in range(discovery.num_schemas)
        ]
        for group_id, face_group in enumerate(face_groups):
            schema_idx = int(discovery.schema_ids[group_id].item())
            if use_uniform:
                effective_weight = h0.new_ones(face_group.num_faces)
            else:
                if learned_instance_weights is None:
                    raise RuntimeError(
                        "Learned instance weights are required in non-uniform mode."
                    )
                learned_weight = learned_instance_weights[group_id]
                if learned_weight.numel() != face_group.num_faces:
                    raise ValueError(
                        "Ordered instance-weight count does not match face-group size."
                    )
                effective_weight = (
                    (1.0 - operator_ramp) + operator_ramp * learned_weight
                )
            weights.append(effective_weight)
            effective_by_schema[schema_idx].append(effective_weight)

        nonempty = torch.tensor(
            [bool(parts) for parts in effective_by_schema],
            dtype=torch.bool,
            device=h0.device,
        )
        means = torch.stack(
            [
                torch.cat(parts).mean() if parts else h0.new_tensor(0.0)
                for parts in effective_by_schema
            ]
        )
        stds = torch.stack(
            [
                torch.cat(parts).std(unbiased=False)
                if parts
                else h0.new_tensor(0.0)
                for parts in effective_by_schema
            ]
        )
        return tuple(weights), means, stds, nonempty

    def encode(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        closure_edge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        epoch: int = -1,
        discovery: DiscoveryMetadata | None = None,
        diffusion_overrides: dict[str, object] | None = None,
    ) -> EncodedState:
        """Runs the task-agnostic sheaf-diffusion backbone.

        Takes explicit tensors rather than a PreparedH2SData object so it can be
        driven by any task's data container (node classification, link
        prediction, or a PyG Batch of many small graphs for graph classification)
        without those containers needing a shared base shape.

        `diffusion_overrides` is forwarded as kwargs to `HodgeDiffusion.forward`
        (e.g. `{"num_steps": 13, "record_trajectory": True}`) for analysis tooling;
        left as None it reproduces the exact training/eval forward pass.
        """
        if self.node_encoder is not None:
            h0 = self.node_encoder(x, node_type)
        else:
            h0 = self.identity_embedding.weight  # type: ignore[union-attr]
        return self._encode_from_h0(
            h0=h0,
            edge_index=edge_index,
            edge_type=edge_type,
            face_groups=face_groups,
            closure_edge_index=closure_edge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            epoch=epoch,
            discovery=discovery,
            diffusion_overrides=diffusion_overrides,
        )

    def encode_from_stalk(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        closure_edge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        epoch: int = -1,
        discovery: DiscoveryMetadata | None = None,
        diffusion_overrides: dict[str, object] | None = None,
        injected_maps: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> EncodedState:
        """Coarse-level entry point for hierarchical pooling (THS-Pool).

        Identical to `encode()` except `h0` is supplied directly as an already-
        encoded `[N, d]` stalk tensor, skipping the input node-encoder. Coarse-graph
        nodes carry `d`-dimensional pooled stalks, not raw `in_channels` features, so
        they cannot go through `encode()`'s `TypeSpecificNodeEncoder`. See
        docs/sheafcovpool_implementation.md, Appendix A.

        `injected_maps=(F_src, F_tgt)` (THS-Pool V1.5): reuse the surviving fine
        node->edge restriction maps for the coarse edges instead of re-learning them
        (coarsen the sheaf by restriction). Shapes must match `edge_index` ([E, d, d]).
        """
        return self._encode_from_h0(
            h0=h0,
            edge_index=edge_index,
            edge_type=edge_type,
            face_groups=face_groups,
            closure_edge_index=closure_edge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            epoch=epoch,
            discovery=discovery,
            diffusion_overrides=diffusion_overrides,
            injected_maps=injected_maps,
        )

    def _encode_from_h0(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        closure_edge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        epoch: int = -1,
        discovery: DiscoveryMetadata | None = None,
        diffusion_overrides: dict[str, object] | None = None,
        injected_maps: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> EncodedState:
        """Shared backbone body used by both `encode()` (from raw features) and
        `encode_from_stalk()` (from a pooled coarse stalk). Nothing here reads the
        raw `x`/`node_type` — the node-type dependence lives only in the encoder that
        produced `h0`."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_real_edges].")
        if edge_type.dim() != 1 or edge_type.numel() != edge_index.size(1):
            raise ValueError("edge_type must align with the real edge domain.")
        if num_edges != edge_index.size(1):
            raise ValueError("num_edges must equal the real edge_index column count.")
        if (
            closure_edge_index.dim() != 2
            or closure_edge_index.size(0) != 2
        ):
            raise ValueError("closure_edge_index must have shape [2, num_faces].")
        declared_faces = sum(group.num_faces for group in face_groups)
        if closure_edge_index.size(1) != declared_faces:
            raise ValueError(
                "closure_edge_index must contain exactly one edge per face instance."
            )
        active_face_groups = face_groups
        active_closure_edge_index = closure_edge_index
        active_discovery = discovery
        if self.model_config.ablate_no_faces:
            active_face_groups = tuple()
            active_closure_edge_index = closure_edge_index.new_empty((2, 0))
            active_discovery = None

        if self.model_config.ablate_no_delta1:
            # No delta1 means no face maps, and the closure edges' restriction maps are
            # produced *by* the face transport — so the closure 1-cells go away with it.
            active_closure_edge_index = closure_edge_index.new_empty((2, 0))

        relation_embeddings = self.relation_embedding.weight
        num_real_edges = num_edges
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        # Map operators over the (extended) edge set under map_family="factorized".
        edge_maps: EdgeMapPair | None = None
        if injected_maps is not None:
            if self.map_family == "factorized":
                raise NotImplementedError(
                    "injected_maps (THS-Pool map injection) needs dense [E, d, d] maps and is "
                    "not implemented for model.map_family='factorized' (D1 scope)."
                )
            # THS-Pool V1.5: coarsen the sheaf by restriction — reuse the surviving fine
            # node->edge maps rather than re-learning them on the tiny coarse graph. delta0_h0 is
            # recomputed from the injected maps and the current (coarse, gated) h0.
            F_src, F_tgt = injected_maps
            F_src_scale = None
        elif self.map_family == "factorized":
            edge_maps = self.node_to_edge(
                h=h0,
                edge_index=edge_index,
                edge_type=edge_type,
                relation_embeddings=relation_embeddings,
            ).maps
            F_src = None
            F_tgt = None
            F_src_scale = None
        else:
            sheaf_output = self.node_to_edge(
                h=h0,
                edge_index=edge_index,
                edge_type=edge_type,
                relation_embeddings=relation_embeddings,
            )
            F_src = sheaf_output.F_src
            F_tgt = sheaf_output.F_tgt
            F_src_scale = sheaf_output.F_src_scale
        if self.model_config.ablate_identity_maps:
            eye = torch.eye(
                self.model_config.d,
                device=h0.device,
                dtype=h0.dtype,
            ).unsqueeze(0)
            F_src = eye.expand(num_real_edges, -1, -1).contiguous()
            F_tgt = eye.expand(num_real_edges, -1, -1).contiguous()
            # Identity maps are conformal-orthogonal with c = 1, so the transport can still
            # use the transpose-and-divide inverse rather than a general solve.
            F_src_scale = h0.new_ones(num_real_edges)

        face_groups_for_maps = (
            tuple() if self.model_config.ablate_no_delta1 else active_face_groups
        )
        instance_weight_scorer = self._build_instance_weight_scorer(
            active_discovery, epoch
        )
        edge_to_face_output = self.edge_to_face(
            h=h0,
            face_groups=face_groups_for_maps,
            relation_embeddings=relation_embeddings,
            F_src=F_src,
            F_tgt=F_tgt,
            F_src_scale=F_src_scale,
            num_real_edges=num_real_edges,
            closure_scale=self.data_config.closure_scale,
            instance_weight_scorer=instance_weight_scorer,
            factorized_maps=edge_maps,
        )
        face_caches = edge_to_face_output.caches
        cached_faces = sum(cache.face_edge_index.size(0) for cache in face_caches)
        if cached_faces != sum(group.num_faces for group in face_groups_for_maps):
            raise ValueError("Face-map caches do not cover every active face exactly once.")
        if self.model_config.ablate_no_delta1:
            active_discovery = None

        # Closure edges become first-class 1-cells: they extend the edge list, so they enter
        # delta0 (and hence L0 and L1's down term) exactly like any other edge. Their
        # restriction maps are *defined* by the path transport rather than learned.
        num_closure_edges = int(edge_to_face_output.closure_F_src.size(0))
        if active_closure_edge_index.size(1) != num_closure_edges:
            raise ValueError(
                "Closure edge indices and transported restriction maps must have "
                "exactly one aligned entry per materialized face."
            )
        if num_closure_edges > 0:
            if (
                edge_to_face_output.closure_F_tgt.size(0) != num_closure_edges
            ):
                raise ValueError(
                    "Closure edge indices and transported restriction maps must have "
                    "exactly one aligned entry per materialized face."
                )
            if edge_maps is None:
                F_src = torch.cat([F_src, edge_to_face_output.closure_F_src], dim=0)
                F_tgt = torch.cat([F_tgt, edge_to_face_output.closure_F_tgt], dim=0)
            else:
                # Observed edges stay matrix-free; the closure edges keep the per-face dense
                # maps the v1 transport produces (holonomy, identity) until D4 removes them.
                edge_maps = EdgeMapPair(
                    src=StackedMapOperator(
                        (edge_maps.src, DenseMapOperator(edge_to_face_output.closure_F_src))
                    ),
                    tgt=StackedMapOperator(
                        (edge_maps.tgt, DenseMapOperator(edge_to_face_output.closure_F_tgt))
                    ),
                )
            edge_index = torch.cat(
                [edge_index, active_closure_edge_index], dim=1
            )
            edge_type = torch.cat(
                [
                    edge_type,
                    edge_type.new_full((num_closure_edges,), self.closure_relation_id),
                ],
                dim=0,
            )
            src_idx = edge_index[0]
            tgt_idx = edge_index[1]
        num_edges = num_real_edges + num_closure_edges

        # delta0 over the *extended* edge set, so the edge cochain and the chain-condition
        # diagnostic both see the closure 1-cells.
        if edge_maps is None:
            delta0_h0 = CoboundaryOps.delta0(
                x=h0,
                F_src=F_src,
                F_tgt=F_tgt,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
            )
        else:
            delta0_h0 = CoboundaryOps.delta0_matrix_free(
                x=h0, maps=edge_maps, src_idx=src_idx, tgt_idx=tgt_idx
            )
        s0 = self.edge_initializer(
            h=h0,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_embeddings=relation_embeddings,
            delta0_h0=delta0_h0 if self.model_config.use_delta0_init else None,
        )

        (
            face_weights,
            instance_weight_means,
            instance_weight_stds,
            instance_weight_nonempty_mask,
        ) = self._build_face_weights(
            active_face_groups,
            active_discovery,
            h0,
            epoch,
            edge_to_face_output.instance_weights,
        )

        def apply_l0(x_in: torch.Tensor) -> torch.Tensor:
            # Closure edges are inside F_src/F_tgt/src_idx/tgt_idx now, so L0 = delta0^T delta0
            # picks them up mechanically. The old `closure_L0` correction (frozen scale*I maps,
            # applied outside delta0) is gone — it was inconsistent with delta1's domain.
            if edge_maps is not None:
                return CoboundaryOps.L0_matrix_free(
                    x=x_in,
                    maps=edge_maps,
                    src_idx=src_idx,
                    tgt_idx=tgt_idx,
                    num_nodes=num_nodes,
                )
            return CoboundaryOps.L0(
                x=x_in,
                F_src=F_src,
                F_tgt=F_tgt,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                num_nodes=num_nodes,
            )

        def apply_l1(s_in: torch.Tensor) -> torch.Tensor:
            if edge_maps is not None:
                return CoboundaryOps.L1_matrix_free(
                    s=s_in,
                    maps=edge_maps,
                    src_idx=src_idx,
                    tgt_idx=tgt_idx,
                    num_nodes=num_nodes,
                    face_caches=face_caches,
                    map_builder=self.edge_to_face.build_maps,
                    num_edges=num_edges,
                    face_weights=face_weights,
                )
            return CoboundaryOps.L1(
                s=s_in,
                F_src=F_src,
                F_tgt=F_tgt,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                num_nodes=num_nodes,
                face_caches=face_caches,
                map_builder=self.edge_to_face.build_maps,
                num_edges=num_edges,
                face_weights=face_weights,
            )

        x_star, s_star, x_trajectory = self.diffusion(
            x0=h0,
            s0=s0,
            apply_l0=apply_l0,
            apply_l1=apply_l1,
            **(diffusion_overrides or {}),
        )
        x_star = self.node_norm(x_star)
        s_star = self.edge_norm(s_star)
        if edge_maps is None:
            delta0_x_star = CoboundaryOps.delta0(
                x=x_star,
                F_src=F_src,
                F_tgt=F_tgt,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
            )
        else:
            delta0_x_star = CoboundaryOps.delta0_matrix_free(
                x=x_star, maps=edge_maps, src_idx=src_idx, tgt_idx=tgt_idx
            )
        raw_face_curl_result = CoboundaryOps.delta1(
            s=s_star,
            face_caches=face_caches,
            map_builder=self.edge_to_face.build_maps,
        )
        raw_face_curl = raw_face_curl_result.concatenated
        delta1_delta0_h0 = CoboundaryOps.delta1(
            s=delta0_h0,
            face_caches=face_caches,
            map_builder=self.edge_to_face.build_maps,
        ).concatenated

        return EncodedState(
            h0=h0,
            x_star=x_star,
            s_star=s_star,
            delta0_h0=delta0_h0,
            delta0_x_star=delta0_x_star,
            raw_face_curl=raw_face_curl,
            delta1_delta0_h0=delta1_delta0_h0,
            F_src=F_src,
            F_tgt=F_tgt,
            edge_index=edge_index,
            num_real_edges=num_real_edges,
            instance_weight_means=instance_weight_means,
            instance_weight_stds=instance_weight_stds,
            instance_weight_nonempty_mask=instance_weight_nonempty_mask,
            operator_ramp=self._operator_ramp(epoch),
            x_trajectory=x_trajectory,
            edge_maps=edge_maps,
        )

    def forward(
        self, data: PreparedH2SData, epoch: int = -1
    ) -> H2SNNOutput:
        encoded = self.encode(
            x=data.x,
            node_type=data.node_type,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            face_groups=data.face_groups,
            closure_edge_index=data.closure_edge_index,
            num_nodes=data.num_nodes,
            num_edges=data.num_edges,
            epoch=epoch,
            discovery=data.discovery,
        )
        logits = self.readout(
            x_star=encoded.x_star,
            h0=encoded.h0,
            s_star=encoded.s_star,
            # Extended edge list: s_star spans real + closure 1-cells.
            edge_index=encoded.edge_index,
            F_src=encoded.F_src,
            F_tgt=encoded.F_tgt,
            target_indices=data.target_indices,
            edge_maps=encoded.edge_maps,
        )

        return H2SNNOutput(
            logits=logits,
            delta0_x_star=encoded.delta0_x_star,
            raw_face_curl=encoded.raw_face_curl,
            delta1_delta0_h0=encoded.delta1_delta0_h0,
            delta0_h0=encoded.delta0_h0,
            x_star=encoded.x_star,
            s_star=encoded.s_star,
            instance_weight_means=encoded.instance_weight_means,
            instance_weight_stds=encoded.instance_weight_stds,
            instance_weight_nonempty_mask=encoded.instance_weight_nonempty_mask,
            operator_ramp=encoded.operator_ramp,
        )


def build_model(config: TrainConfig, data: PreparedH2SData) -> "H2SNN | H2SNNv2":
    """Node-classification model selected by `model.architecture` (v1 default)."""
    if config.model.architecture == "v2":
        return H2SNNv2(config=config, dims=ModelDims.from_prepared_data(data))
    return H2SNN(config=config, dims=ModelDims.from_prepared_data(data))


def build_link_prediction_model(config: TrainConfig, data: PreparedLinkPredData) -> H2SNN:
    return H2SNN(config=config, dims=ModelDims.from_link_pred_data(data))


def build_graph_classification_model(config: TrainConfig, dims: ModelDims) -> H2SNN:
    return H2SNN(config=config, dims=dims)
