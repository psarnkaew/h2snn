from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TypeSpecificNodeEncoder(nn.Module):
    """Project each node type into the shared stalk space with its own linear map.

    Follows the HGB feature-preprocessing convention (Lv et al., §4.3; also HetSheaf
    §3.1): one `nn.Linear(d_t, d)` with bias per node type, sized to that type's own
    raw feature width, and a learned embedding for node types that carry no attributes
    at all. HGB expresses the latter as `sp.eye(n_t)` followed by a linear layer; that
    composition is exactly a free vector per node, so an `nn.Embedding` is the same
    function without materialising an `n_t x n_t` identity.

    `type_input_dims` is optional. When it is omitted every type is built at the shared
    `input_dim`, which is the correct behaviour for datasets whose types all share one
    feature width (e.g. the TUDataset graph-classification benchmarks).

    `input_dropout` (card 23 C1, `model.input_dropout`): dropout on each featured type's raw
    features before its `Linear`; featureless types are untouched. At 0 (the default) the call
    is skipped, so no RNG is consumed and the forward is that of `b0f13fb`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_node_types: int,
        dropout: float,
        type_input_dims: tuple[int, ...] | None = None,
        type_num_nodes: tuple[int, ...] | None = None,
        type_offsets: tuple[int, ...] | None = None,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(input_dropout) < 1.0:
            raise ValueError(f"input_dropout must be in [0, 1), got {input_dropout}.")
        self.input_dropout = float(input_dropout)
        if type_input_dims is None:
            type_input_dims = (input_dim,) * num_node_types
        if len(type_input_dims) != num_node_types:
            raise ValueError(
                f"type_input_dims has {len(type_input_dims)} entries but there are "
                f"{num_node_types} node types."
            )
        if any(dim == 0 for dim in type_input_dims):
            if type_num_nodes is None or type_offsets is None:
                raise ValueError(
                    "type_num_nodes and type_offsets are required when a node type has "
                    "no raw features, so it can be given a learned embedding instead."
                )
            if len(type_num_nodes) != num_node_types or len(type_offsets) != num_node_types:
                raise ValueError(
                    "type_num_nodes and type_offsets must have one entry per node type."
                )

        self.type_input_dims = tuple(int(dim) for dim in type_input_dims)
        # Only consulted for the attribute-less branch; zeros keep the tuples aligned.
        self.type_offsets = tuple(int(offset) for offset in (type_offsets or (0,) * num_node_types))

        # ModuleList entries are positional, so every type occupies a slot in both lists
        # and exactly one of the two is populated. `None` marks the branch not taken.
        self.encoders = nn.ModuleList()
        self.identity_embeddings = nn.ModuleList()
        for node_type_id in range(num_node_types):
            type_dim = self.type_input_dims[node_type_id]
            if type_dim > 0:
                self.encoders.append(
                    nn.Sequential(
                        nn.Linear(type_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                    )
                )
                self.identity_embeddings.append(nn.Identity())
            else:
                num_nodes = int((type_num_nodes or ())[node_type_id])
                self.encoders.append(nn.Identity())
                # LayerNorm here as well, so every type's stalk reaches the restriction
                # maps on the same scale. Without it a free embedding could drift away
                # from the normalised featured types during training.
                self.identity_embeddings.append(
                    nn.Sequential(
                        nn.Embedding(num_nodes, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                    )
                )

        self.type_embeddings = nn.Embedding(num_node_types, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, node_type: torch.Tensor,
                global_node_id: torch.Tensor | None = None) -> torch.Tensor:
        """`global_node_id` (Part MB, MB1): each row's node id in the whole graph, for a batch of
        scopes; featureless types then read the embedding row of that id. None: rows are the
        node ids (full batch)."""
        hidden = torch.zeros(
            (x.size(0), self.type_embeddings.embedding_dim),
            device=x.device,
            dtype=x.dtype,
        )
        for node_type_id, type_dim in enumerate(self.type_input_dims):
            mask = node_type == node_type_id
            if not mask.any():
                continue
            if type_dim > 0:
                # Feature matrices are zero-padded on the right to a shared width, so a
                # type's real attributes are always the prefix columns [0, d_t).
                features = x[mask][:, :type_dim]
                if self.input_dropout > 0.0:  # card 23 C1; skipped at 0 (no RNG drawn)
                    features = F.dropout(features, self.input_dropout, training=self.training)
                encoded = self.encoders[node_type_id](features)
            else:
                # Nodes of a type occupy one contiguous block of the flattened id space
                # (see stack_node_features), so subtracting the block start gives the
                # embedding row for each node.
                rows = mask.nonzero(as_tuple=False).view(-1)
                ids = rows if global_node_id is None else global_node_id[rows]
                local_index = ids - self.type_offsets[node_type_id]
                encoded = self.identity_embeddings[node_type_id](local_index).to(x.dtype)
            encoded = encoded + self.type_embeddings.weight[node_type_id]
            hidden[mask] = self.activation(encoded)
        return self.dropout(hidden)
