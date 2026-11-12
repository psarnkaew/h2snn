"""Relation adjacencies and polynomial term lists for the spectral baselines on LP.

PSHGCN and H2SGNN both consume three things: a dict of row-normalised relation
adjacencies over the flattened node space, and the two term lists ``lis``/``lis_t`` that
enumerate the polynomial's monomials and their transposes.

PSHGCN ships its own link-prediction preprocessing (``PSHGCN/hgb-lp/processing.py``), but
it hard-codes LastFM and Amazon by name and reads HGB's raw directories directly. The same
structures are rebuilt here from the canonical link-prediction bundle so YouTube and PubMed
are covered too, following that file's construction exactly:

* **One key per relation, not per node-type pair.** Their ``adjs`` dict holds a separate
  operator for every relation. On Amazon -- one node type, two product-product relations --
  they emit two keys, ``PP`` and ``UU`` (hgb-lp/processing.py:86-87); the letters are
  arbitrary labels reused from the LastFM branch, not node types. Parallel relations are
  therefore *never* pooled.
* **Same-type relations are symmetrised into one key and given self-loops**; cross-type
  relations contribute two keys, forward and reverse, without self-loops
  (hgb-lp/processing.py:41-64).
* **Row-normalise** each resulting operator, ``D^-1 A`` (hgb-lp/processing.py:67-74).
* ``lis[1]`` is every key; ``lis[k]`` extends each ``lis[k-1]`` entry by another key.
  The type-compatibility test ``previous[-1] == key[0]`` is applied **only on graphs with
  more than one node type** -- their code branches on exactly this and drops the test for
  Amazon (hgb-lp/processing.py:88-111), because with a single node type every composition
  is valid and the key characters no longer denote types.
* ``lis_t`` reverses each monomial two characters at a time (hgb-lp/processing.py:113-122).

Only the *training* message-passing graph is used, so no validation or test edge reaches
the propagation.

The model indexes monomials by 2-character slices (``adjs[j[:2]]``,
``PSHGCN/hgb/model.py:70``), so every key here is exactly two characters.

Key sets this produces, matching the authors' where they published one:

===========  ==========  =============================  =========
dataset      node types  keys                           chaining
===========  ==========  =============================  =========
LastFM       3           UA AU UU AT TA (theirs)        enforced
Amazon       1           2, one per relation (theirs)   dropped
YouTube      1           5, one per relation            dropped
PubMed       4           16, type-pair encoded          enforced
===========  ==========  =============================  =========
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.bundles import LinkBundle


def _type_pair_key(source_type: int, destination_type: int) -> str:
    """Two-character key encoding the relation's source and destination node types.

    Mirrors PSHGCN's LastFM keys ('UA', 'AU', 'UU', 'AT', 'TA'): the first character is
    the source type and the second the destination, so the chaining rule
    ``previous[-1] == key[0]`` reads as "the previous monomial ends where this relation
    starts".
    """

    return f"{chr(ord('A') + source_type)}{chr(ord('A') + destination_type)}"


def _relation_label(index: int) -> str:
    """Two-character label for one relation on a single-node-type graph.

    Their Amazon branch names the two product-product relations 'PP' and 'UU' -- labels
    carried over from LastFM that identify the relation, not any node type.

    The doubled letter is load-bearing, not decoration. ``lis_t`` transposes a monomial by
    reversing it two characters at a time, so a key must reverse to the key of its own
    transpose. Every operator here is a same-type relation symmetrised as ``A + A^T`` with
    self-loops added, hence its own transpose -- and only a palindromic key satisfies that.
    A label like 'RA' would reverse to 'AR' and raise ``KeyError`` in their forward.
    """

    if index >= 26:
        raise ValueError(
            f"relation index {index} exceeds the 2-character key space the polynomial "
            "models require (adjs[j[:2]] in PSHGCN/hgb/model.py:70)"
        )
    letter = chr(ord("A") + index)
    return letter + letter


def _stored_reverse_relations(data: Any) -> frozenset[int]:
    """Relations whose (message-graph) edge set equals the transpose of an earlier relation's,
    with the types swapped accordingly. A non-empty relation only; the earlier one is kept."""

    pairs: dict[int, frozenset[tuple[int, int]]] = {}
    for relation_id in range(len(data.relation_tuples)):
        mask = data.edge_type == relation_id
        rows, columns = data.edge_index[:, mask].cpu().tolist()
        pairs[relation_id] = frozenset(zip(rows, columns))
    reverses: set[int] = set()
    for later, (source, _, destination) in enumerate(data.relation_tuples):
        if not pairs[later]:
            continue
        transposed = frozenset((v, u) for u, v in pairs[later])
        for earlier in range(later):
            earlier_source, _, earlier_destination = data.relation_tuples[earlier]
            if (earlier not in reverses and (earlier_source, earlier_destination)
                    == (destination, source) and pairs[earlier] == transposed):
                reverses.add(later)
                break
    return frozenset(reverses)


def _plan_relation_keys(
    bundle: LinkBundle,
) -> tuple[dict[str, list[int]], dict[str, bool], bool]:
    """Decide one key per relation direction, and whether chaining applies.

    Returns ``(key -> relation ids, key -> add self-loops, enforce_type_chaining)``.
    """

    data = bundle.data
    type_index = {name: index for index, name in enumerate(data.node_type_names)}
    num_node_types = len(type_index)
    enforce_chaining = num_node_types > 1
    # Part P (P2): releases derived from HGB node-classification data store each relation's
    # reverse as its own relation (ACM paper->author rel_2 and author->paper rel_3, cite and
    # ref). A relation whose edge set is exactly the transpose of an earlier relation adds no
    # operator: its partner's reverse key (cross-type) or symmetrisation (same type) is it.
    stored_reverses = _stored_reverse_relations(data) if enforce_chaining else frozenset()

    if enforce_chaining:
        # Type-encoded keys only work if a (source, destination) pair names exactly one
        # relation (after dropping stored reverses). True for LastFM 3/3, PubMed 10/10 and the
        # derived tasks; a violation would silently merge distinct relations, so refuse.
        seen: dict[tuple[int, int], int] = {}
        for relation_id, (source, _, destination) in enumerate(data.relation_tuples):
            if relation_id in stored_reverses:
                continue
            pair = (type_index[source], type_index[destination])
            canonical = tuple(sorted(pair))
            if canonical in seen:
                raise NotImplementedError(
                    f"{bundle.dataset} has several relations between node types "
                    f"{canonical}; PSHGCN's type-encoded keys cannot name them apart, "
                    "and its single-node-type fallback does not apply here"
                )
            seen[canonical] = relation_id

    keys: dict[str, list[int]] = {}
    self_loops: dict[str, bool] = {}
    for relation_id, (source, _, destination) in enumerate(data.relation_tuples):
        if relation_id in stored_reverses:
            continue
        source_type = type_index[source]
        destination_type = type_index[destination]
        same_type = source_type == destination_type

        if not enforce_chaining:
            # Single node type: every relation is its own operator under an arbitrary
            # label, symmetrised with self-loops as their Amazon branch does.
            keys[_relation_label(relation_id)] = [relation_id]
            self_loops[_relation_label(relation_id)] = True
            continue

        if same_type:
            # A + A^T coalesced into one key, then set_diag().
            key = _type_pair_key(source_type, destination_type)
            keys[key] = [relation_id]
            self_loops[key] = True
        else:
            # Forward and reverse are separate operators, no self-loops.
            forward = _type_pair_key(source_type, destination_type)
            reverse = _type_pair_key(destination_type, source_type)
            keys[forward] = [relation_id]
            keys[reverse] = [-(relation_id + 1)]  # negative marks "use the transpose"
            self_loops[forward] = False
            self_loops[reverse] = False

    return keys, self_loops, enforce_chaining


def build_relation_adjacencies(
    bundle: LinkBundle, device: torch.device
) -> tuple[dict[str, Any], bool]:
    """Row-normalised ``SparseTensor`` per relation key, and whether chaining applies."""

    from torch_sparse import SparseTensor, mul
    from torch_sparse import sum as sparse_sum

    data = bundle.data
    num_nodes = int(data.x.size(0))
    plan, self_loops, enforce_chaining = _plan_relation_keys(bundle)

    adjacencies: dict[str, Any] = {}
    for key, signed_ids in plan.items():
        rows: list[torch.Tensor] = []
        columns: list[torch.Tensor] = []
        for signed in signed_ids:
            relation_id = signed if signed >= 0 else -signed - 1
            transpose = signed < 0
            mask = data.edge_type == relation_id
            row, column = data.edge_index[:, mask].cpu()
            if transpose:
                row, column = column, row
            rows.append(row)
            columns.append(column)
            if self_loops[key]:
                # Same-type relations are symmetrised: A + A^T, as their
                # torch.cat((row0, row1)) over the forward and reverse entries does.
                rows.append(column)
                columns.append(row)

        adjacency = SparseTensor(
            row=torch.cat(rows),
            col=torch.cat(columns),
            sparse_sizes=(num_nodes, num_nodes),
        ).coalesce()
        if self_loops[key]:
            adjacency = adjacency.set_diag()

        adjacency = adjacency.fill_value(1.0)
        degree = sparse_sum(adjacency, dim=1)
        inverse = degree.pow_(-1.0)
        inverse.masked_fill_(inverse == float("inf"), 0.0)
        adjacencies[key] = mul(adjacency, inverse.view(-1, 1)).to(device)

    return adjacencies, enforce_chaining


def build_term_lists(
    adjacencies: dict[str, Any],
    order: int,
    *,
    enforce_type_chaining: bool,
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """The ``lis``/``lis_t`` monomial lists, as their hgb-lp/processing.py builds them."""

    lis: dict[int, list[str]] = {}
    for degree in range(1, order + 1):
        lis[degree] = []
        if degree == 1:
            lis[degree] = list(adjacencies.keys())
            continue
        for previous in lis[degree - 1]:
            for key in adjacencies:
                # On multi-type graphs the key characters denote node types, so a
                # monomial may only extend where the types meet. On a single-type graph
                # they are bare relation labels and every composition is valid -- which
                # is why their Amazon branch drops this test.
                if enforce_type_chaining and previous[-1] != key[0]:
                    continue
                lis[degree].append(previous + key)

    lis_t: dict[int, list[str]] = {}
    for degree, monomials in lis.items():
        lis_t[degree] = []
        for monomial in monomials:
            reversed_monomial = monomial[-2:][::-1]
            remaining = monomial[:-2]
            while remaining:
                reversed_monomial += remaining[-2:][::-1]
                remaining = remaining[:-2]
            lis_t[degree].append(reversed_monomial)

    # Every transposed monomial must decompose into keys the model can look up: their
    # forward does adjs[j[:2]] with no fallback, so a key whose reversal is not itself a
    # key raises KeyError mid-training. This is the invariant that makes single-node-type
    # labels palindromic; assert it rather than discover it on the third dataset.
    missing = {
        monomial[position : position + 2]
        for monomials in lis_t.values()
        for monomial in monomials
        for position in range(0, len(monomial), 2)
    } - set(adjacencies)
    if missing:
        raise AssertionError(
            f"transposed monomials reference unknown relation keys {sorted(missing)}; "
            f"known keys are {sorted(adjacencies)}"
        )
    return lis, lis_t


def polynomial_term_count(lis: dict[int, list[str]]) -> int:
    """Total monomials, including the degree-zero identity term.

    Matches ``coe_num`` in PSHGCN's ``init_coe`` (hgb-lp/pshgcn.py:58-62).
    """

    return 1 + sum(len(values) for values in lis.values())


__all__ = [
    "build_relation_adjacencies",
    "build_term_lists",
    "polynomial_term_count",
]
