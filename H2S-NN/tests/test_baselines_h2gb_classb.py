"""Card baselines-h2gb C4 (slim mode): class-B adapters on H2GB's graph, synthetic graphs only.

* SeHGNN: the stored features of every target (their `hg_propagate` on our typed graph, parallel relations merged)
  equal an independent float64 computation, key by key: for a key `c0 c1 ... cn` (one letter per type, target first)
  the value is M[c0<-c1] ... M[c(n-1)<-cn] X[cn], M the row-mean of the merged relation (atol 1e-10, fixed in advance);
  and batching target rows is exact against the full forward in eval mode.
* One training step is finite and moves the parameters: SeHGNN, ChebNetII.
The real data root is unreachable.
"""

from __future__ import annotations

import unittest

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ATOL = 1e-10


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def typed_graph(seed: int = 0, width: int | None = None):
    """paper (target, labels 0..2, masks), author, field (featureless unless `width`); paper-field twice (parallel
    relations), author-paper, paper-paper; reverses stored; float64 features."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(seed)
    n = {"paper": 30, "author": 12, "field": 6}
    data = HeteroData()
    data["paper"].x = torch.randn(n["paper"], width or 7, generator=g, dtype=torch.float64)
    data["author"].x = torch.randn(n["author"], width or 4, generator=g, dtype=torch.float64)
    if width is not None:
        data["field"].x = torch.randn(n["field"], width, generator=g, dtype=torch.float64)
    else:
        data["field"].num_nodes = n["field"]
    data["paper"].y = torch.randint(0, 3, (n["paper"],), generator=g)
    order = torch.randperm(n["paper"], generator=g)
    for name, part in (("train", order[:16]), ("val", order[16:23]), ("test", order[23:])):
        mask = torch.zeros(n["paper"], dtype=torch.bool)
        mask[part] = True
        data["paper"][f"{name}_mask"] = mask

    def rel(src, dst, count):
        return torch.unique(torch.stack([torch.randint(0, n[src], (count,), generator=g),
                                         torch.randint(0, n[dst], (count,), generator=g)]), dim=1)

    pf1, pf2, ap, pp = rel("paper", "field", 25), rel("paper", "field", 20), rel("author", "paper", 40), \
        rel("paper", "paper", 30)
    data["paper", "in_l0", "field"].edge_index = pf1
    data["paper", "in_l1", "field"].edge_index = pf2
    data["field", "rev_in_l0", "paper"].edge_index = pf1.flip(0)
    data["field", "rev_in_l1", "paper"].edge_index = pf2.flip(0)
    data["author", "writes", "paper"].edge_index = ap
    data["paper", "rev_writes", "author"].edge_index = ap.flip(0)
    data["paper", "cites", "paper"].edge_index = pp
    return data


def _row_mean(edge_index: torch.Tensor, num_dst: int, num_src: int) -> torch.Tensor:
    dense = torch.zeros(num_dst, num_src, dtype=torch.float64)
    dense[edge_index[1], edge_index[0]] = 1.0
    degree = dense.sum(1, keepdim=True)
    return torch.where(degree > 0, dense / degree.clamp(min=1), dense)


class SeHGNNPrecomputationTests(unittest.TestCase):
    def test_stored_features_equal_independent_meta_path_means(self) -> None:
        from src.baselines.adapters import _sehgnn_h2gb as h2gb

        data = typed_graph()
        letters = h2gb.type_letters(list(data.node_types))
        by_letter = {v: k for k, v in letters.items()}
        inputs = h2gb.node_inputs(data, 5, seed=0)
        inputs = {k: v.double() for k, v in inputs.items()}
        feats = h2gb.propagate(data, "paper", 2, inputs)
        merged = h2gb.merged_relations(data, letters)
        mean = {(dst, src): _row_mean(torch.stack(edges), int(data[by_letter[dst]].num_nodes),
                                      int(data[by_letter[src]].num_nodes))
                for (src, _, dst), edges in merged.items()}
        self.assertIn("P", feats)
        self.assertTrue(any(len(k) == 3 for k in feats), sorted(feats))
        for key, value in feats.items():
            expected = inputs[by_letter[key[-1]]]
            for i in range(len(key) - 1, 0, -1):
                expected = mean[(key[i - 1], key[i])] @ expected
            with self.subTest(key=key):
                self.assertLessEqual((value.double() - expected).abs().max().item(), ATOL)
        # the parallel paper-field relations are one merged relation (union of both)
        pf = merged[("P", "P-F", "F")]
        union = torch.unique(torch.cat([data["paper", "in_l0", "field"].edge_index,
                                        data["paper", "in_l1", "field"].edge_index], 1), dim=1)
        self.assertTrue(torch.equal(torch.stack(pf), union))

    def test_batching_target_rows_is_exact_in_eval(self) -> None:
        from src.baselines.adapters import _sehgnn_h2gb as h2gb

        data = typed_graph()
        feats = h2gb.propagate(data, "paper", 2, {k: v.double() for k, v in h2gb.node_inputs(data, 5, 0).items()})
        torch.manual_seed(0)
        model = h2gb.build_model(feats, "P", 3, {"embed_size": 8, "hidden": 16}).double().eval()
        n = int(data["paper"].num_nodes)
        zeros = torch.zeros(n, 3, dtype=torch.float64)
        with torch.no_grad():
            full = model({k: v.clone() for k, v in feats.items()}, {}, zeros)
            for idx in (torch.arange(0, 7), torch.tensor([3, 29, 11])):
                part = model({k: v[idx] for k, v in feats.items()}, {}, zeros[idx])
                self.assertLessEqual((part - full[idx]).abs().max().item(), ATOL)


def _one_step_moves(test: unittest.TestCase, model: torch.nn.Module, forward, labels, criterion) -> None:
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    optimizer.zero_grad()
    loss = criterion(forward(), labels)
    test.assertTrue(torch.isfinite(loss).item())
    loss.backward()
    optimizer.step()
    test.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, model.parameters())), "no parameter moved")


class OneStepTests(unittest.TestCase):
    def test_sehgnn_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters import _sehgnn_h2gb as h2gb

        data = typed_graph()
        feats = h2gb.propagate(data, "paper", 2, {k: v.float() for k, v in h2gb.node_inputs(data, 5, 0).items()})
        torch.manual_seed(0)
        model = h2gb.build_model(feats, "P", 3, {"embed_size": 8, "hidden": 16})
        idx = data["paper"].train_mask.nonzero().view(-1)
        _one_step_moves(self, model, lambda: model({k: v[idx] for k, v in feats.items()}, {},
                                                   torch.zeros(idx.numel(), 3)),
                        data["paper"].y[idx], torch.nn.functional.cross_entropy)

    def test_chebnetii_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters import _chebnetii_h2gb as h2gb

        data = typed_graph(width=6)
        x, edge_index, slices = h2gb.homogeneous_inputs(data)
        start, end = slices["paper"]
        bases = h2gb.chebyshev_bases(x.float(), edge_index, 3, torch.arange(start, end))
        torch.manual_seed(0)
        model = h2gb.build_model(6, 3, {"K": 3, "hidden": 16})
        idx = data["paper"].train_mask.nonzero().view(-1)
        _one_step_moves(self, model, lambda: model([b[idx] for b in bases], 0, idx.numel()), data["paper"].y[idx],
                        torch.nn.functional.nll_loss)
        with self.assertRaises(ValueError):  # featureless type: one feature matrix does not exist
            h2gb.homogeneous_inputs(typed_graph())


class ChebNetIIProjectionTests(unittest.TestCase):
    """Part 2 E2, decision C4-c: per-type projection after per-block bases equals projecting first (float64)."""

    def _graph(self):
        data = typed_graph(width=6)
        data["author"].x = torch.randn(int(data["author"].num_nodes), 3, dtype=torch.float64)  # another width
        return data

    def test_block_bases_then_projection_equals_projection_then_propagation(self) -> None:
        from src.baselines.adapters import _chebnetii_h2gb as h2gb

        data = self._graph()
        features, edge_index, slices = h2gb.block_inputs(data)
        start, end = slices["paper"]
        rows = torch.arange(start, end)
        torch.manual_seed(0)
        projected = h2gb.ProjectedBases([int(f.size(1)) for f in features], 5).double()
        blocks = h2gb.block_bases(features, edge_index, slices, 4, rows)
        with torch.no_grad():
            got = projected(blocks)
            x_proj = projected.projection(features)               # project first (HGB adapter order)
            expected = h2gb.chebyshev_bases(x_proj, edge_index, 4, rows)
        self.assertEqual(len(got), len(expected))
        for k, (a, b) in enumerate(zip(got, expected)):
            self.assertLessEqual((a - b).abs().max().item(), ATOL, f"T_{k}")
        with self.assertRaises(ValueError):  # a featureless type is refused (stop item)
            h2gb.block_inputs(typed_graph())

    def test_projected_chebnetii_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters import _chebnetii_h2gb as h2gb

        data = self._graph()
        features, edge_index, slices = h2gb.block_inputs(data)
        start, end = slices["paper"]
        blocks = h2gb.block_bases([f.float() for f in features], edge_index, slices, 3, torch.arange(start, end))
        torch.manual_seed(0)
        model = torch.nn.ModuleDict({"projection": h2gb.ProjectedBases([int(f.size(1)) for f in features], 8),
                                     "chebnet": h2gb.build_model(8, 3, {"K": 3, "hidden": 16})})
        idx = data["paper"].train_mask.nonzero().view(-1)
        _one_step_moves(self, model, lambda: model["chebnet"](
            model["projection"]([[b[idx] for b in per_type] for per_type in blocks]), 0, idx.numel()),
            data["paper"].y[idx], torch.nn.functional.nll_loss)


class HINormerTests(unittest.TestCase):
    """Card 1c F1: the original HINormer on target batches; no label input (its forward reads features, sequences,
    type embeddings and node types, HINormer/model.py:198-217), so no route test."""

    def test_hinormer_step_is_finite_and_moves(self) -> None:
        import dgl

        from src.baselines.adapters import hinormer
        from src.baselines.runtime._dgl_compat import restore_legacy_message_functions

        g = torch.Generator().manual_seed(0)
        sizes, dims = (12, 7), (4, 3)
        n = sum(sizes)
        raw = torch.randint(0, n, (2, 40), generator=g)
        raw = raw[:, raw[0] != raw[1]]
        both = torch.unique(torch.cat([raw, raw.flip(0)], dim=1), dim=1)
        restore_legacy_message_functions()
        graph = dgl.graph((both[0], both[1]), num_nodes=n)
        features = [torch.randn(sizes[0], dims[0], generator=g), torch.randn(sizes[1], dims[1], generator=g)]
        node_type = torch.cat([torch.zeros(sizes[0]), torch.ones(sizes[1])]).long()
        node_seq = hinormer.build_node_sequences_for(graph, torch.arange(sizes[0]), 6)
        self.assertEqual(tuple(node_seq.shape), (sizes[0], 6))
        self.assertTrue(torch.equal(node_seq[:, 0], torch.arange(sizes[0])))
        torch.manual_seed(0)
        model = hinormer.build_model(graph, 3, list(dims), 2, {"hidden_dim": 8, "num_layers": 1, "num_gnns": 2,
                                                                "num_heads": 2, "dropout": 0.0})
        idx = torch.arange(5)
        labels = torch.randint(0, 3, (sizes[0],), generator=g)
        _one_step_moves(self, model, lambda: model(features, node_seq[idx], torch.eye(2), node_type, True),
                        labels[idx], torch.nn.functional.cross_entropy)


class PSHGCNWidthTests(unittest.TestCase):
    """Card 1c F2 (part 2 E2's PSHGCN equivalence): with MAG's widths (P 128, others 256) the adapter's width-built
    model and their `PSHGCN` class, same state dict, give the same output on a synthetic MAG-typed input (float64)."""

    def test_width_built_model_equals_their_class_on_mag_widths(self) -> None:
        from src.baselines.adapters import _pshgcn_mag
        from src.baselines.runtime import pushd_sys_path

        keys = ["P", "PA", "PP", "PF", "AP", "API"]
        expander = ["10P", "1011PA", "1012PP", "1013PF"]  # coefficient indices 10.. as their expander writes them
        label_keys = ["PP", "PAP"]
        args = dict(emb_dim=16, hidden_x=8, hidden_l=8, nclass=3, label_keys=label_keys, layers_x=2, layers_l=2,
                    coe_num=4, expander=expander, dropout=0.0, input_drop=0.0)
        widths = {"P": 128, "A": 256, "I": 256, "F": 256}
        torch.manual_seed(0)
        ours = _pshgcn_mag.build_model(widths, **args).double().eval()
        with pushd_sys_path("pshgcn_mag"):
            from model import PSHGCN  # type: ignore[import-not-found]

            theirs = PSHGCN(16, 8, 8, 3, list(widths), label_keys, 2, 2, 4, expander, 0.0, 0.0, False).double().eval()
        theirs.load_state_dict(ours.state_dict())
        g = torch.Generator().manual_seed(1)
        feats = {k: torch.randn(5, widths[k[-1]], generator=g, dtype=torch.float64) for k in keys}
        label_feats = {k: torch.rand(5, 3, generator=g, dtype=torch.float64) for k in label_keys}
        with torch.no_grad():
            a = ours({k: v.clone() for k, v in feats.items()}, dict(label_feats))
            b = theirs({k: v.clone() for k, v in feats.items()}, dict(label_feats))
        self.assertTrue(torch.equal(a, b))
        # and with H2GB's MAG widths (128 everywhere) the widths are the dataset's
        h2gb = _pshgcn_mag.build_model({"P": 128, "A": 128, "I": 128, "F": 128}, **args)
        self.assertEqual(h2gb.embedings_x["A"].lins[0].in_features, 128)


class SeHGNNMAGInputTests(unittest.TestCase):
    """Card 1c F3 (G-4): their MAG branch's random inputs replace the given features of the listed types; one step."""

    def test_mag_random_inputs_and_step(self) -> None:
        from src.baselines.adapters import _sehgnn_h2gb as h2gb

        data = typed_graph(width=6)
        inputs = h2gb.node_inputs(data, 5, 0, frozenset({"author", "field"}))
        self.assertTrue(torch.equal(inputs["paper"], data["paper"].x))
        for node_type in ("author", "field"):
            self.assertEqual(tuple(inputs[node_type].shape), (int(data[node_type].num_nodes), 5))
            self.assertTrue(bool((inputs[node_type].abs() <= 0.5).all()))
        feats = h2gb.propagate(data, "paper", 2, {k: v.float() for k, v in inputs.items()})
        torch.manual_seed(0)
        model = h2gb.build_model(feats, "P", 3, {"embed_size": 8, "hidden": 16})
        idx = data["paper"].train_mask.nonzero().view(-1)
        _one_step_moves(self, model, lambda: model({k: v[idx] for k, v in feats.items()}, {},
                                                   torch.zeros(idx.numel(), 3)),
                        data["paper"].y[idx], torch.nn.functional.cross_entropy)


class Hetero2NetTests(unittest.TestCase):
    """Card 1c F4: their NeighborLoader scheme with the label channel `y_emb` as main.py:125-175 builds it."""

    def _outputs(self, labels: torch.Tensor):
        from copy import copy

        from src.baselines.adapters import _hetero2net_h2gb as h2gb

        data = typed_graph(width=6)
        for node_type in data.node_types:
            data[node_type].x = data[node_type].x.float()
        data["paper"].y = labels.clone()
        train_y, test_y = h2gb.label_channel(data, "paper", 3, data["paper"].val_mask, data["paper"].test_mask)
        train_data, test_data = copy(data), copy(data)
        train_data["paper"].y_emb, test_data["paper"].y_emb = train_y, test_y
        _, val_loader, test_loader = h2gb.loaders(train_data, test_data, "paper", 4)
        torch.manual_seed(0)
        model = h2gb.build_model(data, "paper", 3, {"hidden": 8, "dropout": 0.0}).eval()
        out = {}
        with torch.no_grad():
            for name, loader in (("val", val_loader), ("test", test_loader)):
                torch.manual_seed(1)
                out[name] = torch.cat([h2gb.seed_logits(model, batch, "paper") for batch in loader])
        return data, out

    def test_label_channel_reads_no_held_out_label(self) -> None:
        data, base = self._outputs(typed_graph(width=6)["paper"].y)
        labels = data["paper"].y.clone()
        val, test = data["paper"].val_mask, data["paper"].test_mask
        both = labels.clone()
        both[val | test] = (both[val | test] + 1) % 3   # every validation and test label changed
        _, changed = self._outputs(both)
        self.assertTrue(torch.equal(base["val"], changed["val"]), "validation/test labels reach validation targets")
        only_test = labels.clone()
        only_test[test] = (only_test[test] + 1) % 3
        _, changed = self._outputs(only_test)
        self.assertTrue(torch.equal(base["test"], changed["test"]), "test labels reach test targets")

    def test_one_training_step_is_finite_and_moves(self) -> None:
        from copy import copy

        from src.baselines.adapters import _hetero2net_h2gb as h2gb
        from src.baselines.runtime import pushd_sys_path

        data = typed_graph(width=6)
        for node_type in data.node_types:
            data[node_type].x = data[node_type].x.float()
        self.assertIsNone(h2gb.prepare_inputs(data, "pdns", None))  # FreeBase branch: no metapath list
        train_y, test_y = h2gb.label_channel(data, "paper", 3, data["paper"].val_mask, data["paper"].test_mask)
        train_data, test_data = copy(data), copy(data)
        train_data["paper"].y_emb, test_data["paper"].y_emb = train_y, test_y
        train_loader, _, _ = h2gb.loaders(train_data, test_data, "paper", 4)
        torch.manual_seed(0)
        model = h2gb.build_model(data, "paper", 3, {"hidden": 8, "dropout": 0.0})
        with torch.no_grad():  # their HeteroGNN has lazy modules: one forward gives most of them their shapes
            h2gb.seed_logits(model, next(iter(train_loader)), "paper")
        from torch.nn.parameter import UninitializedParameter

        before = {name: p.detach().clone() for name, p in model.named_parameters()
                  if not isinstance(p, UninitializedParameter)}
        self.assertTrue(before)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        with pushd_sys_path("hetero2net"):
            from hetero2net.utils import train_mini_batch  # type: ignore[import-not-found]

            loss = train_mini_batch(model, optimizer, h2gb.epoch_batches(train_loader, {}, 1), torch.device("cpu"),
                                    p=0.7, metapaths=None, r=0.5, alpha=0.2, beta=0.4)
        self.assertTrue(torch.isfinite(torch.tensor(float(loss))).item())
        after = dict(model.named_parameters())
        self.assertTrue(any(not torch.equal(value, after[name]) for name, value in before.items()))


class ChebNetIIForwardTests(unittest.TestCase):
    """Card 1c F5 (G-6): projection first, propagation in the forward, on a graph with an identity (featureless) block;
    equal to the per-block precomputation by linearity (`ChebNetIIProjectionTests`)."""

    def test_forward_bases_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters import _chebnetii_h2gb as h2gb

        data = typed_graph()  # `field` has no features: a sparse identity block
        for node_type in ("paper", "author"):
            data[node_type].x = data[node_type].x.float()
        features, edge_index, slices = h2gb.forward_inputs(data)
        self.assertTrue(features[list(data.node_types).index("field")].is_sparse)
        n = sum(int(f.size(0)) for f in features)
        start, _ = slices["paper"]
        torch.manual_seed(0)
        model = torch.nn.ModuleDict({"bases": h2gb.ForwardBases([int(f.size(1)) for f in features], 8, edge_index, n,
                                                                3),
                                     "chebnet": h2gb.build_model(8, 3, {"K": 3, "hidden": 16})})
        idx = data["paper"].train_mask.nonzero().view(-1)
        _one_step_moves(self, model, lambda: model["chebnet"](model["bases"](features, idx + start), 0, idx.numel()),
                        data["paper"].y[idx], torch.nn.functional.nll_loss)


if __name__ == "__main__":
    unittest.main()
