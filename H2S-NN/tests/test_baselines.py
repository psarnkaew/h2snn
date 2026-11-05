"""Tests for the baseline campaign scaffolding.

These guard the properties that make the comparison tables meaningful: that the cell
matrix is complete and explicit, that every model sees the same data, and that the
vendored repositories cannot contaminate each other.
"""

from __future__ import annotations

import ast
import builtins
import functools
import sys
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=None)
def cached_bundle(task: str, dataset: str) -> Any:
    """Build each bundle once for the whole test module.

    Loading DBLP or IMDB costs several seconds, and the split assertions alone would
    otherwise rebuild each of them three times. Safe to share here because every test
    below only reads. Deliberately *not* in src/baselines/bundles.py: handing the same
    mutable bundle to two adapters in a real run is a hazard, and each trial there is a
    fresh process anyway.
    """

    from src.baselines.bundles import build_bundle

    return build_bundle(task, dataset)


class RegistryTests(unittest.TestCase):
    def test_campaign_matrix_matches_planned_cell_counts(self) -> None:
        from src.baselines.registry import DATASETS, campaign_summary, iter_cells

        summary = campaign_summary()
        self.assertEqual(summary["runnable_cells"], 113)

        per_task = {
            task: sum(
                1 for cell in iter_cells() if cell["task"] == task and cell["runnable"]
            )
            for task in DATASETS
        }
        self.assertEqual(per_task["node_classification"], 39)
        # 40 minus LatGRL x {LastFM, YouTube}: its latent-graph construction is built on
        # feature cosine similarity and both of those datasets are attribute-free, which
        # zeroes the low-pass graph outright. See LATGRL_FEATURELESS_REASON.
        self.assertEqual(per_task["link_prediction"], 38)
        self.assertEqual(per_task["graph_classification"], 36)

    def test_every_cell_is_runnable_or_has_a_reason(self) -> None:
        from src.baselines.registry import iter_cells

        for cell in iter_cells():
            if not cell["runnable"]:
                self.assertTrue(
                    cell["reason"],
                    f"{cell['model']}/{cell['task']}/{cell['dataset']} is unsupported "
                    "but carries no reason; it would render as a silent blank.",
                )

    def test_latgrl_graph_classification_reason_is_specific(self) -> None:
        """The generic 'no implementation' fallback must not mask the real reason."""

        from src.baselines.registry import unsupported_reason

        reason = unsupported_reason("latgrl", "graph_classification", "nci1")
        self.assertIsNotNone(reason)
        self.assertIn("unsupervised", reason)
        self.assertIn("60.3 GB", reason)

    def test_every_model_module_declares_an_adapter(self) -> None:
        import importlib.util

        from src.baselines.registry import MODELS

        for spec in MODELS:
            module_path = PROJECT_ROOT / Path(*spec.module.split(".")).with_suffix(".py")
            if not module_path.exists():
                continue  # adapter not written yet; covered by the campaign checklist
            source = module_path.read_text(encoding="utf-8")
            self.assertIn(
                "ADAPTER",
                source,
                f"{spec.module} exists but exposes no ADAPTER instance.",
            )


class VendorIsolationTests(unittest.TestCase):
    def test_colliding_vendor_module_names_do_not_leak(self) -> None:
        """PSHGCN and H2SGNN both ship `processing.py`; each must get its own.

        Regression test: sys.modules is keyed by bare module name, so without eviction the
        first vendor imported would silently satisfy the second vendor's imports. That
        surfaced as an arity error, but could just as easily have produced results
        computed from the wrong graph.
        """

        import sys

        from src.baselines.runtime import pushd_sys_path, vendor_path

        seen: dict[str, Path] = {}
        for vendor in ("pshgcn", "h2sgnn"):
            with pushd_sys_path(vendor):
                import processing  # type: ignore[import-not-found]

                seen[vendor] = Path(processing.__file__).resolve()

        self.assertNotEqual(
            seen["pshgcn"],
            seen["h2sgnn"],
            "both vendors resolved to the same processing module",
        )
        for vendor, resolved in seen.items():
            self.assertEqual(resolved.parent, vendor_path(vendor).resolve())

        # Checked here rather than in its own test: importing a vendor module costs
        # several seconds on this checkout, so the two assertions share one import.
        leaked = {
            name
            for name, module in sys.modules.items()
            if getattr(module, "__file__", "")
            and "PSHGCN" in str(module.__file__)
        }
        self.assertFalse(leaked, f"vendor modules leaked out of the block: {leaked}")

    def test_all_declared_vendor_clones_are_present(self) -> None:
        from src.baselines.runtime import available_vendors

        missing = [name for name, present in available_vendors().items() if not present]
        self.assertFalse(missing, f"missing baseline clones: {missing}")


class PublishedParamsTests(unittest.TestCase):
    def test_published_files_declare_a_source(self) -> None:
        from src.baselines.published import (
            available_published_cells,
            load_published_params,
        )

        cells = available_published_cells()
        self.assertTrue(cells, "no published hyperparameters have been transcribed yet")
        for model, entries in cells.items():
            for task, dataset in entries:
                params = load_published_params(model, task, dataset)
                self.assertIsNotNone(params)
                self.assertEqual(params.source, "published")
                self.assertTrue(params.values)

    def test_published_cells_are_declared_runnable(self) -> None:
        from src.baselines.published import available_published_cells
        from src.baselines.registry import unsupported_reason

        for model, entries in available_published_cells().items():
            for task, dataset in entries:
                self.assertIsNone(
                    unsupported_reason(model, task, dataset),
                    f"{model} has published params for {task}/{dataset} but the registry "
                    "marks that cell unsupported.",
                )


class PolynomialRelationKeyTests(unittest.TestCase):
    """The spectral baselines (PSHGCN, H2SGNN) index relations by 2-character keys.

    PSHGCN publishes its own link-prediction preprocessing for LastFM and Amazon
    (PSHGCN/hgb-lp/processing.py), so those two cells have a ground truth to match. The
    other two datasets have to generalise from it, and the ways that can go wrong are
    silent: pooling parallel relations, or emitting a key whose transpose is not a key.
    """

    def _keys(self, dataset: str):
        import torch

        from src.baselines.adapters._polynomial_lp import build_relation_adjacencies

        return build_relation_adjacencies(
            cached_bundle("link_prediction", dataset), torch.device("cpu")
        )

    def test_lastfm_matches_the_authors_key_set(self) -> None:
        """Their LastFM keys are UA, AU, UU, AT, TA -- 5 operators for 3 relations."""

        from src.baselines.adapters._polynomial_lp import (
            build_term_lists,
            polynomial_term_count,
        )

        adjacencies, chaining = self._keys("lastfm")
        self.assertTrue(chaining, "LastFM has 3 node types, so keys encode types")
        self.assertEqual(len(adjacencies), 5)
        # One key per direction for the two cross-type relations, one symmetrised key
        # for the single same-type relation.
        self.assertEqual(sum(1 for key in adjacencies if key[0] == key[1]), 1)

        lis, _ = build_term_lists(adjacencies, 1, enforce_type_chaining=chaining)
        # init_coe at their published K=1 allocates 1 + len(lis[1]) coefficients.
        self.assertEqual(polynomial_term_count(lis), 6)

    def test_amazon_keeps_parallel_relations_apart(self) -> None:
        """Their Amazon branch emits PP and UU for two product-product relations.

        Pooling them into one operator would silently reduce the model to its
        homogeneous limit on this dataset.
        """

        from src.baselines.adapters._polynomial_lp import (
            build_term_lists,
            polynomial_term_count,
        )

        adjacencies, chaining = self._keys("amazon")
        self.assertEqual(len(adjacencies), 2, "one operator per relation, not per type pair")
        self.assertFalse(chaining, "one node type: every composition is valid")

        lis, _ = build_term_lists(adjacencies, 3, enforce_type_chaining=chaining)
        # Their published K=3 with 2 keys: 2 + 4 + 8 monomials, plus the identity term.
        self.assertEqual(polynomial_term_count(lis), 15)

    def test_youtube_gets_one_operator_per_relation(self) -> None:
        """Five user-user relations must stay five operators."""

        adjacencies, chaining = self._keys("youtube")
        self.assertEqual(len(adjacencies), 5)
        self.assertFalse(chaining)

    def test_pubmed_keys_encode_node_types(self) -> None:
        """Ten relations over four types, every type pair unique, so chaining applies."""

        adjacencies, chaining = self._keys("pubmed")
        self.assertTrue(chaining)
        self.assertEqual(len(adjacencies), 16)

    def test_single_node_type_keys_are_palindromic(self) -> None:
        """lis_t reverses a key in place, so a symmetric operator needs a palindrome.

        A label like 'RA' reverses to 'AR', which is not a key, and the vendor forward
        does a bare ``adjs[j[:2]]`` lookup -- it would raise KeyError mid-training.
        """

        for dataset in ("amazon", "youtube"):
            adjacencies, _ = self._keys(dataset)
            for key in adjacencies:
                self.assertEqual(key, key[::-1], f"{dataset} key {key!r} is not symmetric")

    def test_every_transposed_monomial_resolves_to_known_keys(self) -> None:
        """The invariant behind the palindrome rule, checked on all four datasets."""

        from src.baselines.adapters._polynomial_lp import build_term_lists

        for dataset in ("lastfm", "amazon", "youtube", "pubmed"):
            adjacencies, chaining = self._keys(dataset)
            # build_term_lists asserts this internally; call it at an order high enough
            # to exercise composition, and let a violation fail the test.
            _, lis_t = build_term_lists(
                adjacencies, 2, enforce_type_chaining=chaining
            )
            for monomials in lis_t.values():
                for monomial in monomials:
                    for position in range(0, len(monomial), 2):
                        self.assertIn(monomial[position : position + 2], adjacencies)

    def test_keys_are_exactly_two_characters(self) -> None:
        """The model slices adjs[j[:2]]; a longer key would index the wrong operator."""

        for dataset in ("lastfm", "amazon", "youtube", "pubmed"):
            adjacencies, _ = self._keys(dataset)
            for key in adjacencies:
                self.assertEqual(len(key), 2, f"{dataset} key {key!r}")


class CampaignUnitTests(unittest.TestCase):
    """Unit enumeration decides how many times every cell is fitted."""

    def test_node_and_link_units_are_seeds(self) -> None:
        from src.baselines.campaign import build_units

        for task in ("node_classification", "link_prediction"):
            units = build_units(task, base_seed=42, num_runs=5)
            self.assertEqual(len(units), 5)
            self.assertEqual([unit.seed for unit in units], [42, 43, 44, 45, 46])

    def test_graph_units_are_seeded_holdout_runs(self) -> None:
        """HINPool's protocol: five seeded stratified 80/10/10 holdouts, not folds."""

        from src.baselines.campaign import DEFAULT_NUM_RUNS, build_units

        units = build_units("graph_classification")
        self.assertEqual(len(units), DEFAULT_NUM_RUNS)
        self.assertEqual(len(units), 5)
        # Distinct seeds, because the seed is what redraws the stratified split.
        self.assertEqual(len({unit.seed for unit in units}), 5)

    def test_graph_split_comes_from_seed_not_overrides(self) -> None:
        """The split position travels as run_index/seed, which build_bundle takes."""

        from src.baselines.campaign import Cell, build_raw_base, build_units, bundle_overrides

        cell = Cell(model="csnn", task="graph_classification", dataset="mutag")
        unit = build_units("graph_classification")[3]
        raw = build_raw_base(cell, unit)
        self.assertEqual(bundle_overrides(raw), {})
        self.assertEqual(raw["run_index"], 3)
        self.assertEqual(raw["seed"], 45)

    def test_link_units_carry_no_split_overrides(self) -> None:
        from src.baselines.campaign import Cell, build_raw_base, build_units, bundle_overrides

        cell = Cell(model="gcn", task="link_prediction", dataset="lastfm")
        unit = build_units("link_prediction")[0]
        self.assertEqual(bundle_overrides(build_raw_base(cell, unit)), {})


class CampaignSweepTests(unittest.TestCase):
    def test_every_runnable_cell_builds_a_sweep(self) -> None:
        """A search space that fails to resolve must fail now, not mid-campaign."""

        from src.baselines.campaign import (
            Cell,
            build_sweep_config,
            default_epochs,
            monitor_for,
        )
        from src.baselines.registry import iter_cells

        for entry in iter_cells():
            if not entry["runnable"]:
                continue
            task = entry["task"]
            cell = Cell(model=entry["model"], task=task, dataset=entry["dataset"])
            sweep = build_sweep_config(
                cell,
                monitor=monitor_for(task),
                n_trials=30,
                epochs=default_epochs(task),
            )
            self.assertEqual(sweep["method"], "bayes")
            self.assertEqual(sweep["early_terminate"]["type"], "hyperband")
            self.assertTrue(
                sweep["parameters"],
                f"{cell.token} produced an empty search space",
            )

    def test_hyperband_window_must_be_valid(self) -> None:
        from src.baselines.campaign import Cell, build_sweep_config

        cell = Cell(model="gcn", task="link_prediction", dataset="lastfm")
        with self.assertRaises(ValueError):
            build_sweep_config(cell, monitor="auroc", n_trials=5, epochs=20,
                               hyperband_min_iter=20)


class CampaignSummaryTests(unittest.TestCase):
    def test_graph_summary_reports_one_standard_deviation(self) -> None:
        """Under HINPool's holdout there is no repetition structure to average over."""

        import tempfile
        from pathlib import Path as _Path

        from src.baselines.campaign import summarize_cell

        records = [{"test_macro_f1": value} for value in (0.1, 0.2, 0.3, 0.4, 0.5)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize_cell(
                records, "graph_classification", _Path(tmp) / "s.json"
            )
        entry = summary["metrics"]["test_macro_f1"]
        self.assertAlmostEqual(entry["mean"], 0.3)
        self.assertNotIn("std_over_repetitions", entry)
        self.assertNotIn("std_over_runs", entry)

    def test_seed_tasks_report_one_standard_deviation(self) -> None:
        import tempfile
        from pathlib import Path as _Path

        from src.baselines.campaign import summarize_cell

        records = [{"test_auroc": value} for value in (0.1, 0.2, 0.3, 0.4, 0.5)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize_cell(records, "link_prediction", _Path(tmp) / "s.json")
        entry = summary["metrics"]["test_auroc"]
        self.assertAlmostEqual(entry["mean"], 0.3)
        self.assertNotIn("std_over_repetitions", entry)


class BaselineNotebookTests(unittest.TestCase):
    NOTEBOOKS = {
        "baselines_node_classification.ipynb": "node_classification",
        "baselines_link_prediction.ipynb": "link_prediction",
        "baselines_graph_classification.ipynb": "graph_classification",
    }

    def _payload(self, name: str) -> Any:
        import json

        path = PROJECT_ROOT / "notebooks" / name
        self.assertTrue(path.exists(), f"missing notebook: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_notebooks_are_committed_clean(self) -> None:
        """Outputs and execution counts must not be committed."""

        for name in self.NOTEBOOKS:
            payload = self._payload(name)
            for cell in payload["cells"]:
                self.assertIsNone(cell.get("execution_count"))
                self.assertFalse(cell.get("outputs"))

    def test_notebooks_drive_the_entry_scripts(self) -> None:
        """The notebooks must call the scripts, not reimplement the campaign."""

        for name, task in self.NOTEBOOKS.items():
            payload = self._payload(name)
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in payload["cells"]
            )
            self.assertIn("scripts/run_baseline_tuning.py", source)
            self.assertIn("scripts/run_baseline_published.py", source)
            self.assertIn("--validate_only", source)
            self.assertIn("--resume", source)
            self.assertIn("UNIT_START", source)
            self.assertIn("UNIT_COUNT", source)
            self.assertIn("WANDB_API_KEY", source)
            self.assertIn("h2snn_baseline", source)
            self.assertIn(f"TASK = {task!r}", source)


class HomogeneousViewTests(unittest.TestCase):
    def test_symmetrization_adds_self_loops_and_drops_duplicates(self) -> None:
        import torch

        from src.baselines.bundles import _symmetrize_with_self_loops

        # 0->1 and 1->0 already present, plus a self-loop that must not be duplicated.
        edge_index = torch.tensor([[0, 1, 2, 2], [1, 0, 2, 0]])
        out = _symmetrize_with_self_loops(edge_index, num_nodes=3)

        self.assertEqual(out.size(1), torch.unique(out, dim=1).size(1))
        loops = {int(i) for i in out[0][out[0] == out[1]]}
        self.assertEqual(loops, {0, 1, 2})
        pairs = {(int(s), int(d)) for s, d in zip(*out.tolist())}
        for source, destination in [(0, 1), (1, 0), (0, 2), (2, 0)]:
            self.assertIn((source, destination), pairs)

    def test_link_prediction_view_keeps_real_features(self) -> None:
        """Regression: the default view once replaced real features with an identity."""

        import torch

        from src.baselines.bundles import build_homogeneous_view

        view = build_homogeneous_view(
            x=torch.randn(5, 4),
            node_type=torch.zeros(5, dtype=torch.long),
            node_type_names=("product",),
            node_slices={"product": (0, 5)},
            feature_dims={"product": 4},
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            target_node_type="product",
            feats_type=0,
        )
        self.assertEqual(view.in_dims, [4])
        self.assertEqual(view.features_list[0].shape, (5, 4))

    def test_featureless_type_falls_back_to_identity(self) -> None:
        import torch

        from src.baselines.bundles import build_homogeneous_view

        view = build_homogeneous_view(
            x=torch.zeros(3, 0),
            node_type=torch.zeros(3, dtype=torch.long),
            node_type_names=("user",),
            node_slices={"user": (0, 3)},
            feature_dims={"user": 0},
            edge_index=torch.tensor([[0], [1]]),
            target_node_type="user",
            feats_type=0,
        )
        self.assertEqual(view.in_dims, [3])
        self.assertTrue(view.features_list[0].is_sparse)


class MultiLabelProtocolTests(unittest.TestCase):
    """HGB IMDB is multi-label and the two binarisations are not interchangeable."""

    @staticmethod
    def _vendor_evaluation(output, labels):
        """Verbatim reimplementation of PSHGCN/hgb/utils.py:Evaluation."""
        import numpy as np
        from sklearn import metrics as skm

        preds = output.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        binary_pred = np.zeros(preds.shape).astype("int")
        for i in range(preds.shape[0]):
            k = labels[i].sum().astype("int")
            topk_idx = preds[i].argsort()[-k:]
            binary_pred[i][topk_idx] = 1
        return (
            skm.f1_score(labels, binary_pred, average="micro"),
            skm.f1_score(labels, binary_pred, average="macro"),
        )

    def test_oracle_topk_matches_the_vendor_implementation(self) -> None:
        import torch

        from src.baselines.metrics import classification_metrics

        torch.manual_seed(0)
        for _ in range(3):
            logits = torch.randn(200, 5)
            labels = torch.zeros(200, 5)
            for row in range(200):
                count = int(torch.randint(1, 4, (1,)))
                labels[row, torch.randperm(5)[:count]] = 1.0

            micro, macro = self._vendor_evaluation(logits, labels)
            mine = classification_metrics(
                logits, labels, is_multilabel=True, multilabel_protocol="oracle_topk"
            )
            self.assertAlmostEqual(mine["micro_f1"], micro, places=12)
            self.assertAlmostEqual(mine["macro_f1"], macro, places=12)

    def test_topk_selects_exactly_the_true_label_count(self) -> None:
        import torch

        from src.baselines.metrics import binarize_multilabel

        logits = torch.tensor([[3.0, 1.0, 2.0], [0.1, 0.2, 0.3]])
        labels = torch.tensor([[1, 0, 1], [0, 0, 1]])
        predictions = binarize_multilabel(logits, labels, protocol="oracle_topk")
        self.assertEqual(predictions.tolist(), [[1, 0, 1], [0, 0, 1]])
        self.assertEqual(
            predictions.sum(dim=1).tolist(), labels.sum(dim=1).tolist()
        )

    def test_rejects_an_unknown_protocol(self) -> None:
        import torch

        from src.baselines.metrics import binarize_multilabel

        with self.assertRaises(ValueError):
            binarize_multilabel(
                torch.randn(2, 3), torch.ones(2, 3).long(), protocol="argmax"
            )

    def test_project_helper_can_reproduce_the_hgb_convention(self) -> None:
        """src/utils/misc.py must be able to reproduce PSHGCN/H2SGNN's convention."""
        import torch

        from src.baselines.metrics import classification_metrics
        from src.utils.misc import compute_multilabel_classification_metrics

        torch.manual_seed(1)
        logits = torch.randn(120, 4)
        labels = torch.zeros(120, 4)
        for row in range(120):
            labels[row, torch.randperm(4)[:2]] = 1.0

        shared = classification_metrics(
            logits, labels, is_multilabel=True, multilabel_protocol="oracle_topk"
        )
        project = compute_multilabel_classification_metrics(
            logits, labels, protocol="oracle_topk"
        )
        self.assertAlmostEqual(shared["macro_f1"], project["macro_f1"], places=12)
        self.assertAlmostEqual(shared["micro_f1"], project["micro_f1"], places=12)


class HGBSplitTests(unittest.TestCase):
    """Every baseline must use HGB's official split ratio, not its own.

    Three of the vendored repositories re-split the data themselves: Hetero2Net replaces
    HGB's masks with ``RandomNodeSplit(0.2, 0.2)`` (a 60/20/20 partition, roughly double
    HGB's training labels), while PSHGCN and HGB's own loader carve validation out with an
    unseeded ``np.random.shuffle``. The campaign overrides all of them, so these assertions
    are what keeps a silently-reintroduced vendor split from making a row incomparable.
    """

    EXPECTED = {
        # dataset: (HGB train count, HGB test count) from label.dat / label.dat.test
        "acm": (907, 2118),
        "dblp": (1217, 2840),
        "imdb": (1371, 3202),
    }

    def test_train_plus_val_equals_hgb_train_and_test_is_untouched(self) -> None:
        for dataset, (hgb_train, hgb_test) in self.EXPECTED.items():
            with self.subTest(dataset=dataset):
                data = cached_bundle("node_classification", dataset).data
                train = int(data.target_train_mask.sum())
                val = int(data.target_val_mask.sum())
                test = int(data.target_test_mask.sum())

                self.assertEqual(
                    train + val,
                    hgb_train,
                    "validation must be carved out of HGB's training mask, not taken "
                    "from test or from a fresh partition",
                )
                self.assertEqual(test, hgb_test, "HGB's test mask must be used verbatim")

    def test_validation_is_twenty_percent_of_hgb_train(self) -> None:
        for dataset in self.EXPECTED:
            with self.subTest(dataset=dataset):
                data = cached_bundle("node_classification", dataset).data
                train = int(data.target_train_mask.sum())
                val = int(data.target_val_mask.sum())
                self.assertAlmostEqual(val / (train + val), 0.20, places=2)

    def test_splits_are_disjoint(self) -> None:
        for dataset in self.EXPECTED:
            with self.subTest(dataset=dataset):
                data = cached_bundle("node_classification", dataset).data
                train = data.target_train_mask
                val = data.target_val_mask
                test = data.target_test_mask
                self.assertEqual(int((train & val).sum()), 0)
                self.assertEqual(int((train & test).sum()), 0)
                self.assertEqual(int((val & test).sum()), 0)


class ReproducibilityTests(unittest.TestCase):
    def test_vendor_model_init_is_reproducible_under_seed_everything(self) -> None:
        """Seeding torch alone is not enough for these vendored models.

        PSHGCN initialises its propagation coefficients with ``np.random.uniform``
        (PSHGCN/hgb/model.py:init_coe), so an adapter that only calls
        ``torch.manual_seed`` produces a different model on every run of the same seed --
        silently turning a 5-seed mean into noise.

        Tested against the initialiser directly on a synthetic term list rather than by
        training on IMDB: the mechanism is the numpy RNG, and loading the dataset twice
        cost two minutes for the same assertion.
        """

        import torch
        from types import SimpleNamespace

        from src.baselines.runtime import pushd_sys_path
        from src.utils.misc import seed_everything

        args = SimpleNamespace(
            emb_dim=8, hidden=8, K=2, dropout=0.0, input_drop=0.0, dataset="ACM"
        )
        lis = {1: ["AB"], 2: ["ABBA"]}

        def build_coefficients(seed: int) -> torch.Tensor:
            seed_everything(seed, "cpu")
            with pushd_sys_path("pshgcn"):
                from model import PSHGCN  # type: ignore[import-not-found]

                model = PSHGCN([4], 3, lis, lis, args)
            return model.W.detach().clone()

        self.assertTrue(torch.equal(build_coefficients(11), build_coefficients(11)))
        self.assertFalse(torch.equal(build_coefficients(11), build_coefficients(12)))



class AdapterContractTests(unittest.TestCase):
    """`supports()` must not promise what `fit_task` cannot deliver.

    An over-promising `supports()` is worse than a missing adapter: the campaign accepts
    the cell at setup, spends the scheduling budget, and only fails once a worker starts
    training. PSHGCN shipped exactly that bug -- `supports()` returned True for link
    prediction while `fit_task` raised UnsupportedCell.
    """

    def test_supports_agrees_with_search_space(self) -> None:
        from src.baselines.protocol import UnsupportedCell
        from src.baselines.registry import DATASETS, MODELS, get_adapter

        for spec in MODELS:
            try:
                adapter = get_adapter(spec.name)
            except Exception:
                continue  # adapter not written yet
            for task, datasets in DATASETS.items():
                for dataset in datasets:
                    if not adapter.supports(task, dataset):
                        continue
                    with self.subTest(model=spec.name, task=task, dataset=dataset):
                        try:
                            space = adapter.search_space(task, dataset)
                        except UnsupportedCell:
                            self.fail(
                                f"{spec.name}.supports({task}, {dataset}) is True but "
                                "search_space raises UnsupportedCell"
                            )
                        self.assertTrue(space, "an empty search space is not tunable")

    def test_supported_cells_are_runnable_in_the_registry(self) -> None:
        from src.baselines.registry import DATASETS, MODELS, get_adapter, unsupported_reason

        for spec in MODELS:
            try:
                adapter = get_adapter(spec.name)
            except Exception:
                continue
            for task, datasets in DATASETS.items():
                for dataset in datasets:
                    if adapter.supports(task, dataset):
                        with self.subTest(model=spec.name, task=task, dataset=dataset):
                            self.assertIsNone(
                                unsupported_reason(spec.name, task, dataset),
                                "adapter claims a cell the registry marks unsupported",
                            )


class CoverageReportTests(unittest.TestCase):
    def test_report_current_adapter_coverage(self) -> None:
        """Not a pass/fail gate -- prints how much of the matrix is actually implemented.

        Keeps the real number visible so progress is never inferred from a summary.
        """

        from src.baselines.registry import DATASETS, MODELS, get_adapter, iter_cells

        covered = 0
        runnable = sum(1 for cell in iter_cells() if cell["runnable"])
        missing: list[str] = []
        for spec in MODELS:
            try:
                adapter = get_adapter(spec.name)
            except Exception:
                missing.append(spec.name)
                continue
            for task, datasets in DATASETS.items():
                covered += sum(
                    1
                    for dataset in datasets
                    if adapter.supports(task, dataset)
                    and any(
                        cell["runnable"]
                        for cell in iter_cells((task,), (spec.name,))
                        if cell["dataset"] == dataset
                    )
                )

        print(
            f"[coverage] {covered}/{runnable} runnable cells have an adapter; "
            f"no adapter yet: {', '.join(missing) if missing else 'none'}"
        )
        self.assertGreater(covered, 0)



class MultiCallSafetyTests(unittest.TestCase):
    def test_adapters_can_be_called_twice_in_one_process(self) -> None:
        """A second fit in the same process must not fail on import side effects.

        Regression: H2GB registers modules via decorators at import time, and
        ``pushd_sys_path`` was evicting it on exit, so a second ``import H2GB`` re-ran the
        registrations and raised ``KeyError: Module with 'GraphormerBias' already
        defined``. Each *tuned* trial runs in a fresh subprocess and would have survived,
        but the notebooks' published-run loops fit several seeds in one process, so this
        would have failed there.
        """

        from src.baselines.runtime import pushd_sys_path

        for _ in range(2):
            with pushd_sys_path("h2gb"):
                import H2GB  # type: ignore[import-not-found]  # noqa: F401




class VendorRootIntegrityTests(unittest.TestCase):
    def test_every_adapter_module_names_a_registered_vendor(self) -> None:
        """Every VENDOR referenced by an adapter must exist in VENDOR_ROOTS.

        Regression: an edit to the VENDOR_ROOTS literal dropped the ``csnn`` entry, and
        nothing caught it until a CSNN cell was actually run -- by which point both its
        node-classification and link-prediction cells were broken.
        """

        import re

        from src.baselines.runtime import VENDOR_ROOTS

        adapters = (PROJECT_ROOT / "src" / "baselines" / "adapters").glob("*.py")
        referenced: set[str] = set()
        for module_path in adapters:
            source = module_path.read_text(encoding="utf-8")
            for match in re.finditer(r'^VENDORS?\s*=\s*(.+)$', source, re.M):
                referenced.update(re.findall(r'"([a-z0-9_]+)"', match.group(1)))

        self.assertTrue(referenced, "no adapter declared a vendor")
        unknown = sorted(referenced - set(VENDOR_ROOTS))
        self.assertFalse(unknown, f"adapters reference unregistered vendors: {unknown}")



class HeadTests(unittest.TestCase):
    def test_link_decoders_score_and_backpropagate(self) -> None:
        import torch

        from src.baselines.heads import SharedLinkHead

        embeddings = torch.randn(6, 8, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [2, 3]])
        relations = torch.tensor([0, 1])
        for decoder_type in ("dot", "distmult"):
            head = SharedLinkHead(decoder_type, num_relations=2, dim=8)
            scores = head(embeddings, edge_index, relations)
            self.assertEqual(scores.shape, (2,))
            scores.sum().backward()
            self.assertIsNotNone(embeddings.grad)
            embeddings.grad = None

    def test_graph_head_pools_per_graph(self) -> None:
        import torch

        from src.baselines.heads import SharedGraphHead

        head = SharedGraphHead(dim=4, num_classes=3)
        logits = head(torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 2, 2]))
        self.assertEqual(logits.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()


class RequirementGroupTests(unittest.TestCase):
    """The optional-dependency split that lets one bad install cost only its own models.

    ``requirements.txt`` aggregates per-group files; the notebooks install them one at a
    time and drop the models a failed group backs. These guard the two ways that goes
    wrong silently: a group whose pin file has drifted away from the map, and a model that
    is absent from the map entirely and so is never skipped when its library is missing.
    """

    def test_every_group_has_a_pin_file_that_requirements_aggregates(self) -> None:
        from src.baselines.runtime.dependencies import REQUIREMENT_GROUPS

        aggregate = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        for name, group in REQUIREMENT_GROUPS.items():
            with self.subTest(group=name):
                self.assertTrue(group.path.is_file(), f"missing pin file: {group.path}")
                self.assertIn(f"-r requirements/{group.filename}", aggregate)
        self.assertIn("-r requirements/core.txt", aggregate)

    def test_every_registered_model_is_in_the_dependency_map(self) -> None:
        """A model missing from the map would raise instead of being skipped cleanly."""

        from src.baselines.registry import MODELS
        from src.baselines.runtime.dependencies import REQUIREMENT_GROUPS, groups_for

        for spec in MODELS:
            for task in spec.tasks:
                with self.subTest(model=spec.name, task=task):
                    for group in groups_for(spec.name, task):
                        self.assertIn(group, REQUIREMENT_GROUPS)

    def test_unknown_model_is_rejected(self) -> None:
        from src.baselines.runtime.dependencies import groups_for

        with self.assertRaises(KeyError):
            groups_for("not_a_baseline", "node_classification")

    def test_pshgcn_needs_dgl_only_for_link_prediction(self) -> None:
        """PSHGCN/hgb-lp/pshgcn.py opens with a leftover `import dgl`; hgb/ does not."""

        from src.baselines.protocol import (
            GRAPH_CLASSIFICATION,
            LINK_PREDICTION,
            NODE_CLASSIFICATION,
        )
        from src.baselines.runtime.dependencies import groups_for

        self.assertIn("dgl", groups_for("pshgcn", LINK_PREDICTION))
        self.assertNotIn("dgl", groups_for("pshgcn", NODE_CLASSIFICATION))
        self.assertNotIn("dgl", groups_for("pshgcn", GRAPH_CLASSIFICATION))

    def test_dgl_free_models_stay_runnable_when_dgl_is_absent(self) -> None:
        """The whole point of the split: a missing group must not spread."""

        from src.baselines.protocol import NODE_CLASSIFICATION
        from src.baselines.runtime import dependencies

        requested = ("gcn", "gat", "han", "sehgnn", "hinormer", "csnn", "latgrl", "pshgcn")
        real = dependencies.REQUIREMENT_GROUPS["dgl"].is_installed
        try:
            dependencies.REQUIREMENT_GROUPS["dgl"].__class__.is_installed = (
                lambda self: self.name != "dgl"
            )
            blocked = dependencies.models_missing_dependencies(
                requested, NODE_CLASSIFICATION
            )
        finally:
            dependencies.REQUIREMENT_GROUPS["dgl"].__class__.is_installed = real.__func__
        self.assertEqual(
            sorted(blocked), ["gat", "gcn", "han", "hinormer", "sehgnn"]
        )
        for survivor in ("csnn", "latgrl", "pshgcn"):
            self.assertNotIn(survivor, blocked)


class SplitInstallNotebookTests(unittest.TestCase):
    """The three notebooks the split-install flow was built for."""

    NOTEBOOKS = (
        "baselines_node_classification_subset.ipynb",
        "baselines_link_prediction.ipynb",
        "baselines_graph_classification.ipynb",
    )

    def _source(self, name: str) -> str:
        import json

        payload = json.loads(
            (PROJECT_ROOT / "notebooks" / name).read_text(encoding="utf-8")
        )
        return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    def test_notebooks_install_core_then_groups_and_filter(self) -> None:
        for name in self.NOTEBOOKS:
            with self.subTest(notebook=name):
                source = self._source(name)
                self.assertIn("requirements' / 'core.txt", source)
                self.assertIn("REQUIREMENT_GROUPS", source)
                self.assertIn("models_missing_dependencies", source)
                # A bare `import dgl` at module scope is what broke on a runtime without
                # DGL; it must stay behind the installed-group check.
                self.assertNotIn("\nimport dgl\n", source)

    def test_every_code_cell_parses(self) -> None:
        import ast
        import json

        for name in self.NOTEBOOKS:
            payload = json.loads(
                (PROJECT_ROOT / "notebooks" / name).read_text(encoding="utf-8")
            )
            for index, cell in enumerate(payload["cells"]):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(notebook=name, cell=index):
                    ast.parse("".join(cell["source"]))


class TrainingBudgetTests(unittest.TestCase):
    """Epochs and patience are protocol, so they must be identical across a table.

    Before ``src/baselines/budget.py`` these were per-adapter literals copied from each
    author's own defaults, which put H2SGNN on 1000 epochs / patience 100 and GCN on
    300 / 30 in the same node-classification table, and GCN/GAT on 100 / 30 against
    everyone else's 300 / 100 in graph classification.
    """

    ADAPTER_DIR = PROJECT_ROOT / "src" / "baselines" / "adapters"

    def test_no_adapter_sets_its_own_budget(self) -> None:
        import re

        pattern = re.compile(r'params\.get\(\s*"(epochs|early_stopping)"')
        offenders = []
        for path in sorted(self.ADAPTER_DIR.glob("*.py")):
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "Budget must come from src/baselines/budget.py, not an adapter literal:\n"
            + "\n".join(offenders),
        )

    def test_budget_is_never_searched(self) -> None:
        """A tuned trial must not be able to buy itself a longer run."""

        from src.baselines.registry import get_adapter, iter_cells

        for entry in iter_cells():
            if not entry["runnable"]:
                continue
            space = get_adapter(entry["model"]).search_space(
                entry["task"], entry["dataset"]
            )
            with self.subTest(model=entry["model"], task=entry["task"]):
                self.assertNotIn("epochs", space)
                self.assertNotIn("early_stopping", space)

    def test_tuned_params_get_the_shared_budget(self) -> None:
        from src.baselines.budget import default_budget, resolve_budget
        from src.baselines.protocol import TASKS, BaselineParams

        tuned = BaselineParams(values={"lr": 1e-3}, source="tuned")
        for task in TASKS:
            with self.subTest(task=task):
                self.assertEqual(resolve_budget(task, tuned), default_budget(task))

    def test_published_params_keep_the_authors_budget(self) -> None:
        """Reproducing a reported number means using the budget it was reported at."""

        from src.baselines.budget import resolve_budget
        from src.baselines.protocol import NODE_CLASSIFICATION, BaselineParams

        published = BaselineParams(
            values={"epochs": 1000, "early_stopping": 42}, source="published"
        )
        budget = resolve_budget(NODE_CLASSIFICATION, published)
        self.assertEqual((budget.epochs, budget.patience), (1000, 42))

    def test_hyperband_bracket_matches_the_real_epoch_budget(self) -> None:
        from src.baselines.budget import default_budget
        from src.baselines.campaign import default_epochs
        from src.baselines.protocol import TASKS

        for task in TASKS:
            with self.subTest(task=task):
                self.assertEqual(default_epochs(task), default_budget(task).epochs)


class SearchSpaceCoverageTests(unittest.TestCase):
    """A tuned cell must be able to reach its own published configuration.

    Otherwise the model is reported at the best of a grid its authors would not have
    chosen, and the comparison understates it. Twenty-five published values sat outside
    their search space before ``src/baselines/search_grids.py``, and nineteen more keys
    were neither searched nor left at the published value.
    """

    SKIP = frozenset({"epochs", "early_stopping", "probe_epochs"})

    @staticmethod
    def _reachable(spec: dict[str, Any], value: Any) -> bool:
        if "values" in spec:
            return value in spec["values"]
        if "min" in spec and "max" in spec:
            return spec["min"] <= value <= spec["max"]
        return True

    def _adapter_defaults(self, model: str) -> dict[str, set[Any]]:
        import ast
        import re

        from src.baselines.registry import model_spec

        path = PROJECT_ROOT / (model_spec(model).module.replace(".", "/") + ".py")
        defaults: dict[str, set[Any]] = {}
        for key, raw in re.findall(
            r'params\.get\(\s*"([A-Za-z_0-9]+)"\s*,\s*([^)]+?)\s*\)',
            path.read_text(encoding="utf-8"),
        ):
            try:
                defaults.setdefault(key, set()).add(ast.literal_eval(raw))
            except (SyntaxError, ValueError):
                continue
        return defaults

    def test_every_published_value_is_reachable(self) -> None:
        from src.baselines.published import load_published_params
        from src.baselines.registry import get_adapter, iter_cells

        unreachable: list[str] = []
        for entry in iter_cells():
            if not entry["runnable"]:
                continue
            model, task, dataset = entry["model"], entry["task"], entry["dataset"]
            published = load_published_params(model, task, dataset)
            if published is None:
                continue
            space = get_adapter(model).search_space(task, dataset)
            defaults = self._adapter_defaults(model)
            for key, value in published.values.items():
                if key in self.SKIP:
                    continue
                if key in space:
                    if not self._reachable(space[key], value):
                        unreachable.append(
                            f"{model}/{task}/{dataset}: {key}={value!r} "
                            f"outside {space[key]}"
                        )
                elif value not in defaults.get(key, set()):
                    unreachable.append(
                        f"{model}/{task}/{dataset}: {key}={value!r} is neither searched "
                        f"nor the adapter default "
                        f"{sorted(defaults.get(key, set()), key=str)}"
                    )
        self.assertEqual(unreachable, [], "\n".join(unreachable))


class EntryPointBootstrapTests(unittest.TestCase):
    """Every script importing ``src`` must work when spawned by absolute path.

    ``run_unit_sweep`` re-spawns its entry point once per trial as
    ``python <abs path> --worker_context ...``, which puts scripts/ on ``sys.path[0]``
    and the project root nowhere. Without the bootstrap the worker dies on
    ``import src`` before it can report anything.
    """

    def test_scripts_importing_src_bootstrap_the_project_root(self) -> None:
        missing = []
        for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "\nfrom src." not in source and "\nimport src." not in source:
                continue
            if "sys.path.insert" not in source:
                missing.append(path.name)
        self.assertEqual(missing, [], f"missing sys.path bootstrap: {missing}")


class EpochReportingTests(unittest.TestCase):
    """Hyperband can only prune a trial it can see mid-flight."""

    def test_shared_loops_accept_a_report(self) -> None:
        import inspect

        from src.baselines import training

        for name in (
            "train_full_batch_node_classification",
            "train_link_prediction",
            "train_graph_classification",
        ):
            with self.subTest(loop=name):
                signature = inspect.signature(getattr(training, name))
                self.assertIn("report", signature.parameters)

    def test_report_key_matches_the_sweep_metric(self) -> None:
        """Live series and sweep metric must share a key, or nothing can prune."""

        from src.baselines.campaign import Cell, build_sweep_config, monitor_for
        from src.baselines.progress import WandbEpochReport

        for task, dataset in (
            ("node_classification", "acm"),
            ("link_prediction", "lastfm"),
        ):
            cell = Cell(model="gcn", task=task, dataset=dataset)
            monitor = monitor_for(task)
            sweep = build_sweep_config(cell, monitor=monitor, n_trials=5, epochs=200)
            with self.subTest(task=task):
                self.assertEqual(
                    sweep["metric"]["name"], WandbEpochReport(monitor)._best_key
                )

    def test_a_hyperband_stop_is_not_a_failure(self) -> None:
        """W&B terminates a pruned run thread with a bare ``Exception``."""

        from src.utils.wandb_campaign import is_sweep_prune

        self.assertTrue(is_sweep_prune(Exception()))
        self.assertFalse(is_sweep_prune(Exception("boom")))
        self.assertFalse(is_sweep_prune(RuntimeError()))
        self.assertFalse(is_sweep_prune(ValueError("CUDA out of memory")))


class LinkProtocolTests(unittest.TestCase):
    """One message-passing graph and one split rule for every link baseline."""

    def test_every_hgb_dataset_holds_out_at_hgbs_rate(self) -> None:
        """HGB draws validation at 1 - train_ratio = 0.10 on every link dataset."""

        from src.configs.config import LINK_PREDICTION_PRESETS

        for name in ("lastfm", "pubmed", "youtube", "amazon"):
            with self.subTest(dataset=name):
                self.assertEqual(LINK_PREDICTION_PRESETS[name]["link_val_ratio"], 0.10)

    def test_an_explicit_ratio_survives_the_preset(self) -> None:
        from src.configs.config import DataConfig

        config = DataConfig(dataset_name="amazon", link_val_ratio=0.09)
        self.assertEqual(
            config.resolved(frozenset({"link_val_ratio"})).link_val_ratio, 0.09
        )
        self.assertEqual(config.resolved().link_val_ratio, 0.10)

    def test_homogeneous_link_baselines_share_one_graph(self) -> None:
        """GCN/GAT read ``view.edge_index``, the reduction NSD and CSNN already got.

        Reading the directed ``data.edge_index`` instead gave the two families different
        evidence about the same held-out edge wherever relations are near-symmetric.
        """

        gnn = (PROJECT_ROOT / "src/baselines/adapters/hgb_gnn.py").read_text(
            encoding="utf-8"
        )
        shared = (PROJECT_ROOT / "src/baselines/adapters/_homogeneous_lp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("edge_index = view.edge_index.to(device)", gnn)
        self.assertNotIn("edge_index = data.edge_index.to(device)", gnn)
        self.assertIn("view.edge_index if self_loops", shared)

    def test_split_overlap_is_measured(self) -> None:
        """HGB's split leaves part of each held-out set visible; it must be recorded."""

        from src.baselines.link_audit import link_split_overlap

        overlap = link_split_overlap(cached_bundle("link_prediction", "lastfm").data)
        for split in ("val", "test"):
            self.assertIn(f"{split}_reverse_in_train_fraction", overlap)
            self.assertIn(f"{split}_forward_in_train_fraction", overlap)
        # LastFM's target is user-artist, so no held-out edge can have a reverse under
        # the same relation and the exposure is exactly zero. A non-zero value here
        # means the split or the reduction changed.
        self.assertEqual(overlap["val_reverse_in_train"], 0.0)
        self.assertEqual(overlap["test_forward_in_train"], 0.0)


class SelectionMetricTests(unittest.TestCase):
    """Each task selects on the metric its own table leads with.

    Selecting on one metric and reporting another tunes a model for something the paper
    never shows. Graph classification is the case worth guarding: it reports Macro-F1,
    and ``src/baselines/training.py`` carried a module docstring claiming it selected on
    AUROC long after the code had moved to Macro-F1.
    """

    EXPECTED = {
        "node_classification": "macro_f1",
        "graph_classification": "macro_f1",
        "link_prediction": "auroc",
    }

    def test_monitor_matches_the_reported_headline_metric(self) -> None:
        from src.utils.run_reports import (
            default_monitor_for_task,
            report_metrics_for_task,
        )

        for task, monitor in self.EXPECTED.items():
            with self.subTest(task=task):
                self.assertEqual(default_monitor_for_task(task), monitor)
                # The headline metric is the first one the table reports.
                self.assertEqual(report_metrics_for_task(task)[0], f"test_{monitor}")

    def test_sweeps_optimise_the_same_metric_training_selects_on(self) -> None:
        from src.baselines.campaign import Cell, build_sweep_config, monitor_for

        for task, dataset in (
            ("node_classification", "acm"),
            ("graph_classification", "mutag"),
            ("link_prediction", "lastfm"),
        ):
            monitor = monitor_for(task)
            sweep = build_sweep_config(
                Cell(model="gcn", task=task, dataset=dataset),
                monitor=monitor,
                n_trials=5,
                epochs=200,
            )
            with self.subTest(task=task):
                self.assertEqual(monitor, self.EXPECTED[task])
                self.assertEqual(sweep["metric"]["name"], f"best/{monitor}")
                self.assertEqual(sweep["metric"]["goal"], "maximize")

    def test_each_loop_selects_on_its_documented_metric(self) -> None:
        """Read the loops' source: the stopping rule must use the expected key."""

        import inspect

        from src.baselines import training

        node = inspect.getsource(training.train_full_batch_node_classification)
        graph = inspect.getsource(training.train_graph_classification)
        link = inspect.getsource(training.train_link_prediction)

        self.assertIn('monitor = val_metrics["macro_f1"]', node)
        self.assertIn('monitor = val_metrics["macro_f1"]', graph)
        self.assertNotIn('val_metrics["auroc"]', graph)
        self.assertIn('val_metrics.get("auroc_macro", val_metrics["auroc"])', link)

    def test_the_module_docstring_does_not_contradict_the_code(self) -> None:
        from src.baselines import training

        docstring = (training.__doc__ or "").lower()
        self.assertIn("macro-f1 for node classification", docstring)
        self.assertIn("graph classification", docstring)
        # The retired claim: AUROC covering graph classification as well as link
        # prediction. If it comes back, the file is lying about what it does again.
        self.assertNotIn("auroc for link prediction and\ngraph classification", docstring)


class DglCudaGateTests(unittest.TestCase):
    """A CPU-only DGL must not pass as installed on a platform where CUDA DGL exists.

    ``find_spec("dgl")`` cannot tell a CPU build from a CUDA one, so the notebooks'
    ``if group.is_installed(): continue`` printed ``already present`` and skipped the
    ``+cu124`` wheel whenever any DGL was there. GCN, GAT, HAN, SeHGNN and HINormer then
    trained on CPU -- correct numbers, wrong hardware, no error anywhere.
    """

    def _patched_group(self, *, platform: str, cuda_torch: bool, cuda_dgl: bool):
        """The dgl group with its environment probes replaced."""

        import contextlib
        from unittest import mock

        from src.baselines.runtime import dependencies

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(dependencies.sys, "platform", platform))
        stack.enter_context(
            mock.patch("torch.cuda.is_available", return_value=cuda_torch)
        )
        stack.enter_context(
            mock.patch(
                "src.baselines.runtime._dgl_compat.dgl_supports_cuda",
                return_value=cuda_dgl,
            )
        )
        return stack, dependencies.REQUIREMENT_GROUPS["dgl"]

    def test_cpu_only_dgl_on_linux_counts_as_not_installed(self) -> None:
        """Linux publishes a +cu124 wheel, so a CPU build there is a fixable defect."""

        stack, group = self._patched_group(
            platform="linux", cuda_torch=True, cuda_dgl=False
        )
        with stack:
            defect = group.defect()
            self.assertIsNotNone(defect)
            self.assertIn("--force-reinstall", defect)
            self.assertFalse(group.is_installed())

    def test_cuda_dgl_on_linux_is_healthy(self) -> None:
        stack, group = self._patched_group(
            platform="linux", cuda_torch=True, cuda_dgl=True
        )
        with stack:
            self.assertIsNone(group.defect())
            self.assertTrue(group.is_installed())

    def test_cpu_only_dgl_on_windows_stays_installed(self) -> None:
        """No CUDA wheel is published for torch 2.x on Windows.

        Treating that as a defect would make the local development environment report
        five baselines as unrunnable and retry an install that cannot succeed.
        """

        stack, group = self._patched_group(
            platform="win32", cuda_torch=True, cuda_dgl=False
        )
        with stack:
            self.assertIsNone(group.defect())
            self.assertTrue(group.is_installed())

    def test_a_machine_without_a_gpu_has_no_defect(self) -> None:
        stack, group = self._patched_group(
            platform="linux", cuda_torch=False, cuda_dgl=False
        )
        with stack:
            self.assertIsNone(group.defect())

    def test_report_explains_itself_on_every_platform(self) -> None:
        """The warning must reach the user even where the defect check stays silent."""

        from src.baselines.runtime.dependencies import dgl_cuda_report

        for platform in ("linux", "win32"):
            stack, _ = self._patched_group(
                platform=platform, cuda_torch=True, cuda_dgl=False
            )
            with stack, self.subTest(platform=platform):
                uses_gpu, explanation = dgl_cuda_report()
                self.assertFalse(uses_gpu)
                self.assertIn("without CUDA support", explanation)

        stack, _ = self._patched_group(platform="linux", cuda_torch=True, cuda_dgl=True)
        with stack:
            uses_gpu, _ = dgl_cuda_report()
            self.assertTrue(uses_gpu)

    def test_blocked_models_name_the_defect_not_a_missing_module(self) -> None:
        """`unavailable_reason` has to say which of the two problems this is."""

        from src.baselines.runtime.dependencies import unavailable_reason

        stack, _ = self._patched_group(
            platform="linux", cuda_torch=True, cuda_dgl=False
        )
        with stack:
            reason = unavailable_reason("hinormer", "node_classification")
            self.assertIsNotNone(reason)
            self.assertIn("CUDA", reason)
            self.assertIn("baselines-dgl.txt", reason)

    def test_only_dgl_backed_models_are_gated(self) -> None:
        """A CPU-only DGL must not disable the baselines that never touch DGL."""

        from src.baselines.runtime.dependencies import unavailable_reason

        stack, _ = self._patched_group(
            platform="linux", cuda_torch=True, cuda_dgl=False
        )
        with stack:
            for model in ("csnn", "nsd", "latgrl", "h2sgnn", "chebnetii"):
                with self.subTest(model=model):
                    self.assertIsNone(
                        unavailable_reason(model, "node_classification")
                    )
            # PSHGCN needs DGL on link prediction only.
            self.assertIsNone(unavailable_reason("pshgcn", "node_classification"))
            self.assertIsNotNone(unavailable_reason("pshgcn", "link_prediction"))


class LaneNotebookTests(unittest.TestCase):
    """The generated one-group-per-lane Colab notebooks.

    Each installs ``core.txt`` plus exactly one ``baselines-*.txt``, so a lane's runtime
    carries only what its own models need and the DGL lane can stay pinned to torch 2.6
    without holding the others there. They are generated artefacts:
    ``scripts/generate_baseline_notebooks.py`` is the source, and
    :meth:`test_committed_notebooks_match_the_generator` is what keeps that true.
    """

    @staticmethod
    def _lanes() -> list[Any]:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_baseline_notebooks",
            PROJECT_ROOT / "scripts" / "generate_baseline_notebooks.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`, so the
        # module has to be registered before it executes, not after.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.build_lanes(), module

    def _source(self, path: Path) -> str:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    def test_committed_notebooks_match_the_generator(self) -> None:
        """Editing the .ipynb instead of the generator must fail here, not in Colab."""

        lanes, module = self._lanes()
        for lane in lanes:
            with self.subTest(lane=lane.slug):
                self.assertTrue(lane.path.exists(), f"not generated: {lane.path}")
                self.assertEqual(
                    lane.path.read_text(encoding="utf-8"),
                    module.render(lane),
                    f"{lane.path.name} is stale; rerun "
                    "scripts/generate_baseline_notebooks.py",
                )

    def test_each_lane_installs_only_its_own_group(self) -> None:
        """The whole reason for the split: one optional group per runtime."""

        lanes, module = self._lanes()
        others = set(module.REQUIREMENT_GROUPS) - {module.CORE_LANE}
        for lane in lanes:
            source = self._source(lane.path)
            with self.subTest(lane=lane.slug):
                if lane.group == module.CORE_LANE:
                    self.assertNotIn("REQUIREMENT_GROUPS[", source)
                    continue
                self.assertIn(f"REQUIREMENT_GROUPS[{lane.group!r}]", source)
                for other in others - {lane.group}:
                    self.assertNotIn(f"REQUIREMENT_GROUPS[{other!r}]", source)

    def test_every_lane_model_needs_that_lane_group(self) -> None:
        lanes, module = self._lanes()
        for lane in lanes:
            for model in lane.models:
                with self.subTest(lane=lane.slug, model=model):
                    groups = module.groups_for(model, lane.task)
                    expected = groups[0] if groups else module.CORE_LANE
                    self.assertEqual(expected, lane.group)

    def test_lanes_partition_every_requested_cell_exactly_once(self) -> None:
        """No cell may be dropped between lanes, and none may be run twice."""

        lanes, module = self._lanes()
        for task in ("node_classification", "link_prediction", "graph_classification"):
            expected = {
                (model, dataset)
                for model, datasets in module.task_cells(task).items()
                for dataset in datasets
            }
            seen: list[tuple[str, str]] = []
            for lane in lanes:
                if lane.task != task:
                    continue
                seen += [(m, d) for m, ds in lane.cells for d in ds]
            with self.subTest(task=task):
                self.assertEqual(sorted(seen), sorted(set(seen)), "a cell is duplicated")
                self.assertEqual(set(seen), expected)

    def test_node_classification_runs_only_the_requested_subset(self) -> None:
        """HINormer on ACM alone; the other four on all three datasets."""

        lanes, module = self._lanes()
        seen = {
            model: sorted(datasets)
            for lane in lanes
            if lane.task == "node_classification"
            for model, datasets in lane.cells
        }
        self.assertEqual(
            seen,
            {
                "csnn": ["acm", "dblp", "imdb"],
                "nsd": ["acm", "dblp", "imdb"],
                "hinormer": ["acm"],
                "h2gformer": ["acm", "dblp", "imdb"],
                "latgrl": ["acm", "dblp", "imdb"],
            },
        )

    def test_the_dgl_lane_refuses_to_run_on_cpu(self) -> None:
        """A CPU-only DGL wastes the whole session; the lane must stop, not warn."""

        lanes, _ = self._lanes()
        dgl_lanes = [lane for lane in lanes if lane.group == "dgl"]
        self.assertTrue(dgl_lanes)
        for lane in dgl_lanes:
            source = self._source(lane.path)
            with self.subTest(lane=lane.slug):
                self.assertIn("dgl_cuda_report", source)
                self.assertIn("Refusing to run the DGL lane on CPU", source)
        for lane in lanes:
            if lane.group == "dgl":
                continue
            self.assertNotIn("dgl_cuda_report", self._source(lane.path))

    def test_no_lane_overrides_the_shared_training_budget(self) -> None:
        """`--epochs` reaches the training loops now, so a notebook must not set it."""

        lanes, _ = self._lanes()
        for lane in lanes:
            source = self._source(lane.path)
            with self.subTest(lane=lane.slug):
                self.assertNotIn("'--epochs'", source)
                self.assertNotIn("'--patience'", source)
                self.assertIn("from src.baselines.budget import default_budget", source)

    def test_every_lane_drives_the_entry_scripts_and_can_resume(self) -> None:
        lanes, _ = self._lanes()
        for lane in lanes:
            source = self._source(lane.path)
            with self.subTest(lane=lane.slug):
                for needed in (
                    "scripts/run_baseline_tuning.py",
                    "scripts/run_baseline_published.py",
                    "--validate_only",
                    "--resume",
                    "UNIT_START",
                    "UNIT_COUNT",
                    "WANDB_API_KEY",
                    "h2snn_baseline",
                    "models_missing_dependencies",
                    f"TASK = {lane.task!r}",
                ):
                    self.assertIn(needed, source)

    def test_notebooks_are_committed_clean_and_parse(self) -> None:
        import ast
        import json

        lanes, _ = self._lanes()
        for lane in lanes:
            payload = json.loads(lane.path.read_text(encoding="utf-8"))
            for index, cell in enumerate(payload["cells"]):
                with self.subTest(lane=lane.slug, cell=index):
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertFalse(cell.get("outputs"))
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]))

    def test_a_lane_declares_the_vendor_clones_its_models_read(self) -> None:
        lanes, module = self._lanes()
        for lane in lanes:
            source = self._source(lane.path)
            with self.subTest(lane=lane.slug):
                for vendor in lane.vendors:
                    self.assertIn(repr(vendor), source)


class _ModuleLevelNames(ast.NodeVisitor):
    """Names a cell reads, and names it binds, at module level only.

    Function and class bodies are skipped deliberately: they run later, so a global they
    reference may legitimately be defined by a cell further down. Everything at module
    level, though, runs the moment the cell does -- which is where a notebook that reads
    ``GROUP_OK`` in a lane that never assigns it goes wrong.
    """

    def __init__(self, bound: set[str]) -> None:
        self.bound = bound
        self.unbound: list[str] = []

    # Deferred scopes: bind the name, do not descend.
    def visit_FunctionDef(self, node: Any) -> None:
        self.bound.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: Any) -> None:
        self.bound.add(node.name)

    def visit_Lambda(self, node: Any) -> None:
        return

    # Evaluation order the default field order gets wrong.
    def visit_Assign(self, node: Any) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: Any) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_AnnAssign(self, node: Any) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_For(self, node: Any) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        for statement in node.body + node.orelse:
            self.visit(statement)

    def _comprehension(self, node: Any) -> None:
        # Comprehension variables are scoped to the comprehension, but binding them here
        # is harmless: the alternative is flagging every loop variable as unbound.
        for generator in node.generators:
            self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for field in ("elt", "key", "value"):
            child = getattr(node, field, None)
            if child is not None:
                self.visit(child)

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_DictComp = _comprehension

    def visit_ExceptHandler(self, node: Any) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self.bound.add(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_Import(self, node: Any) -> None:
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    visit_ImportFrom = visit_Import

    def visit_Name(self, node: Any) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.bound and not hasattr(builtins, node.id):
                self.unbound.append(node.id)
        else:
            self.bound.add(node.id)

    def visit_Global(self, node: Any) -> None:
        self.bound.update(node.names)


class NotebookNameResolutionTests(unittest.TestCase):
    """Every name a generated notebook reads must be bound by an earlier cell.

    ``ast.parse`` proves a cell is syntactically valid and nothing more. The core lanes
    installed no optional group, so they defined neither ``GROUP`` nor ``GROUP_OK``, and
    then the shared gate cell read both -- three notebooks that raised ``NameError`` on
    their eighth cell, with every syntax check passing.
    """

    #: Supplied by the runtime rather than by a cell.
    PROVIDED = frozenset({"get_ipython", "display", "In", "Out", "_", "__"})

    def _notebooks(self) -> list[Path]:
        paths = sorted((PROJECT_ROOT / "notebooks" / "baselines").glob("*.ipynb"))
        self.assertTrue(paths, "no generated lane notebooks found")
        return paths

    def test_no_cell_reads_a_name_no_earlier_cell_defines(self) -> None:
        import json

        for path in self._notebooks():
            payload = json.loads(path.read_text(encoding="utf-8"))
            bound: set[str] = set(self.PROVIDED)
            problems: list[str] = []
            for index, cell in enumerate(payload["cells"]):
                if cell["cell_type"] != "code":
                    continue
                visitor = _ModuleLevelNames(bound)
                visitor.visit(ast.parse("".join(cell["source"])))
                for name in visitor.unbound:
                    problems.append(f"cell {index}: {name}")
            with self.subTest(notebook=path.name):
                self.assertEqual(
                    problems,
                    [],
                    f"{path.name} reads names nothing defines yet: {problems}",
                )

    def test_the_checker_catches_the_bug_it_was_written_for(self) -> None:
        """A gate reading GROUP_OK with no install cell above it must be flagged."""

        visitor = _ModuleLevelNames(set())
        visitor.visit(ast.parse("if not GROUP_OK:\n    raise RuntimeError(GROUP.name)\n"))
        self.assertIn("GROUP_OK", visitor.unbound)
        self.assertIn("GROUP", visitor.unbound)

    def test_only_the_lanes_that_install_a_group_define_one(self) -> None:
        """A core lane neither defines nor reads the install-status variables.

        Read through the parsed code rather than the raw source: a comment naming
        ``GROUP_OK`` is prose, not a use of it, and failing on that would push the next
        person to reword a comment instead of to fix a notebook.
        """

        import json

        for path in self._notebooks():
            payload = json.loads(path.read_text(encoding="utf-8"))
            names: set[str] = set()
            for cell in payload["cells"]:
                if cell["cell_type"] != "code":
                    continue
                for node in ast.walk(ast.parse("".join(cell["source"]))):
                    if isinstance(node, ast.Name):
                        names.add(node.id)
            with self.subTest(notebook=path.name):
                if path.stem.endswith("_core"):
                    self.assertNotIn("GROUP_OK", names)
                    self.assertNotIn("GROUP", names)
                else:
                    self.assertIn("GROUP_OK", names)
                    self.assertIn("GROUP", names)
