"""Part M3b B7: SeHGNN's and PSHGCN's own meta-path enumeration (`src/baselines/metapath_counts.py`) on a
synthetic schema with known counts (no data; the vendor clones are needed).

Schema P <-> A (relations P->A, A->P), target P:
* SeHGNN feature keys, 2 hops: P (raw), PA, PAP -- the walks of the schema ending at P of length <= 3;
* SeHGNN label keys, 2 label hops: PAP (label paths start and end at the target);
* PSHGCN, K = 3: lis = {1: [PA, AP], 2: [PAAP, APPA], 3: [PAAPPA, APPAAP]}.
A three-type schema P <-> A, P <-> S checks the growth: feature keys at 2 hops P, PA, PS, PAP, PSP.
"""

from __future__ import annotations

import unittest

from src.baselines.runtime.vendor_paths import available_vendors


@unittest.skipUnless(available_vendors().get("sehgnn") and available_vendors().get("pshgcn"),
                     "needs the SeHGNN and PSHGCN clones")
class MetapathCountTests(unittest.TestCase):
    def test_sehgnn_feature_keys(self) -> None:
        from src.baselines.metapath_counts import sehgnn_feature_keys

        self.assertEqual(sehgnn_feature_keys("PA", [("P", "A"), ("A", "P")], "P", 2), ["P", "PA", "PAP"])
        self.assertEqual(sehgnn_feature_keys("PAS", [("P", "A"), ("A", "P"), ("P", "S"), ("S", "P")], "P", 2),
                         ["P", "PA", "PAP", "PS", "PSP"])

    def test_sehgnn_label_keys(self) -> None:
        from src.baselines.metapath_counts import sehgnn_label_keys

        self.assertEqual(sehgnn_label_keys("PA", [("P", "A"), ("A", "P")], "P", 2), ["PAP"])

    def test_pshgcn_terms(self) -> None:
        from src.baselines.metapath_counts import pshgcn_terms

        lis, lis_t = pshgcn_terms(["PA", "AP"], 3)
        self.assertEqual(lis, {1: ["PA", "AP"], 2: ["PAAP", "APPA"], 3: ["PAAPPA", "APPAAP"]})
        self.assertEqual({k: len(v) for k, v in lis_t.items()}, {1: 2, 2: 2, 3: 2})


if __name__ == "__main__":
    unittest.main()
