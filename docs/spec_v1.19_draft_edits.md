# Spec v1.19 — draft edits (literature list of the comparison row; H2G-former's encoder), for the author's approval

**Status:** DRAFT, not applied. Written against spec **v1.18** on disk (md5 `c667b68a70523323bb564a9ed3e78791`). Re-read before applying; every Old string must occur exactly once; back up to `docs/spec_backups/H2S-NN_methodology_spec_v1.18.md`.
**Source:** author decisions of 2026-11-13 03:11 (O-1 = (a); no comparison row on H2GB datasets; reference file `docs/predefined_metapaths.md`; H2G-former must not be restricted to `Hetero_Raw`). **The method is unchanged.**
**Recommendation written in (author to confirm):** D-P3, H2G-former's encoder on datasets that have no H2GB config.

---

## T-1 §6 Ablation table — the list of the fixed-literature-schema row (O-1 = (a))
**Old:**
> List: Hetero²Net's predefined list, the same list as §1 (ACM PAP, PSP, PTP; DBLP APA; IMDB MDM, MAM, MKM; ogbn-mag PFP, PAP); the row exists only on node-classification datasets that have such a list.

**New:**
> List (1.19, author): **HGB's published meta-path table**, the paths that start at the target type: ACM PAP, PSP, PcPAP, PcPSP, PrPAP, PrPSP; DBLP APA, APTPA, APVPA; IMDB MDM, MAM; Freebase BB, BFB, BLMB, BPB, BPSB, BOFB, BUB. Relation-level definitions follow HGB's code (MAGNN `NC/benchmark/methods/MAGNN/run_ACM.py:20`, `run_DBLP.py:20`, `run_IMDB.py:19`; HAN `NC/benchmark/methods/HAN/utils.py:234` for Freebase; commit `ca6fd5b`), collected in `docs/predefined_metapaths.md`. The row exists on these four HGB datasets only; the H2GB datasets have no row (H2GB's predefined lists are MetaPath2Vec walk schemas for a positional encoding, not lists for aggregation). The heterophily statistics keep their own sets (§1).

## T-2 §6 Baselines — H2G-former's encoder as H2GB's configs
**Insert after:** `Hetero²Net stays (supervised; its masked terms are auxiliary losses).`
> (1.19, author) **H2G-former uses H2GB's own node-encoder stack** (`gnn.node_encoder_name` of `configs/<dataset>/<dataset>-H2Gformer.yaml`, H2GB commit `d3b49db`): `Hetero_Raw+Hetero_Label+Hetero_Metapath` on ogbn-mag, mag-year, oag-chem, PDNS and H-Pokec; `Hetero_Raw+Hetero_Label` on IEEE-CIS-G, oag-cs and oag-eng; the label channel's masking and the MetaPath2Vec walk lists (`H2GB/loader/encoding_generator.py`) as H2GB's code. The label channel may read training labels only (checked by an information-route test). **Datasets without an H2GB config (D-P3, recommended):** HGB node classification `Hetero_Raw+Hetero_Label` (the part common to every H2GB config; no MetaPath2Vec, because H2GB defines no walk list there); link prediction and graph classification `Hetero_Raw` (no node-class label channel in those tasks). Supersedes the adapter's restriction to `Hetero_Raw` (`src/baselines/adapters/h2gformer.py:19-25`).

---

## Proposed changelog entry (prepended above the 1.18 entry)
> **Changes from 1.18 (to 1.19; author decisions of 2026-11-13 in `docs/experiment_settings_decisions.md`, draft `docs/spec_v1.19_draft_edits.md`; the method is unchanged):** (T1) the fixed-literature-schema row uses HGB's published meta-path table (paths starting at the target type) on ACM, DBLP, IMDB and Freebase only; no row on H2GB datasets; every predefined list and its use is collected in `docs/predefined_metapaths.md`; (T2) H2G-former uses H2GB's own encoder stack per dataset (raw, label, MetaPath2Vec as its configs), with `Hetero_Raw+Hetero_Label` on HGB node classification and `Hetero_Raw` on LP and GC (D-P3).
