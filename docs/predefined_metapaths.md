# Predefined meta-paths per dataset (reference; not an input of H2S-NN)

**Purpose:** one place for every predefined meta-path list that touches this project — which source defines it, where in its code, and who uses it in our experiments (baselines, the fixed-literature-schema row, heterophily). H2S-NN's method (WISP) uses none of them (spec R8; advisor decision 2026-11-12). Written 2026-11-13 from the vendored code (`vendor/HGB` `ca6fd5b`, `vendor/H2GB` `d3b49db`, `vendor/Hetero2Net`), each entry with file:line. Tags: VERIFIED-CODE unless marked.

Notation: letters are node types; `Pc P` = paper –cite→ paper, `Pr P` = paper –reference→ paper (ACM's two citation directions). A meta-path starting and ending at the target type is marked **T**.

## 1. Who uses which list (in this project)

| Use | List | Spec |
|---|---|---|
| **Fixed-literature-schema row** (ablation, NC) | **HGB's published table**, paths that start at the target type (§2, column "HGB table") — ACM, DBLP, IMDB, Freebase only | v1.19 (O-1 = (a)) |
| Heterophily MLH, MDE | Hetero²Net's list where it has one, else `get_metapaths` (§2, column "Hetero²Net") | v1.14–1.15 §1 |
| Heterophily ℋ² | H2GB's rule: every length-2 type sequence target → target (`calcHomophily.extract_metapath`), not a fixed list | v1.15 §1 |
| HAN baseline (NC, HGB datasets) | HGB's HAN code lists (§2, column "HAN in HGB"), read through HGB's loader by our adapter (`src/baselines/adapters/hgb_han.py:150-161`) | H9 (original code) |
| H2G-former baseline | H2GB datasets: H2GB's MetaPath2Vec walk lists (§3) where H2GB's config enables them; other datasets: MetaPath2Vec on/off (tuned) with one cyclic schema = the HGB-table paths starting at the target type, concatenated in table order (as H2GB builds its schemas) | v1.19 T2 |
| WISP (H2S-NN) | none | R8 |

## 2. HGB datasets (node classification) and LastFM (link prediction)

| Dataset (target) | HGB table (published) | MAGNN in HGB code | HAN in HGB code | Hetero²Net |
|---|---|---|---|---|
| **ACM** (paper) | PAP, PSP, PcPAP, PcPSP, PrPAP, PrPSP (all T) | same six: relation ids `[2,3] [4,5] [0,2,3] [0,4,5] [1,2,3] [1,4,5]` (`NC/benchmark/methods/MAGNN/run_ACM.py:20`) | PcPSP `['pp','ps','sp']`, PrPSP `['-pp','ps','sp']`, PAP, PSP, **PTP** (`NC/benchmark/methods/HAN/utils.py:170`) | PAP, PSP, PTP (Appendix B.1; spec §1) |
| **DBLP** (author) | APA, APTPA, APVPA (T) | same: `[0,1] [0,2,3,1] [0,4,5,1]` (`run_DBLP.py:20`) | APA, APTPA, APCPA (C = venue) (`HAN/utils.py:284`) | APA |
| **IMDB** (movie) | target M: MDM, MAM (T); also DMD, DMAMD (target D), AMA, AMDMA (target A) | per target type: M `[0,1] [2,3]`; D `[1,0] [1,2,3,0]`; A `[3,2] [3,0,1,2]` (`run_IMDB.py:19-21`) | MDM, MAM, **MKM** (`HAN/utils.py:332`) | MDM, MAM, MKM |
| **Freebase** (book) | BB, BFB, BLMB, BPB, BPSB, BOFB, BUB (T) | 4 paths read from `meta1.json`/`meta2.json` (first two of `node_0` each: `-1`, `0`, `3,-11`, `4,24`), each closed by its reversal, ids shifted by 36 (`run_Freebase.py:20-27`); decoding to type letters UNVERIFIED | the same seven as the table, as relation-code pairs `['00','00'] ['01','10'] ['05','52','20'] ['04','40'] ['04','43','30'] ['06','61','10'] ['07','70']` (`HAN/utils.py:234`) | — (no list; `get_metapaths`) |
| **LastFM** (LP: user–artist) | UU, UAU, UATAU, AUA, ATA, AUUA | user side `(0,1,0) (0,1,2,1,0) (0,0)`; artist side `(1,0,1) (1,2,1) (1,0,0,1)` (`run_LastFM.py:19-30`) | — | — |

Notes: (i) the published table combines MAGNN's lists (ACM, DBLP, IMDB, LastFM) with HAN's Freebase list; HAN's ACM and IMDB lists in HGB's code differ from it (PTP, MKM). (ii) The source document of the table (HGB paper appendix) was seen as an image supplied by the author; its page was not read here (UNVERIFIED which table number). (iii) (Part O report, out-of-scope finding 3, 2026-11-13) MAGNN's numbers for DBLP and IMDB are MAGNN's internal edge-type ids, not HGB link ids (HGB's DBLP link 1 is paper–term, `data/NC/dblp/raw/DBLP/info.dat`), so `[0,1]` must not be read as HGB links; for ACM they coincide with HGB's link ids (`data/NC/acm/raw/ACM/info.dat`, Part O O2). Every list of this project is defined by type and relation names, not by these numbers. HAN's Freebase codes are decoded to relations in Part O O2 (`src/configs/config.py` `LITERATURE_METAPATHS`).

## 3. H2GB datasets (MetaPath2Vec walk lists, `H2GB/loader/encoding_generator.py`, `preprocess_Metapath`)

These are **cyclic walk schemas for MetaPath2Vec embeddings** used as a positional encoding (`H2GB/loader/master_loader.py:262-281`), not lists for aggregation; H2GB's HAN uses `HANConv` on the raw relations, not meta-paths (`H2GB/network/hetero_gnn_model.py:128`).

| Dataset | Walk schema | Code | Used by H2GB's H2G-former config? |
|---|---|---|---|
| oag-cs / oag-eng / oag-chem (`cs`, `engineering`, `chemistry`) | author→paper→field→paper→author→affiliation→author→paper→paper(cite)→author | `encoding_generator.py:228-238` | oag-chem yes; oag-cs, oag-eng no (`Hetero_Raw+Hetero_Label`) |
| ogbn-mag | author→paper→field_of_study→paper→author→institution→author→paper→paper(cites)→author | `:263-274` | yes |
| mag-year | **no branch** in `preprocess_Metapath` (UNVERIFIED whether its loader reports the name `ogbn-mag`) | — | config says yes (`configs/mag-year/mag-year-H2Gformer.yaml`) — to check when the H2GB data are prepared |
| IEEE-CIS-G | transaction→X→transaction for X ∈ {card1–card6, ProductCD, addr1, addr2, P_emaildomain, R_emaildomain} | `:392-422` | no (`Hetero_Raw+Hetero_Label`) |
| H-Pokec | user→user (friend), then user→X→user for 15 club types | `:302-341` | yes |
| PDNS | domain→ip→domain→domain (similar)→domain (apex) | `:423-435` | yes |

H2G-former's encoder stack per dataset (`configs/*/*-H2Gformer.yaml`, `gnn.node_encoder_name`): `Hetero_Raw+Hetero_Label+Hetero_Metapath` on ogbn-mag, mag-year, oag-chem, PDNS, Pokec; `Hetero_Raw+Hetero_Label` on IEEE-CIS, oag-cs, oag-eng, RCDD.

## 4. Graph classification
TU datasets (typed-by-label): no predefined meta-paths in HetSheaf or any source used here.

## 5. Open
- Decode MAGNN's Freebase relation ids to type letters (if the paper's baseline section needs MAGNN's Freebase list).
- mag-year's MetaPath2Vec branch (with the H2GB card).
