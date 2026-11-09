# J1.5 PubMed labels: not available, nothing computed (2026-11-09)
**Card:** check whether HGB PubMed's node-classification ids or names map onto the link-prediction release. Compute task-level `H_edge` of disease–disease only if every disease node matches. The author's instruction for this run: "PubMed node-classification labels are unavailable: J1.5 reports that and computes nothing."

**On disk** (VERIFIED-RUN, `ls`/`find`): `H2S-NN/data/PubMed/` holds only the link-prediction release, `node.dat`, `link.dat` and `link.dat.test`. There is no label file anywhere under `H2S-NN/data/` for PubMed.

**Why** (VERIFIED-SOURCE `H2S-NN/data/PROVENANCE.md:6`): the node-classification labels come from the HNE release (github.com/yangji9181/HNE, `Data/README.md`, Drive id `1Pkbl2wkwAXVRYrUWKpa1C4YQdjl_oIu2`). That release was not downloadable on 2026-11-09 (Drive 404). As instructed, nothing else was downloaded.

**Result:** no mapping was attempted, because there is nothing to map: matched fraction undefined, method none. Task-level `H_edge` (disease–disease) was **not computed**. The §1 heterophily measures need node labels, so for PubMed the paper states "not measurable" (no public label release) unless the labels are obtained.

**What would resolve it:** a copy of HNE's PubMed `label.dat`. The check would then be: map its node ids or names onto the link-prediction `node.dat` (disease type, id 1). Report the matched fraction. Compute `H_edge` with `scripts/heterophily/compute_heterophily.py`'s functions only if every disease node matches.
