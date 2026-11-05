"""The hyperparameter axes the baselines share, in one place.

Each adapter owns its own search space -- the models genuinely differ -- but the axes they
have in common (learning rate, dropout, weight decay, width) were copied literal by
literal into fifteen files. Nothing then compared them against what the authors actually
published, and the grids drifted into a shape that could not express the settings the
baselines are known to work at:

* ``dropout`` was ``[0.0, 0.2, 0.5]`` while PSHGCN reports 0.25 on ACM, H2SGNN 0.7 on
  IMDB, and Hetero2Net 0.9 on IMDB.
* ``input_drop`` was the same three values while PSHGCN reports 0.85 on ACM and H2SGNN
  0.8 on DBLP.
* ``lr`` was a five-point grid that skipped PSHGCN's 0.004 and 0.006 and CSNN's 0.002.
* ``weight_decay`` had no 5e-4, which is what PSHGCN and H2SGNN report on IMDB.
* ``hidden`` started at 64 while H2SGNN reports 16 on ACM and 32 on DBLP.

A tuned cell that cannot reach its own published configuration is not a fair search: the
model is being reported at the best of a grid its authors would not have chosen. The
smooth axes are therefore continuous, which reaches every published value exactly and
lets the Bayes optimiser interpolate; the structural axes stay categorical, because
"1.7 layers" is not a model.

``tests/test_baselines.py: SearchSpaceCoverageTests`` asserts the property this module
exists for -- every published value is reachable by the corresponding search space -- so
a future edit to either side fails loudly.
"""

from __future__ import annotations

from typing import Any

#: Learning rate. Log-uniform over the range the baselines' published values occupy
#: (1e-4 .. 1e-2), which includes PSHGCN's 0.004/0.006 and CSNN's 0.002.
LR: dict[str, Any] = {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2}

#: Dropout on hidden activations. Uniform to 0.9, the largest published value
#: (Hetero2Net on IMDB); the old three-point grid could not express most of them.
DROPOUT: dict[str, Any] = {"distribution": "uniform", "min": 0.0, "max": 0.9}

#: Dropout on the input features. Same range: PSHGCN reports 0.85 on ACM.
INPUT_DROP: dict[str, Any] = {"distribution": "uniform", "min": 0.0, "max": 0.9}

#: Weight decay. Categorical rather than log-uniform because 0.0 is a value several of
#: these models publish, and a log scale cannot produce it.
WEIGHT_DECAY: dict[str, Any] = {"values": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]}

#: Hidden width. Extended downward to 16 and 32, which is where H2SGNN reports its ACM
#: and DBLP results.
HIDDEN: dict[str, Any] = {"values": [16, 32, 64, 128, 256]}

#: PSHGCN's separate learning rate for the polynomial propagation coefficients; their
#: published DBLP value is 0.002, between the old grid's 0.001 and 0.008.
PROP_LR: dict[str, Any] = {"distribution": "log_uniform_values", "min": 1e-3, "max": 5e-2}

#: H2SGNN's weight decay on the temperature list. Published ACM value is 0.4, between the
#: old grid's 0.3 and 1.0.
TEMP_WD: dict[str, Any] = {"distribution": "log_uniform_values", "min": 0.01, "max": 1.0}

#: HINormer's attention temperature. Their README runs 0.1/0.2/0.5/1/2 across datasets,
#: including 0.1 on DBLP-HGB, which the old ``[0.5, 1.0]`` grid excluded.
TEMPERATURE: dict[str, Any] = {"values": [0.1, 0.2, 0.5, 1.0, 2.0]}

#: HINormer's sampled sequence length. Their README runs 20/30/50/80/150, including 150
#: on IMDB-HGB, which the old ``[20, 50, 100]`` grid excluded.
LEN_SEQ: dict[str, Any] = {"values": [20, 30, 50, 80, 100, 150]}

#: Sheaf-model depth. Extended down to 1, which is what CSNN publishes on ENZYMES.
SHEAF_LAYERS: dict[str, Any] = {"values": [1, 2, 3, 4]}


__all__ = [
    "DROPOUT",
    "HIDDEN",
    "INPUT_DROP",
    "LEN_SEQ",
    "LR",
    "PROP_LR",
    "SHEAF_LAYERS",
    "TEMPERATURE",
    "TEMP_WD",
    "WEIGHT_DECAY",
]
