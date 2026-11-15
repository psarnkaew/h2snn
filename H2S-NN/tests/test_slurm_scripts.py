"""Cluster-setup CS1: the SLURM test script sets its limits and activates the right conda.

Reads `scripts/slurm/run_tests.sbatch` as text; nothing is submitted.
Cluster-setup3 CS8-3 (stated test change): the script activates `h2snn-b` (not `h2snn`), exports
`TORCH_EXTENSIONS_DIR` for h2snn-b and the three lines of the pinned CPU kernel environment E.
"""

import re
import unittest
from pathlib import Path

SBATCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "run_tests.sbatch"
MINIFORGE_LINE = ". ~/miniforge3/etc/profile.d/conda.sh"
PINNED_KERNEL_LINES = ("export ATEN_CPU_CAPABILITY=avx2", "export MKL_CBWR=AVX2",
                       "export ONEDNN_MAX_CPU_ISA=AVX2")
EXTENSIONS_LINE = "export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b"


def sbatch_problems(text: str) -> list[str]:
    """Every rule of the cluster-setup card that the script text breaks."""

    problems: list[str] = []
    directives = [line.strip() for line in text.splitlines() if line.strip().startswith("#SBATCH")]
    for option in ("--mem", "--time", "--cpus-per-task", "--gres"):
        if not any(re.match(rf"#SBATCH\s+{re.escape(option)}=\S+", d) for d in directives):
            problems.append(f"missing #SBATCH {option}")
    code = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not any(line.startswith(MINIFORGE_LINE) for line in code):
        problems.append("Miniforge is not initialised")
    if not any(re.search(r"conda activate h2snn-b(\s|$)", line) for line in code):
        problems.append("h2snn-b is not activated")
    for required in PINNED_KERNEL_LINES:
        if required not in code:
            problems.append(f"missing {required}")
    if EXTENSIONS_LINE not in code:
        problems.append("missing TORCH_EXTENSIONS_DIR for h2snn-b")
    if "/opt/conda" in "\n".join(code):
        problems.append("uses /opt/conda")
    if any(re.match(r"cd\s+(\$HOME|\$\{HOME\}|~)/?\s*($|;|&|\|)", line) for line in code):
        problems.append("runs in $HOME")
    if not any(re.match(r"cd\s+/mnt/data/peeradon/h2snn/H2S-NN\b", line) for line in code):
        problems.append("does not cd into the repository")
    return problems


class SbatchScriptTests(unittest.TestCase):
    def test_script_follows_cluster_rules(self) -> None:
        self.assertEqual(sbatch_problems(SBATCH_PATH.read_text(encoding="utf-8")), [])

    def test_checker_catches_each_violation(self) -> None:
        good = SBATCH_PATH.read_text(encoding="utf-8")
        no_time = "\n".join(l for l in good.splitlines() if not l.startswith("#SBATCH --time"))
        self.assertIn("missing #SBATCH --time", sbatch_problems(no_time))
        no_mem = "\n".join(l for l in good.splitlines() if not l.startswith("#SBATCH --mem"))
        self.assertIn("missing #SBATCH --mem", sbatch_problems(no_mem))
        opt = good.replace(MINIFORGE_LINE, ". /opt/conda/etc/profile.d/conda.sh")
        self.assertIn("uses /opt/conda", sbatch_problems(opt))
        self.assertIn("Miniforge is not initialised", sbatch_problems(opt))
        home = good + "\ncd $HOME\n"
        self.assertIn("runs in $HOME", sbatch_problems(home))
        self.assertIn("h2snn-b is not activated", sbatch_problems(good.replace("conda activate h2snn-b", "conda activate h2snn")))
        for required in PINNED_KERNEL_LINES:
            dropped = "\n".join(l for l in good.splitlines() if l.strip() != required)
            self.assertIn(f"missing {required}", sbatch_problems(dropped))
        no_ext = "\n".join(l for l in good.splitlines() if l.strip() != EXTENSIONS_LINE)
        self.assertIn("missing TORCH_EXTENSIONS_DIR for h2snn-b", sbatch_problems(no_ext))


if __name__ == "__main__":
    unittest.main()
