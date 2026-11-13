"""Cluster-setup CS1: the SLURM test script sets its limits and activates the right conda.

Reads `scripts/slurm/run_tests.sbatch` as text; nothing is submitted.
"""

import re
import unittest
from pathlib import Path

SBATCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "run_tests.sbatch"
MINIFORGE_LINE = ". ~/miniforge3/etc/profile.d/conda.sh"


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
    if not any("conda activate h2snn" in line for line in code):
        problems.append("h2snn is not activated")
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
        self.assertIn("h2snn is not activated", sbatch_problems(good.replace("conda activate h2snn", "conda activate base")))


if __name__ == "__main__":
    unittest.main()
