"""I5 mutations (pre-registered): each must make its targeted new test fail.
Run from H2S-NN/: python ../docs/WISP_evidence/i5_mutation_check.py"""
import subprocess
import sys
from pathlib import Path

T = "tests.test_wisp."
MUTATIONS = {
    "evaluation sees the training draw (trainer mutates the evaluation cells)": (
        "src/trainers/trainer.py",
        """        return replace(prepared_data,
                       face_groups=tuple(group.to(self.device) for group in groups))""",
        """        prepared_data.face_groups = tuple(group.to(self.device) for group in groups)
        return prepared_data""",
        T + "WispTrainerDrawTests"),
    "adaptive proposal silently dropped (q = p)": (
        "src/trainers/trainer.py",
        """        if (self.cfg.data.wisp_proposal == "adaptive" and self._previous_draw is not None""",
        """        if (False and self.cfg.data.wisp_proposal == "adaptive" and self._previous_draw is not None""",
        T + "WispTrainerDrawTests"),
    "gamma_init check only in from_dict (direct construction bypasses it)": (
        "src/configs/config.py",
        """        if self.path_weight == "exp" and self.gamma_init != 1.0:""",
        """        if False and self.path_weight == "exp" and self.gamma_init != 1.0:""",
        T + "WispConfigValidationTests.test_direct_construction_is_validated"),
    "wisp_* keys accepted under schemas": (
        "src/configs/config.py",
        """        if self.path_source == "schemas" and (
            self.wisp_max_length != 4""",
        """        if False and self.path_source == "schemas" and (
            self.wisp_max_length != 4""",
        T + "WispConfigValidationTests"),
    "a never-tuned key dropped (wisp_proposal)": (
        "src/configs/config.py",
        """                                           "wisp_proposal", "path_weight_context")""",
        """                                           "path_weight_context")""",
        T + "WispConfigValidationTests.test_every_never_tuned_wisp_key_is_rejected"),
    "IS ratio removed (transported T-H2 under a biased q)": (
        "src/data/wisp_sampler.py",
        """    return 2.0 / (1.0 + torch.exp(log_q - log_p))""",
        """    return torch.ones_like(log_p)""",
        T + "WispTransportedUnbiasednessTests.test_transported_TH2_under_a_biased_q_with_control"),
}

caught = 0
for name, (path, old, new, test) in MUTATIONS.items():
    file = Path(path)
    original = file.read_text(encoding="utf-8")
    assert original.count(old) == 1, name
    file.write_text(original.replace(old, new), encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
    finally:
        file.write_text(original, encoding="utf-8")
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})", flush=True)
clean = subprocess.run([sys.executable, "-m", "unittest", T + "WispTrainerDrawTests",
                        T + "WispConfigValidationTests", T + "WispTransportedUnbiasednessTests"],
                       capture_output=True, text=True)
print(f"caught {caught}/{len(MUTATIONS)}; clean after restore: {clean.stderr.strip().splitlines()[-1]}")
