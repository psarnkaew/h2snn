"""Cluster-setup2 CS7-1a: the option-B lock and its install script, read as text (nothing is installed)."""

import re
import unittest
from pathlib import Path

from scripts.setup.check_env import parse_lock

ROOT = Path(__file__).resolve().parents[1]
LOCK_B = ROOT / "requirements" / "linux-lock-b.txt"
INSTALL_B = ROOT / "scripts" / "setup" / "install_env_b.sh"


def code_lines(text: str) -> list[str]:
    """Non-empty lines that are not comments."""

    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def lock_problems(text: str) -> list[str]:
    problems: list[str] = []
    if any("data.dgl.ai" in ln for ln in code_lines(text)):
        problems.append("lock has a data.dgl.ai line")
    dgl_lines = [ln for ln in code_lines(text) if re.match(r"\s*dgl\s*==", ln)]
    if len(dgl_lines) != 1:
        problems.append(f"expected exactly one dgl== line, found {len(dgl_lines)}")
    pins = parse_lock(text)
    if pins.get("hyper-connections") != "0.4.2":
        problems.append("hyper-connections==0.4.2 missing")
    if pins.get("torch") != "2.4.1":
        problems.append("torch==2.4.1 missing")
    if not re.search(r"sha256 [0-9a-f]{64}", text):
        problems.append("no sha256 of the DGL package in the lock")
    return problems


def script_problems(text: str) -> list[str]:
    problems: list[str] = []
    lines = code_lines(text)

    def first(pattern: str) -> int | None:
        for i, ln in enumerate(lines):
            if re.search(pattern, ln):
                return i
        return None

    # dgl is filtered from the lock before pip sees it; pip never reads the lock directly.
    filt = first(r"grep\s+-v\b.*dgl.*\$\{?LOCK\}?.*>\s*\"?\$\{?REQ\}?")
    if filt is None:
        problems.append("dgl is not filtered from the lock")
    pip_req = first(r"pip install.*-r\s+\"?\$\{?REQ\}?")
    if pip_req is None:
        problems.append("pip does not install the filtered requirements")
    elif filt is not None and pip_req < filt:
        problems.append("pip installs before dgl is filtered")
    if first(r"pip install.*-r\s+\"?\$\{?LOCK\}?") is not None:
        problems.append("pip installs the unfiltered lock")
    # sha256 is checked (with an exit) before conda install.
    sha = first(r"sha256sum")
    sha_exit = first(r"SHA_HAVE.*!=.*SHA_WANT")
    conda = first(r"^\s*conda install\b")
    if conda is None:
        problems.append("no conda install")
    else:
        if sha is None or sha_exit is None or not (sha < sha_exit < conda):
            problems.append("sha256 is not checked before conda install")
        elif not any("exit 1" in ln for ln in lines[sha_exit:conda]):
            problems.append("sha256 mismatch does not exit")
        if "--no-deps" not in lines[conda]:
            problems.append("conda install without --no-deps")
    # empty env name is refused before anything runs.
    refuse = first(r'\[\s+-z\s+"\$\{?ENV_NAME\}?"\s+\]')
    activate = first(r"conda activate")
    if refuse is None or activate is None or refuse > activate or not any("exit" in ln for ln in lines[refuse:refuse + 3]):
        problems.append("an empty env name is not refused")
    # h2snn is never a target.
    for ln in lines:
        if re.search(r"(-n|--name|activate|envs/)\s*\"?h2snn\"?(\s|$)", ln):
            problems.append(f"h2snn named as a target: {ln.strip()}")
    return problems


class LockBTests(unittest.TestCase):
    def test_repository_lock_b(self) -> None:
        self.assertEqual(lock_problems(LOCK_B.read_text(encoding="utf-8")), [])

    def test_checker_catches_lock_violations(self) -> None:
        good = LOCK_B.read_text(encoding="utf-8")
        cases = {
            "data.dgl.ai": good + "\n-f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html\n",
            "exactly one dgl (two)": good + "\ndgl==2.4.1\n",
            "exactly one dgl (none)": good.replace("\ndgl==2.4.0+cu124", "\n"),
            "hyper-connections": good.replace("hyper-connections==0.4.2", "hyper-connections==0.4.3"),
            "torch==2.4.1": good.replace("torch==2.4.1", "torch==2.4.0"),
        }
        for needle, text in cases.items():
            with self.subTest(needle):
                try:
                    problems = lock_problems(text)
                except ValueError as err:  # a duplicated pin is rejected by parse_lock
                    problems = [str(err)]
                self.assertTrue(problems, needle)


class InstallScriptBTests(unittest.TestCase):
    def test_repository_install_script(self) -> None:
        self.assertEqual(script_problems(INSTALL_B.read_text(encoding="utf-8")), [])

    def test_checker_catches_script_violations(self) -> None:
        good = INSTALL_B.read_text(encoding="utf-8")
        filt = next(ln for ln in good.splitlines() if ln.startswith("grep -v") and "dgl" in ln)
        sha_block = good[good.index('SHA_WANT="'):good.index('echo "== step 6')]
        cases = {
            "dgl not filtered": good.replace(filt, 'cp "${LOCK}" "${REQ}"'),
            "unfiltered lock": good.replace('-r "${REQ}"', '-r "${LOCK}"'),
            "sha256 removed": good.replace(sha_block, ""),
            "no --no-deps": good.replace("--no-deps ", ""),
            "empty name": good.replace('if [ -z "${ENV_NAME}" ]; then', 'if false; then'),
            "h2snn target": good.replace('conda activate "${ENV_NAME}"', "conda activate h2snn"),
        }
        for name, text in cases.items():
            with self.subTest(name):
                self.assertNotEqual(text, good, f"mutation {name!r} did not change the script")
                self.assertTrue(script_problems(text), name)


if __name__ == "__main__":
    unittest.main()
