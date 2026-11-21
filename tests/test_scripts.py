"""Tests für die Shell-Skripte in slurm/ und die Konfigurationen in configs/.

Die Jobs laufen erst auf dem Cluster. Hier wird geprüft, was ohne Slurm geht: die
Syntax, dass kein Name und kein Pfad einer Person darin steht, und das Verhalten
von common.sh und setup.sh mit nachgebauten Befehlen (scontrol, uv, curl).
Unter Windows fehlt bash, dort werden diese Tests übersprungen.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLURM = ROOT / "slurm"
CONFIGS = ROOT / "configs"

needs_bash = unittest.skipIf(
    os.name == "nt" or shutil.which("bash") is None, "braucht bash"
)

# Ein Pfad unter /home, ein @ außer in den Bash-Formen $@ und [@],
# oder ein Benutzername wie auf DACHS (of_ und das HSO-Kürzel)
PERSONAL = re.compile(r"/home/|(?<![$\[])@|\bof_[a-z]")

# Nachgebautes uv: protokolliert jeden Aufruf und schreibt beim Aufruf des
# Mini-Buchs das echte Mini-Buch. Die Datei läuft direkt, ohne das Paket und
# damit ohne torch zu importieren. Sonst lädt und installiert es nichts.
FAKE_UV = """#!/bin/bash
echo "UV_NO_DEV=${UV_NO_DEV:-} $*" >> "$CALL_LOG"
case "$1" in
  --version) echo "uv ${FAKE_UV_VERSION:-0.13.9} (fake)" ;;
  run)
    if [[ "$*" == "run python -m infm_sota2.data.mini_book "* ]]; then
      exec "$TEST_PYTHON" "$MINI_BOOK_FILE" "${@:5}"
    fi ;;
esac
"""

# Nachgebautes curl: protokolliert und scheitert, als gäbe es kein Netz
FAKE_CURL = """#!/bin/bash
echo "curl $*" >> "$CALL_LOG"
exit 1
"""


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def run_bash(
    args: list[str], env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


class ScriptFilesTest(unittest.TestCase):
    @needs_bash
    def test_shell_scripts_have_valid_syntax(self):
        scripts = sorted([*SLURM.glob("*.sh"), *SLURM.glob("*.sbatch")])
        self.assertTrue(scripts, f"no shell scripts in {SLURM}")
        for script in scripts:
            with self.subTest(script=script.name):
                result = run_bash(["-n", str(script)], dict(os.environ), ROOT)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_scripts_name_nobody(self):
        files = sorted(SLURM.iterdir())
        if CONFIGS.is_dir():
            files += sorted(CONFIGS.iterdir())
        for path in files:
            with self.subTest(file=str(path.relative_to(ROOT))):
                match = PERSONAL.search(path.read_text(encoding="utf-8"))
                self.assertIsNone(match, match and match.group())

    def test_setup_creates_logs_folder(self):
        setup = (SLURM / "setup.sh").read_text(encoding="utf-8")
        self.assertIn("mkdir -p logs", setup)


@needs_bash
class CommonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clone = Path(self.tmp.name)
        fake_bin = self.clone / "fake-bin"
        fake_bin.mkdir()
        # scontrol gibt die Knoten des Jobs aus, einen pro Zeile
        write_executable(
            fake_bin / "scontrol", "#!/bin/bash\nprintf 'node07\\nnode08\\n'\n"
        )
        self.env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.clone / "home"),
            "SLURM_JOB_ID": "123456",
            "SLURM_JOB_NODELIST": "node[07-08]",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def source_common(self) -> subprocess.CompletedProcess[str]:
        script = (
            f'set -euo pipefail; source "{SLURM / "common.sh"}"; '
            'echo "$UV_NO_DEV|$TIKTOKEN_CACHE_DIR|${VENV_TEST:-}'
            '|$MASTER_ADDR|$MASTER_PORT"'
        )
        return run_bash(["-c", script], self.env, self.clone)

    def test_common_sets_job_environment(self):
        venv = self.clone / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "activate").write_text("VENV_TEST=aktiv\n", encoding="utf-8")

        result = self.source_common()

        self.assertEqual(result.returncode, 0, result.stderr)
        cache = f"{self.env['HOME']}/.cache/tiktoken"
        # Port 20000 + 123456 % 20000
        self.assertEqual(result.stdout.strip(), f"1|{cache}|aktiv|node07|23456")

    def test_common_without_venv_points_to_setup(self):
        result = self.source_common()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("slurm/setup.sh", result.stderr)


@needs_bash
class SetupTest(unittest.TestCase):
    """setup.sh in einem nachgebauten Klon, aufgerufen von außerhalb des Klons."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.clone = self.base / "clone"
        (self.clone / "slurm").mkdir(parents=True)
        shutil.copy(SLURM / "setup.sh", self.clone / "slurm" / "setup.sh")
        fake_bin = self.base / "fake-bin"
        fake_bin.mkdir()
        write_executable(fake_bin / "uv", FAKE_UV)
        write_executable(fake_bin / "curl", FAKE_CURL)
        self.home = self.base / "home"
        self.home.mkdir()
        self.calls = self.base / "calls.txt"
        # Eigene Umgebung statt os.environ: W&B- und Datenvariablen der Person,
        # die die Tests ausführt, dürfen das Ergebnis nicht ändern
        self.env = {
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.home),
            "CALL_LOG": str(self.calls),
            "TEST_PYTHON": sys.executable,
            "MINI_BOOK_FILE": str(ROOT / "src/infm_sota2/data/mini_book.py"),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_setup(self, **extra: str) -> subprocess.CompletedProcess[str]:
        script = str(self.clone / "slurm" / "setup.sh")
        return run_bash([script], {**self.env, **extra}, self.base)

    def test_setup_without_book_is_ready_for_smoke_run(self):
        result = self.run_setup()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.clone / "data" / "mini-book.txt").is_file())
        self.assertTrue((self.clone / "logs").is_dir())
        calls = self.calls.read_text("utf-8")
        self.assertIn("UV_NO_DEV=1 sync --no-dev --locked", calls)
        self.assertIn("run python -m infm_sota2.data.mini_book data\n", calls)
        self.assertIn("scp", result.stdout)
        self.assertIn("download_weights", result.stdout)
        self.assertIn("export WANDB_MODE=disabled", result.stdout)
        self.assertIn("bereit für den Smoke-Lauf", result.stdout)

    def test_setup_with_book_weights_and_wandb_is_ready_for_all_runs(self):
        data = self.clone / "data"
        data.mkdir()
        (data / "The-lord-of-the-rings.txt").write_text("Platzhalter\n", "utf-8")
        (data / "gpt2-small-124M.pth").write_bytes(b"")
        netrc = "machine api.wandb.ai\n  login user\n  password key\n"
        (self.home / ".netrc").write_text(netrc, encoding="utf-8")

        result = self.run_setup()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("fehlt", result.stdout)
        self.assertIn("bereit für alle Läufe", result.stdout)

    def test_setup_uses_sota2_data_dir(self):
        shared = self.base / "shared-data"

        result = self.run_setup(SOTA2_DATA_DIR=str(shared), WANDB_MODE="disabled")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((shared / "mini-book.txt").is_file())
        self.assertFalse((self.clone / "data").exists())
        self.assertNotIn("W&B-Login", result.stdout)

    def test_setup_installs_uv_when_too_old(self):
        result = self.run_setup(FAKE_UV_VERSION="0.12.3")

        # Das nachgebaute curl scheitert, setup.sh bricht vor uv sync ab
        self.assertNotEqual(result.returncode, 0)
        calls = self.calls.read_text("utf-8")
        self.assertIn("curl -LsSf https://astral.sh/uv/install.sh", calls)
        self.assertNotIn("sync", calls)
