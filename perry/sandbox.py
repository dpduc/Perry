"""The Sandbox: runs model-written Python in a throwaway Docker container.

Each run gets a fresh container from a fixed image (`sandbox/Dockerfile`) with no
network, CPU, memory and time limits, a read-only filesystem, no capabilities and no
environment from the host. The only host folder it sees is a per-run temp dir holding
the input data files; whatever the script saves there (PNG charts, CSV tables) comes
back, and the dir is deleted afterwards. It never sees `.env`, credentials, the project
files or the database.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DOCKERFILE_DIR = Path(__file__).resolve().parents[1] / "sandbox"

TIME_LIMIT_S = float(os.getenv("PERRY_SANDBOX_TIME_LIMIT_S", "60"))
MEMORY_LIMIT = os.getenv("PERRY_SANDBOX_MEMORY", "1g")
CPU_LIMIT = os.getenv("PERRY_SANDBOX_CPUS", "1")
LABEL = "perry.sandbox"
# The most output text handed back to the model.
OUTPUT_LIMIT = 20_000


class SandboxUnavailable(Exception):
    """Docker isn't running or the image can't be built."""


@dataclass(frozen=True)
class Result:
    output: str  # what the script printed
    error: str = ""  # why it failed, empty when it succeeded
    charts: list[bytes] = field(default_factory=list)  # PNG files it saved
    tables: dict[str, str] = field(default_factory=dict)  # CSV files it saved, by name

    @property
    def ok(self) -> bool:
        return not self.error


def image() -> str:
    """The image tag, which changes whenever the Dockerfile does."""
    # Line endings depend on the checkout, so they don't count.
    dockerfile = (DOCKERFILE_DIR / "Dockerfile").read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(dockerfile).hexdigest()
    return f"perry-sandbox:{digest[:12]}"


def docker_running() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False
    return info.returncode == 0


def running_containers() -> list[str]:
    """Names of Sandbox containers that still exist (normally none between runs)."""
    found = subprocess.run(
        ["docker", "ps", "--all", "--filter", f"label={LABEL}", "--format", "{{.Names}}"],
        capture_output=True, check=False,
        text=True,
    )
    return found.stdout.split()


def ensure_image() -> str:
    """Build the image unless it already exists. Returns its tag."""
    tag = image()
    if not docker_running():
        raise SandboxUnavailable("Docker isn't running, so Python can't be run.")
    inspect = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False)
    if inspect.returncode != 0:
        build = subprocess.run(
            ["docker", "build", "--tag", tag, str(DOCKERFILE_DIR)],
            capture_output=True, check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if build.returncode != 0:
            raise SandboxUnavailable(f"Building the sandbox image failed:\n{build.stderr[-2000:]}")
    return tag


def run(code: str, files: dict[str, bytes] | None = None, *, time_limit_s: float = TIME_LIMIT_S) -> Result:
    """Run `code` with `files` (name -> content) in its working directory."""
    tag = ensure_image()
    workdir = Path(tempfile.mkdtemp(prefix="perry-sandbox-"))
    try:
        inputs = set()
        for name, content in (files or {}).items():
            safe = Path(name).name
            (workdir / safe).write_bytes(content)
            inputs.add(safe)
        result = _run_container(tag, code, workdir, time_limit_s)
        return _collect(result, workdir, inputs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@dataclass(frozen=True)
class _Finished:
    output: str
    error: str


def _run_container(tag: str, code: str, workdir: Path, time_limit_s: float) -> _Finished:
    name = f"perry-sandbox-{uuid.uuid4().hex[:12]}"
    command = [
        "docker", "run", "--rm", "--interactive", "--name", name, "--label", LABEL,
        "--network", "none",
        "--cpus", CPU_LIMIT,
        "--memory", MEMORY_LIMIT, "--memory-swap", MEMORY_LIMIT,
        "--pids-limit", "128",
        "--read-only", "--tmpfs", "/tmp:size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--mount", f"type=bind,source={workdir},target=/work",
        tag,
    ]  # fmt: skip
    try:
        done = subprocess.run(command, input=code.encode(), capture_output=True, check=False, timeout=time_limit_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)
        return _Finished("", f"The script took longer than {time_limit_s:g} seconds and was stopped.")
    output = _text(done.stdout)
    if done.returncode == 0:
        return _Finished(output, "")
    if done.returncode == 137:
        return _Finished(output, f"The script ran out of memory (limit {MEMORY_LIMIT}) and was stopped.")
    stderr = _text(done.stderr).strip() or f"exit code {done.returncode}"
    return _Finished(output, f"The script failed:\n{stderr}")


def _collect(finished: _Finished, workdir: Path, inputs: set[str]) -> Result:
    charts, tables = [], {}
    for path in sorted(workdir.rglob("*")):
        if not path.is_file() or path.name in inputs:
            continue
        if path.suffix.lower() == ".png":
            charts.append(path.read_bytes())
        elif path.suffix.lower() == ".csv":
            tables[path.name] = _text(path.read_bytes())
    return Result(finished.output, finished.error, charts, tables)


def _text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if len(text) > OUTPUT_LIMIT:
        return text[:OUTPUT_LIMIT] + "\n(output cut short)"
    return text
