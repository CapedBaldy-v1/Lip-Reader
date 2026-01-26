"""
Setup script for Swin-VALLR with GPU-aware backend installation.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None, unit=None, leave=None):
        if iterable is None:
            class _NoOpTqdm:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self, n=1):
                    return None

                def close(self):
                    return None

            return _NoOpTqdm()
        return iterable

from logging_utils import setup_logging, log_system_info, log_exception
from artifact_utils import get_useful_dir, save_json, save_text


LOGGER = setup_logging("setup")
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
ENV_TEST_FILE = PROJECT_ROOT / "env_test.py"

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist"
}

GPU_PACKAGES = {"torch", "torchvision", "torchaudio"}


@dataclass
class RequirementSpec:
    name: str
    spec: Optional[str]
    raw: str


@dataclass
class EnvironmentCandidate:
    python_path: Path
    env_root: Path
    kind: str
    backend_ok: bool
    satisfied: int
    total: int
    missing: List[str]
    conflicts: List[str]
    details: Dict


def _run(cmd: List[str], check: bool = False, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    LOGGER.debug("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, env=env, capture_output=True, text=True)


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def _is_linux() -> bool:
    return platform.system().lower() == "linux"


def _read_requirements(path: Path) -> List[str]:
    entries: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def _parse_requirement(raw: str) -> RequirementSpec:
    try:
        from packaging.requirements import Requirement

        req = Requirement(raw)
        spec = str(req.specifier) if str(req.specifier) else None
        return RequirementSpec(name=req.name, spec=spec, raw=raw)
    except Exception:
        name = raw
        spec = None
        for op in [">=", "==", "<=", ">"]:
            if op in raw:
                name, spec = raw.split(op, 1)
                name = name.strip()
                spec = op + spec.strip()
                break
        if "[" in name:
            name = name.split("[", 1)[0].strip()
        return RequirementSpec(name=name, spec=spec, raw=raw)


def _version_key(version: str):
    try:
        from packaging.version import Version

        return Version(version)
    except Exception:
        parts = re.split(r"[^0-9]+", version)
        return tuple(int(p) for p in parts if p.isdigit())


def _version_satisfies(version: Optional[str], spec: Optional[str]) -> bool:
    if version is None or spec is None:
        return version is not None

    clauses = [c.strip() for c in spec.split(",") if c.strip()]
    v_key = _version_key(version)

    for clause in clauses:
        if clause.startswith(">="):
            if v_key < _version_key(clause[2:].strip()):
                return False
        elif clause.startswith("=="):
            if v_key != _version_key(clause[2:].strip()):
                return False
        elif clause.startswith("<="):
            if v_key > _version_key(clause[2:].strip()):
                return False
        elif clause.startswith(">"):
            if v_key <= _version_key(clause[1:].strip()):
                return False
    return True


def _detect_gpu_vendor() -> str:
    if _is_windows():
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        ]
        try:
            result = _run(cmd)
            if result.returncode != 0:
                LOGGER.warning("GPU detection failed on Windows.")
                return "unknown"
            names = result.stdout.lower()
        except Exception as exc:
            LOGGER.warning("GPU detection failed on Windows: %s", exc)
            return "unknown"
    elif _is_linux():
        names = ""
        if shutil.which("lspci"):
            result = _run(["lspci", "-nn"])
            names = result.stdout.lower()
        elif shutil.which("lshw"):
            result = _run(["lshw", "-C", "display"])
            names = result.stdout.lower()
        else:
            LOGGER.warning("No GPU detection tools available (lspci/lshw).")
            return "unknown"
    else:
        return "unknown"

    LOGGER.debug("GPU detection output: %s", names)
    if "nvidia" in names:
        return "nvidia"
    if "amd" in names or "radeon" in names or "ati" in names:
        return "amd"
    return "unknown"


def _find_python_executable(env_root: Path) -> Optional[Path]:
    if _is_windows():
        python_path = env_root / "Scripts" / "python.exe"
    else:
        python_path = env_root / "bin" / "python"
    if python_path.exists():
        return python_path
    return None


def _find_venv_roots(search_root: Path, max_depth: int = 3) -> List[Path]:
    envs: List[Path] = []
    def _onerror(err):
        LOGGER.warning("Skipping path during venv scan: %s", err)

    walk_iter = os.walk(search_root, onerror=_onerror)
    for root, dirs, files in tqdm(walk_iter, desc=f"Scanning {search_root}", unit="dir", leave=False):
        depth = Path(root).relative_to(search_root).parts
        if len(depth) > max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if "pyvenv.cfg" in files:
            envs.append(Path(root))
    return envs


def _list_candidate_envs() -> List[EnvironmentCandidate]:
    candidates: List[EnvironmentCandidate] = []
    system_python = Path(sys.executable).resolve()
    candidates.append(EnvironmentCandidate(
        python_path=system_python,
        env_root=system_python.parent,
        kind="system",
        backend_ok=False,
        satisfied=0,
        total=0,
        missing=[],
        conflicts=[],
        details={}
    ))

    search_roots = [PROJECT_ROOT]
    workon_home = os.environ.get("WORKON_HOME")
    if workon_home:
        search_roots.append(Path(workon_home))
    default_virtualenvs = Path.home() / ".virtualenvs"
    if default_virtualenvs.exists():
        search_roots.append(default_virtualenvs)

    seen = {system_python}

    for root in search_roots:
        if not root.exists():
            continue
        for env_root in _find_venv_roots(root):
            env_python = _find_python_executable(env_root)
            if env_python is None:
                continue
            env_python = env_python.resolve()
            if env_python in seen:
                continue
            seen.add(env_python)
            candidates.append(EnvironmentCandidate(
                python_path=env_python,
                env_root=env_root,
                kind="venv",
                backend_ok=False,
                satisfied=0,
                total=0,
                missing=[],
                conflicts=[],
                details={}
            ))

    LOGGER.info("Discovered %d Python environment candidates.", len(candidates))
    return candidates


def _inspect_environment(python_path: Path, package_names: List[str]) -> Optional[Dict]:
    script = r"""
import json
import sys

try:
    from importlib import metadata
except Exception:
    import importlib_metadata as metadata

result = {
    "packages": {},
    "torch": {},
    "directml": {}
}

names = json.loads(sys.argv[1])
for name in names:
    try:
        result["packages"][name] = metadata.version(name)
    except Exception:
        result["packages"][name] = None

try:
    import torch
    result["torch"]["version"] = torch.__version__
    result["torch"]["cuda_available"] = torch.cuda.is_available()
    result["torch"]["cuda_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    result["torch"]["cuda_version"] = torch.version.cuda
    result["torch"]["hip_version"] = torch.version.hip
except Exception as exc:
    result["torch"]["error"] = str(exc)

try:
    import torch_directml
    _ = torch_directml.device()
    result["directml"]["available"] = True
except Exception as exc:
    result["directml"]["available"] = False
    result["directml"]["error"] = str(exc)

print(json.dumps(result))
"""
    cmd = [str(python_path), "-c", script, json.dumps(package_names)]
    result = _run(cmd)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout.strip())
    except Exception:
        return None


def _evaluate_requirements(
    requirements: List[RequirementSpec],
    package_versions: Dict[str, Optional[str]]
) -> Tuple[int, int, List[str], List[str]]:
    satisfied = 0
    missing: List[str] = []
    conflicts: List[str] = []
    total = len(requirements)
    versions = {k.lower(): v for k, v in package_versions.items()}

    for req in requirements:
        installed_version = versions.get(req.name.lower())
        if installed_version is None:
            missing.append(req.raw)
            continue
        if req.spec and not _version_satisfies(installed_version, req.spec):
            conflicts.append(f"{req.raw} (installed {installed_version})")
            continue
        satisfied += 1

    return satisfied, total, missing, conflicts


def _backend_ok(details: Dict, expected_backend: str, gpu_vendor: str) -> bool:
    torch_info = details.get("torch", {})
    directml = details.get("directml", {})

    if expected_backend == "directml":
        return bool(directml.get("available"))

    if expected_backend == "cuda":
        if not torch_info.get("cuda_available"):
            return False
        name = (torch_info.get("cuda_name") or "").lower()
        return "nvidia" in name

    if expected_backend == "rocm":
        if not torch_info.get("cuda_available"):
            return False
        if not torch_info.get("hip_version"):
            return False
        return True

    return False


def _choose_best_env(candidates: List[EnvironmentCandidate]) -> Optional[EnvironmentCandidate]:
    if not candidates:
        return None
    candidates.sort(
        key=lambda c: (c.satisfied, -len(c.conflicts), c.kind == "venv"),
        reverse=True
    )
    return candidates[0]


def _create_venv(env_dir: Path) -> Path:
    print(f"[Setup] Creating virtual environment: {env_dir}")
    LOGGER.info("Creating virtual environment: %s", env_dir)
    _run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    python_path = _find_python_executable(env_dir)
    if python_path is None:
        raise RuntimeError("Failed to locate python in new venv.")
    _run([str(python_path), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    return python_path


def _pip_install(
    python_path: Path,
    args: List[str],
    use_user: bool = False,
    extra_index_url: Optional[str] = None
) -> None:
    cmd = [str(python_path), "-m", "pip", "install", "--upgrade"]
    if use_user:
        cmd.append("--user")
    if extra_index_url:
        cmd.extend(["--index-url", extra_index_url])
    cmd.extend(args)
    print(f"[Setup] Running: {' '.join(cmd)}")
    if len(args) == 1:
        desc = f"Installing {args[0]}"
    else:
        desc = "Installing packages"
    with tqdm(total=1, desc=desc, unit="step", leave=False) as progress:
        _run(cmd, check=True)
        progress.update(1)
    LOGGER.info("pip install completed: %s", " ".join(args))


def _parse_pip_check_output(output: str) -> List[Tuple[str, str]]:
    issues: List[Tuple[str, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("no broken requirements found"):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match:
            issues.append((match.group(1), line))
        else:
            issues.append(("", line))
    return issues


def _pip_check(python_path: Path, requirements: List[RequirementSpec]) -> None:
    cmd = [str(python_path), "-m", "pip", "check"]
    with tqdm(total=1, desc="Running pip check", unit="step", leave=False) as progress:
        result = _run(cmd)
        progress.update(1)
    if result.returncode == 0:
        LOGGER.info("pip check passed.")
        return

    output = "\n".join(filter(None, [result.stdout, result.stderr])).strip()
    issues = _parse_pip_check_output(output)
    if not issues:
        raise RuntimeError(f"pip check failed:\n{result.stdout}\n{result.stderr}")

    required = {req.name.lower() for req in requirements}
    blocking = [line for pkg, line in issues if pkg.lower() in required]
    ignored = [line for pkg, line in issues if pkg.lower() not in required]

    if blocking:
        raise RuntimeError("pip check failed for required packages:\n" + "\n".join(blocking))

    if ignored:
        LOGGER.warning("pip check found issues in unrelated packages:\n%s", "\n".join(ignored))
        print("[Setup] pip check found issues in unrelated packages; continuing:")
        for line in ignored:
            print(f"  - {line}")


def _read_os_release() -> Dict[str, str]:
    info: Dict[str, str] = {}
    if not Path("/etc/os-release").exists():
        return info
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip().strip('"')
    return info


def _rocm_installed() -> bool:
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return True
    return Path("/opt/rocm").exists()


def _install_rocm_system() -> None:
    if _rocm_installed():
        print("[Setup] ROCm is already installed.")
        LOGGER.info("ROCm already installed; skipping system install.")
        return

    if not _is_linux():
        raise RuntimeError("ROCm installation is only supported on Linux.")

    os_release = _read_os_release()
    distro = os_release.get("ID", "").lower()

    if shutil.which("apt-get") and distro in {"ubuntu", "debian"}:
        rocm_version = os.environ.get("ROCM_VERSION", "5.6")
        print(f"[Setup] Installing ROCm {rocm_version} via apt (this may take a while).")
        steps = [
            (["sudo", "apt-get", "update"], "apt-get update"),
            (["sudo", "apt-get", "install", "-y", "wget", "gnupg2", "ca-certificates"], "install prerequisites"),
            ([
                "sudo", "bash", "-c",
                "wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm.gpg"
            ], "add ROCm GPG key"),
            ([
                "sudo", "bash", "-c",
                f"echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/{rocm_version}/ {distro} main\" > /etc/apt/sources.list.d/rocm.list"
            ], "add ROCm apt repo"),
            (["sudo", "apt-get", "update"], "apt-get update (ROCm)"),
            (["sudo", "apt-get", "install", "-y", "rocm-dev", "rocm-utils"], "install ROCm"),
        ]
        with tqdm(total=len(steps), desc="Installing ROCm", unit="step", leave=False) as progress:
            for cmd, _label in steps:
                _run(cmd, check=True)
                progress.update(1)
    else:
        raise RuntimeError(
            "Automatic ROCm install not supported on this distro. "
            "Please install ROCm manually, then re-run setup.py."
        )


def _detect_rocm_index_url() -> str:
    rocm_version = None
    version_file = Path("/opt/rocm/.info/version")
    if version_file.exists():
        rocm_version = version_file.read_text(encoding="utf-8").strip()
    if rocm_version and rocm_version.startswith("6"):
        return "https://download.pytorch.org/whl/rocm6.0"
    return "https://download.pytorch.org/whl/rocm5.6"


def _detect_cuda_index_url() -> str:
    override = os.environ.get("CUDA_INDEX_URL")
    if override:
        return override

    if not shutil.which("nvidia-smi"):
        return "https://download.pytorch.org/whl/cu118"

    result = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if result.returncode != 0:
        return "https://download.pytorch.org/whl/cu118"

    match = re.search(r"(\d+)\.(\d+)", result.stdout.strip())
    if not match:
        return "https://download.pytorch.org/whl/cu118"

    major = int(match.group(1))
    minor = int(match.group(2))
    version = major + minor / 100.0

    if version >= 528:
        return "https://download.pytorch.org/whl/cu121"
    return "https://download.pytorch.org/whl/cu118"


def _install_base_requirements(python_path: Path, use_user: bool) -> None:
    requirements_raw = _read_requirements(REQUIREMENTS_FILE)
    base_requirements = [
        req for req in requirements_raw
        if _parse_requirement(req).name.lower() not in GPU_PACKAGES
    ]
    if not base_requirements:
        return
    _pip_install(python_path, base_requirements, use_user=use_user)


def _install_directml(python_path: Path, use_user: bool) -> None:
    _pip_install(python_path, ["torch-directml"], use_user=use_user)


def _install_cuda_pytorch(python_path: Path, use_user: bool) -> None:
    index_url = _detect_cuda_index_url()
    _pip_install(
        python_path,
        ["torch", "torchvision", "torchaudio"],
        use_user=use_user,
        extra_index_url=index_url
    )


def _install_rocm_pytorch(python_path: Path, use_user: bool) -> None:
    index_url = _detect_rocm_index_url()
    _pip_install(
        python_path,
        ["torch", "torchvision", "torchaudio"],
        use_user=use_user,
        extra_index_url=index_url
    )


def _run_env_test(python_path: Path) -> Dict:
    cmd = [str(python_path), str(ENV_TEST_FILE)]
    with tqdm(total=1, desc="Running environment test", unit="step", leave=False) as progress:
        result = _run(cmd)
        progress.update(1)
    output = result.stdout.strip()
    for line in output.splitlines():
        if line.startswith("ENV_TEST_RESULT:"):
            payload = line.split("ENV_TEST_RESULT:", 1)[1].strip()
            return json.loads(payload)
    raise RuntimeError(f"env_test.py did not return a valid result.\nOutput:\n{output}")


def _verify_env_result(result: Dict, expected_backend: str) -> bool:
    if not result.get("ok"):
        return False

    if expected_backend == "directml":
        return result.get("backend") == "directml"
    if expected_backend == "cuda":
        gpu_name = (result.get("gpu_name") or "").lower()
        return result.get("torch_cuda_available") and "nvidia" in gpu_name
    if expected_backend == "rocm":
        return result.get("torch_cuda_available") and result.get("torch_hip_version")

    return False


def _print_next_steps(python_path: Path, env_root: Optional[Path]) -> None:
    print("\n[Setup] Next steps:")
    if env_root is not None and _is_windows():
        activate_cmd = str(env_root / "Scripts" / "activate")
        print(f"  Activate environment: {activate_cmd}")
    elif env_root is not None and not _is_windows():
        activate_cmd = str(env_root / "bin" / "activate")
        print(f"  Activate environment: {activate_cmd}")
    print("  Run the demo: streamlit run app.py")


def _save_setup_artifacts(summary: Dict) -> None:
    try:
        root = get_useful_dir("setup", "env_summary")
        save_json(root / "summary.json", summary)
        save_text(root / "notes.txt", "Setup summary and environment diagnostics.")
        if REQUIREMENTS_FILE.exists():
            save_text(root / "requirements_snapshot.txt", REQUIREMENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        log_exception(LOGGER, "Failed to save setup artifacts.")


def _expected_backend(os_name: str, gpu_vendor: str) -> str:
    if _is_linux() and gpu_vendor == "amd":
        return "rocm"
    if _is_windows() and gpu_vendor == "amd":
        return "directml"
    if _is_windows() and gpu_vendor == "nvidia":
        return "cuda"
    raise RuntimeError("Unsupported OS/GPU combination for this setup script.")


def _collect_candidates_for_os() -> List[EnvironmentCandidate]:
    candidates = _list_candidate_envs()
    if _is_linux():
        return [c for c in candidates if c.kind == "system"]
    return candidates


def _score_candidates(
    candidates: List[EnvironmentCandidate],
    requirements: List[RequirementSpec],
    package_names: List[str],
    expected_backend: str,
    gpu_vendor: str
) -> List[EnvironmentCandidate]:
    evaluated: List[EnvironmentCandidate] = []
    for candidate in tqdm(candidates, desc="Inspecting environments", unit="env", leave=False):
        details = _inspect_environment(candidate.python_path, package_names)
        if details is None:
            continue
        satisfied, total, missing, conflicts = _evaluate_requirements(requirements, details.get("packages", {}))
        backend_ok = _backend_ok(details, expected_backend, gpu_vendor)

        evaluated.append(EnvironmentCandidate(
            python_path=candidate.python_path,
            env_root=candidate.env_root,
            kind=candidate.kind,
            backend_ok=backend_ok,
            satisfied=satisfied,
            total=total,
            missing=missing,
            conflicts=conflicts,
            details=details
        ))
        LOGGER.debug(
            "Env evaluated: python=%s backend_ok=%s satisfied=%s/%s conflicts=%s missing=%s",
            candidate.python_path, backend_ok, satisfied, total, len(conflicts), len(missing)
        )

    return evaluated


def _select_existing_environment(evaluated: List[EnvironmentCandidate]) -> Optional[EnvironmentCandidate]:
    usable = [candidate for candidate in evaluated if candidate.backend_ok]
    if not usable:
        LOGGER.info("No existing environment with required backend found.")
        return None
    chosen = _choose_best_env(usable)
    print(f"[Setup] Selected existing environment: {chosen.python_path}")
    LOGGER.info("Selected existing environment: %s", chosen.python_path)
    return chosen


def _install_for_backend(
    backend: str,
    python_path: Path,
    use_user: bool
) -> None:
    LOGGER.info("Installing dependencies for backend=%s (python=%s, user=%s).", backend, python_path, use_user)
    if backend == "rocm":
        _install_rocm_system()
        _install_rocm_pytorch(python_path, use_user=use_user)
        _install_base_requirements(python_path, use_user=use_user)
        return

    if backend == "directml":
        _install_directml(python_path, use_user=False)
        _install_base_requirements(python_path, use_user=False)
        return

    if backend == "cuda":
        _install_cuda_pytorch(python_path, use_user=False)
        _install_base_requirements(python_path, use_user=False)
        return

    raise RuntimeError("Unsupported backend.")


def _create_env_for_backend(backend: str) -> Tuple[Path, Optional[Path]]:
    if backend == "rocm":
        LOGGER.info("Using system python for ROCm backend (no venv).")
        python_path = Path(sys.executable).resolve()
        return python_path, None
    if backend == "directml":
        env_root = PROJECT_ROOT / ".venv-directml"
        LOGGER.info("Creating DirectML virtual environment at %s", env_root)
        python_path = _create_venv(env_root)
        return python_path, env_root
    if backend == "cuda":
        env_root = PROJECT_ROOT / ".venv-cuda"
        LOGGER.info("Creating CUDA virtual environment at %s", env_root)
        python_path = _create_venv(env_root)
        return python_path, env_root
    raise RuntimeError("Unsupported backend.")


def main() -> None:
    if not REQUIREMENTS_FILE.exists():
        raise RuntimeError("requirements.txt not found.")
    if not ENV_TEST_FILE.exists():
        raise RuntimeError("env_test.py not found.")

    log_system_info(LOGGER)

    os_name = platform.system().lower()
    gpu_vendor = _detect_gpu_vendor()

    print(f"[Setup] OS: {os_name}")
    print(f"[Setup] GPU vendor: {gpu_vendor}")
    LOGGER.info("Detected OS=%s GPU vendor=%s", os_name, gpu_vendor)

    expected_backend = _expected_backend(os_name, gpu_vendor)

    requirements_raw = _read_requirements(REQUIREMENTS_FILE)
    requirements = [_parse_requirement(r) for r in requirements_raw]
    package_names = [req.name for req in requirements]
    LOGGER.info("Loaded %d requirements from %s", len(requirements), REQUIREMENTS_FILE)

    candidates = _collect_candidates_for_os()
    evaluated = _score_candidates(candidates, requirements, package_names, expected_backend, gpu_vendor)
    chosen = _select_existing_environment(evaluated)

    LOGGER.info("Environment candidates evaluated: %d", len(evaluated))

    use_user = _is_linux() and os.geteuid() != 0

    if chosen is None:
        python_path, env_root = _create_env_for_backend(expected_backend)
        _install_for_backend(expected_backend, python_path, use_user)
    else:
        python_path = chosen.python_path
        env_root = chosen.env_root if chosen.kind == "venv" else None
        _install_for_backend(expected_backend, python_path, use_user)

    _pip_check(python_path, requirements)

    result = _run_env_test(python_path)
    LOGGER.info("env_test result: %s", result)
    if not _verify_env_result(result, expected_backend):
        raise RuntimeError(f"env_test.py failed validation: {result}")

    _save_setup_artifacts({
        "os": os_name,
        "gpu_vendor": gpu_vendor,
        "expected_backend": expected_backend,
        "python_path": str(python_path),
        "env_root": str(env_root) if env_root else None,
        "env_test": result
    })

    print("[Setup] Environment validation passed.")
    _print_next_steps(python_path, env_root)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_exception(LOGGER, "Setup failed with an unhandled exception.")
        raise
