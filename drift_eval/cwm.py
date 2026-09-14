"""Where the CWM-owned pieces come from, and proof of which bytes were used.

* Generator: the VENDORED copy drift_eval/vendor/drift_generator.py (CWM commit befabde; see
  vendor/VENDORED.md) by default; `--generator PATH` / DRIFT_GENERATOR is an
  optional override. The sha256 is asserted at load and recorded in every output file. The generator
  is CWM's single source for schedules and is never re-implemented here.
* CWM root: `CAUSAL_WORLD_MODEL_ROOT` (default: the pinned clean tree /home/jovyan/m10_code/45e8d5d).
  The fork's contextual_mbrl/dreamer/envs.py reads the SAME variable at import time (envs.py:11-19)
  to import `benchmark.wrappers.carl_intra_episode`, so it must be set before that import. The files
  below are hash-checked against their 45e8d5d bytes; a mismatch fails closed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

PINNED_CWM_ROOT = "/home/jovyan/m10_code/45e8d5d"
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
DEFAULT_GENERATOR = os.path.join(VENDOR_DIR, "drift_generator.py")
VENDORED_GENERATOR_SHA256 = "39b72cea67126f781605a2ebf7c658fa316ed380e9e2ee3badec132c561e09f6"

# sha256 at CWM 45e8d5d (`sha256sum` in the pinned tree, 2026-09-14). The three config files are also
# byte-identical at befabde, the generator's commit (checked with git show befabde:<path> | sha256sum).
EXPECTED_SHA256 = {
    "benchmark/wrappers/carl_intra_episode.py":
        "f12f0f0ae4eb5c7c91ca3c4062ea862e54ebae277e3f8272d00633516aa5dd2f",
    "benchmark/wrappers/carl_reward_switch.py":
        "5d8e5bbbf6a7f817353d1c10568f477ec76d406f10727cb29f8929a1d9975a3c",
    "envs/dmc_reward_factors.py":
        "4a980159a03954a1b916b2aef82f5737b561d12b82c6a9f385a6bcdc7207a54c",
    "benchmark/switch_recovery.py":
        "09264a7871dae5afe3bc62cbcaf2005f96a334e36640443a542dce0058071e60",
    "config_files/latent_factor_values.yaml":
        "d5f751cd6e4c0b6b24090bf4bb0f90d302a3478779ed3a601dd9bf7edb3e3c1a",
    "config_files/experiments/dmc.yaml":
        "93e9112b81c1b2193fa0f43aabdfc50602b7c78b98b3591faef2dbaf02bd7920",
    "config_files/configure.yaml":
        "aac5bdda7c7fabe82047bedb3d4af34616a80bc72ace7c12a7e2f9794c920e4f",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cwm_root():
    return os.environ.get("CAUSAL_WORLD_MODEL_ROOT") or PINNED_CWM_ROOT


def ensure_cwm(verify=True):
    """Put the CWM root on sys.path (and in the env var envs.py reads) and verify its bytes.

    Returns {relative path: sha256} of the files checked. Raises on any mismatch, and if the
    wrapper module that actually gets imported does not live under this root.
    """
    root = os.path.abspath(cwm_root())
    os.environ["CAUSAL_WORLD_MODEL_ROOT"] = root
    if root not in sys.path:
        sys.path.insert(0, root)
    hashes = {}
    for rel, want in EXPECTED_SHA256.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            raise RuntimeError(f"[drift_eval] CWM root {root} has no {rel}")
        got = sha256_file(p)
        hashes[rel] = got
        if verify and got != want:
            raise RuntimeError(
                f"[drift_eval] {rel} under {root} has sha256 {got}, expected the 45e8d5d bytes "
                f"{want}. Refusing: the factor semantics this port was checked against differ.")
    import benchmark.wrappers.carl_intra_episode as ci
    import benchmark.wrappers.carl_reward_switch as cr
    for mod in (ci, cr):
        if not os.path.abspath(mod.__file__).startswith(root + os.sep):
            raise RuntimeError(
                f"[drift_eval] {mod.__name__} resolved to {mod.__file__}, outside the CWM root "
                f"{root}. Another CWM tree is earlier on sys.path; refusing.")
    return hashes


CONFIG_FILES = ("config_files/experiments/dmc.yaml", "config_files/configure.yaml",
                "config_files/latent_factor_values.yaml")


def cwm_provenance():
    """What every output records about the (external, not vendored) CWM tree it read: root, commit,
    dirty-file count, and the sha256 of the three config files the generator and wrappers read."""
    import subprocess
    root = os.path.abspath(cwm_root())
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = len([l for l in subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().splitlines() if l.strip()])
    except Exception:
        commit, dirty = "not a git checkout", None
    return {"cwm_root": root, "cwm_commit": commit, "cwm_dirty_count": dirty,
            "cwm_config_sha256": {rel: sha256_file(os.path.join(root, rel)) for rel in CONFIG_FILES}}


def carl_modules():
    """The wrapper modules under the names the fork's envs.py imports them by.

    Module identity matters: `wrappers.carl_intra_episode` (benchmark/ on sys.path, as CWM's
    benchmark/eval_embodied.py:404 imports it) is a DIFFERENT module object from
    `benchmark.wrappers.carl_intra_episode` (envs.py:863-904), with its own schedule global.
    """
    ensure_cwm(verify=False)
    import benchmark.wrappers.carl_intra_episode as ci
    import benchmark.wrappers.carl_reward_switch as cr
    return ci, cr


def load_generator(path=None, expected_sha256=VENDORED_GENERATOR_SHA256):
    """Load CWM's generator module; assert its sha256; return (module, sha256).

    Default: the vendored copy. `expected_sha256=None` disables the assertion (never used by the
    runner or the launcher)."""
    path = os.path.abspath(path or os.environ.get("DRIFT_GENERATOR") or DEFAULT_GENERATOR)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[drift_eval] generator not found: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"[drift_eval] generator {path} has sha256 {digest}, expected "
                           f"{expected_sha256}. Refusing to generate or read schedules with it.")
    spec = importlib.util.spec_from_file_location(f"cwm_drift_generator_{digest[:12]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__drift_eval_path__ = path
    mod.__drift_eval_sha256__ = digest
    return mod, digest


def load_values_table():
    import yaml
    with open(os.path.join(cwm_root(), "config_files", "latent_factor_values.yaml")) as f:
        return yaml.safe_load(f)
