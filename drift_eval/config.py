"""Training config for a cRSSM checkpoint: the logdir's config.yaml, or the fork's own recipe.

Walker checkpoints on this PVC carry a config.yaml; the quadruped ones do not (HF `ssubhnil/cwm`
holds only checkpoint.ckpt for every cRSSM slug). The recipe is the fork's train.py:195-203
(defaults -> presets -> overrides), as in CWM benchmark/reconstruct_crssm_config.py, whose
docstring records the DALI round-trip that validates it and the limit of that validation.
Either way the checkpoint's tensor shapes are checked against the bench (context dim K, action and
observation dims) before anything is built.
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path

from drift_eval import benches as B

CRSSM_DIR = Path(__file__).resolve().parent.parent
STATIC_OVERRIDES = {"envs.parallel": "none", "wandb.project": ""}


def reconstruct(bench_name, logdir, seed):
    from drift_eval.stack import verify_stack
    verify_stack()
    import ruamel.yaml as yaml
    from dreamerv3 import embodied
    b = B.bench(bench_name)
    master = yaml.YAML(typ="safe").load(
        (CRSSM_DIR / "contextual_mbrl" / "dreamer" / "configs.yaml").read_text())
    cfg = embodied.Config(master["defaults"])
    for name in ["carl", b["preset"]]:
        cfg = cfg.update(master[name])
    return cfg.update({**STATIC_OVERRIDES, "logdir": str(logdir), "seed": int(seed)})


def load_config(bench_name, ckpt_dir, seed):
    """Return (embodied.Config, source) with source in {'logdir config.yaml', 'reconstructed'}."""
    from drift_eval.stack import verify_stack
    verify_stack()
    b = B.bench(bench_name)
    p = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(p):
        import ruamel.yaml as yaml
        from dreamerv3 import embodied
        with open(p) as f:
            raw = yaml.YAML(typ="safe").load(f)
        raw.setdefault("env", {}).setdefault("carl", {}).setdefault("regime_b_factor_splits", [""])
        cfg, source = embodied.Config(raw), "logdir config.yaml"
    else:
        cfg, source = reconstruct(bench_name, ckpt_dir, seed), "reconstructed"
    if cfg.task != b["task"] or cfg.env.carl.regime_b_factors != b["regime_b_factors"] \
            or cfg.env.carl.regime != "B":
        raise RuntimeError(f"[drift_eval] {source} for {ckpt_dir} is task={cfg.task} "
                           f"factors={cfg.env.carl.regime_b_factors}; bench {bench_name} needs "
                           f"{b['task']}/{b['regime_b_factors']}")
    return cfg, source


class _StubUnpickler(pickle.Unpickler):
    """Read the checkpoint pickle without jax/optax; unknown classes become inert stubs."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (object,), {"__module__": module,
                                          "__init__": lambda self, *a, **k: None,
                                          "_make": classmethod(lambda cls, it: cls.__new__(cls)),
                                          "__setstate__": lambda self, st: None})


def ckpt_signature(ckpt_file):
    with open(ckpt_file, "rb") as f:
        blob = _StubUnpickler(io.BufferedReader(f)).load()
    ag = blob["agent"]
    act = ag["agent/task_behavior/ac/actor/dist_out/out/kernel"].shape[1]
    img_in = ag["agent/wm/rssm/img_in/kernel"].shape[0]
    stoch = ag["agent/wm/rssm/img_stats/kernel"].shape[1]
    deter = ag["agent/wm/rssm/initial"].shape[0]
    head_in = ag["agent/wm/rew/h0/kernel"].shape[0]
    return {"step": int(blob["step"]), "act_dim": int(act), "ctx_dim": int(img_in - stoch - act),
            "head_ctx_dim": int(head_in - deter - stoch),
            "obs_dim": (int(ag["agent/wm/enc/mlp/h0/kernel"].shape[0])
                        if "agent/wm/enc/mlp/h0/kernel" in ag else None),
            "has_cnn": any("cnn" in k for k in ag)}


def verify_ckpt(bench_name, ckpt_file):
    b = B.bench(bench_name)
    sig = ckpt_signature(ckpt_file)
    bad = []
    for key, want in (("act_dim", b["act_dim"]), ("ctx_dim", b["ctx_dim"]),
                      ("head_ctx_dim", b["ctx_dim"]), ("obs_dim", b["obs_dim"])):
        if sig[key] != want:
            bad.append(f"{key}={sig[key]} expected {want}")
    if sig["has_cnn"]:
        bad.append("checkpoint has CNN tensors (pixels); cRSSM benches are featurized")
    if bad:
        raise RuntimeError(f"[drift_eval] {ckpt_file} does not fit bench {bench_name}: {bad}")
    return sig
