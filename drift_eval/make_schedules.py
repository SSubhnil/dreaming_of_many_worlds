"""Write CWM schedules for one bench with the vendored generator's own CLI, unchanged.

    python -m drift_eval.make_schedules --bench walker_3f_phys --out-dir <dir> [--episodes 64]

The generator's `--bench` path resolves presets relative to its repository root (`_REPO_ROOT`,
the directory above benchmark/). A vendored copy has no CWM tree around it, so `_REPO_ROOT` is set to
the verified CWM root (its dmc.yaml, configure.yaml and latent_factor_values.yaml are hash-checked
in cwm.ensure_cwm and are byte-identical at the generator's commit befabde) and the generator's
`main()` then runs as written. The output directory gets the generator's own files (ir_s.json,
ir_a.json, h_s.json, h_a.json, tau_*.json, reward_fixed.json on mixed benches, bench_spec.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    from drift_eval import benches as B
    from drift_eval.cwm import (DEFAULT_GENERATOR, VENDORED_GENERATOR_SHA256, cwm_root,
                                ensure_cwm, load_generator)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--taus", default="1,8,16,64,256")
    ap.add_argument("--generator", default=None, help=f"default {DEFAULT_GENERATOR}")
    ap.add_argument("--generator-sha256", default=VENDORED_GENERATOR_SHA256)
    args = ap.parse_args(argv)

    bench = B.resolve_name(args.bench)
    ensure_cwm(verify=True)
    gen, sha = load_generator(args.generator, expected_sha256=args.generator_sha256)
    gen._REPO_ROOT = cwm_root()
    values = os.path.join(cwm_root(), "config_files", "latent_factor_values.yaml")
    old = sys.argv
    sys.argv = [gen.__drift_eval_path__, "--bench", bench, "--values", values,
                "--out-dir", args.out_dir, "--episodes", str(args.episodes), "--taus", args.taus]
    try:
        rc = gen.main()
    finally:
        sys.argv = old
    with open(os.path.join(args.out_dir, "bench_spec.json")) as f:
        spec = json.load(f)
    if spec.get("generator_sha256") != sha:
        raise RuntimeError(f"bench_spec.json records generator {spec.get('generator_sha256')}, "
                           f"loaded {sha}")
    print(f"[drift_eval] schedules for {bench} -> {args.out_dir} (generator sha256 {sha})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
