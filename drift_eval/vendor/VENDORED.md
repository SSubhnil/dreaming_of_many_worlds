drift_generator.py — vendored, unmodified

source repo:    CausalWorldModel (github.com/SSubhnil/CausalWorldModel)
source path:    benchmark/drift_generator.py
commit:         befabde38bfea2e9d150fa2da41facf738c49d59
sha256:         39b72cea67126f781605a2ebf7c658fa316ed380e9e2ee3badec132c561e09f6
vendored:       2026-09-14; `git show befabde:benchmark/drift_generator.py | sha256sum` equals the value
                above (checked 2026-09-14)

This copy is the default generator of drift_eval (drift_eval/cwm.py:load_generator), which asserts the
sha256 above at load and records the loaded sha256 in every output file. `--generator PATH` (or
DRIFT_GENERATOR) is an optional override that must carry the same bytes unless --generator-sha256 names
other ones. The generator resolves its bench presets relative to its repository root; for the vendored
copy drift_eval/make_schedules.py points that root at the verified CWM tree (45e8d5d, whose
config_files/experiments/dmc.yaml, configure.yaml and latent_factor_values.yaml are byte-identical to
befabde's) and then runs the generator's own main() unchanged.
