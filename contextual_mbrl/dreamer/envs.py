import copy
import itertools
import os
import pathlib as _pathlib
import random
import sys
from functools import partial as bind
from multiprocessing import current_process
from typing import Tuple

# Add CausalWorldModel to path for benchmark wrappers (atari_context, procgen_context, etc.)
# Supports CAUSAL_WORLD_MODEL_ROOT env var override for non-sibling layouts.
_cwm_root = _pathlib.Path(
    os.environ.get("CAUSAL_WORLD_MODEL_ROOT", "")
) if os.environ.get("CAUSAL_WORLD_MODEL_ROOT") else (
    _pathlib.Path(__file__).resolve().parents[2].parent / "CausalWorldModel"
)
if _cwm_root.exists() and str(_cwm_root) not in sys.path:
    sys.path.insert(0, str(_cwm_root))

import dreamerv3
import gymnasium as gym
import numpy as np
from carl.context.context_space import UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from carl.context.selection import AbstractSelector
from carl.envs.carl_env import CARLEnv
from carl.envs.dmc import CARLDmcWalkerEnv
try:
    from carl.envs.dmc import CARLDmcBallInCupEnv
except ImportError:
    CARLDmcBallInCupEnv = None  # Not available in carl-bench >= 1.1.0
try:
    from carl.envs.gymnasium.box2d import CARLBipedalWalker, CARLLunarLander
except (ImportError, ModuleNotFoundError):
    CARLBipedalWalker = CARLLunarLander = None  # Box2D optional
try:
    from carl.envs.gymnasium.classic_control import CARLCartPole, CARLPendulum
except ImportError:
    CARLCartPole = CARLPendulum = None
from carl.utils.types import Context, Contexts
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gymnasium
from gymnasium import Wrapper, spaces
try:
    from gymnasium.wrappers.time_limit import TimeLimit
except ModuleNotFoundError:
    from gymnasium.wrappers import TimeLimit

CARTPOLE_TRAIN_GRAVITY_RANGE = [4.9, 14.70]
CARTPOLE_TRAIN_LENGTH_RANGE = [0.35, 0.75]

PENDULUM_LENGTH_RANGE = [0.5, 1.5]
PENDULUM_TRAIN_MASS_RANGE = [0.5, 1.5]

WALKER_TRAIN_GRAVITY_RANGE = [4.9, 14.70]
WALKER_TRAIN_ACTUATOR_STRENGTH_RANGE = [0.5, 1.5]

BIPEDAL_WALKER_TRAIN_GRAVITY_RANGE = [-13.0, -7.0]
BIPEDAL_WALKER_TRAIN_SPEED_KNEE_RANGE = [4.0, 9.0]

BIPEDAL_WALKER_TRAIN_LEG_H_RANGE = [1.5, 2.5]
BIPEDAL_WALKER_TRAIN_LEG_W_RANGE = [0.45, 0.55]

LUNAR_LANDER_TRAIN_GRAVITY_RANGE = [-13.0, -7.0]
LUNAR_LANDER_MAIN_ENGINE_POWER_RANGE = [8.0, 17.0]


_TASK2CONTEXTS = {
    "classic_cartpole": [
        {
            "context": "gravity",
            "train_range": CARTPOLE_TRAIN_GRAVITY_RANGE,
            "interpolate_single": [4.9, 7.35, 9.8, 12.25, 14.7],
            "interpolate_double": [4.9, 9.8, 14.7],
            "extrapolate_single": [
                0.98,
                1.715,
                2.45,
                3.185,
                3.92,
                15.68,
                16.66,
                17.64,
                18.62,
                19.6,
            ],
            "extrapolate_double": [0.98, 2.45, 3.92, 15.68, 17.64, 19.6],
        },
        {
            "context": "length",
            "train_range": CARTPOLE_TRAIN_LENGTH_RANGE,
            "interpolate_single": [0.4, 0.5, 0.6, 0.7],
            "interpolate_double": [0.5, 0.7],
            "extrapolate_single": [
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ],
            "extrapolate_double": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        },
    ],
    "classic_pendulum": [
        {
            "context": "l",
            "train_range": PENDULUM_LENGTH_RANGE,
            "interpolate_single": [0.5, 0.75, 1.0, 1.25, 1.5],
            "interpolate_double": [0.5, 1.0, 1.5],
            "extrapolate_single": [
                0.1,
                0.2,
                0.3,
                0.4,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
            ],
            "extrapolate_double": [0.1, 0.3, 1.6, 1.8, 2.0],
        },
        {
            "context": "m",
            "train_range": PENDULUM_TRAIN_MASS_RANGE,
            "interpolate_single": [0.5, 0.75, 1.0, 1.25, 1.5],
            "interpolate_double": [0.5, 1.0, 1.5],
            "extrapolate_single": [
                0.1,
                0.2,
                0.3,
                0.4,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
            ],
            "extrapolate_double": [0.1, 0.3, 1.6, 1.8, 2.0],
        },
    ],
    "box2d_bipedal_walker": [
        {
            "context": "GRAVITY_Y",
            "train_range": BIPEDAL_WALKER_TRAIN_GRAVITY_RANGE,
            "interpolate_single": [-7.0, -8.0, -9, -10.0, -11.0, -12.0, -13.0],
            "interpolate_double": [-7.0, -10.0, -13.0],
            "extrapolate_single": [
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
                -15.0,
                -16.0,
                -17.0,
                -18.0,
                -19.0,
            ],
            "extrapolate_double": [-1.0, -3.0, -5.0, -15.0, -17.0, -19.0],
        },
        {
            "context": "SPEED_KNEE",
            "train_range": BIPEDAL_WALKER_TRAIN_SPEED_KNEE_RANGE,
            "interpolate_single": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "interpolate_double": [4.0, 6.0, 9.0],
            "extrapolate_single": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "extrapolate_double": [1.0, 3.0, 10.0, 12.0, 15.0],
        },
    ],
    "box2d_bipedal_walker_new": [
        {
            "context": "LEG_H",
            "train_range": BIPEDAL_WALKER_TRAIN_LEG_H_RANGE,
            "interpolate_single": [1.5, 1.7, 1.9, 2.0, 2.1, 2.3, 2.5],
            "interpolate_double": [1.5, 2.0, 2.5],
            "extrapolate_single": [0.75, 0.85, 0.95, 2.6, 2.7, 2.8, 2.9, 3.0],
            "extrapolate_double": [0.75, 0.95, 2.6, 2.8, 3.0],
        },
        {
            "context": "LEG_W",
            "train_range": BIPEDAL_WALKER_TRAIN_LEG_W_RANGE,
            "interpolate_single": [0.45, 0.475, 0.5, 0.525, 0.55],
            "interpolate_double": [0.45, 0.5, 0.55],
            "extrapolate_single": [0.3, 0.35, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8],
            "extrapolate_double": [0.3, 0.4, 0.6, 0.7, 0.8],
        },
    ],
    "box2d_lunar_lander": [
        {
            "context": "GRAVITY_Y",
            "train_range": LUNAR_LANDER_TRAIN_GRAVITY_RANGE,
            "interpolate_single": [-7.0, -8.0, -9, -10.0, -11.0, -12.0, -13.0],
            "interpolate_double": [-7.0, -10.0, -13.0],
            "extrapolate_single": [
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
                -15.0,
                -16.0,
                -17.0,
                -18.0,
                -19.0,
            ],
            "extrapolate_double": [-1.0, -3.0, -5.0, -15.0, -17.0, -19.0],
        },
        {
            "context": "MAIN_ENGINE_POWER",
            "train_range": LUNAR_LANDER_MAIN_ENGINE_POWER_RANGE,
            "interpolate_single": [8.0, 10.0, 13.0, 15.0, 17.0],
            "interpolate_double": [8.0, 13.0, 17.0],
            "extrapolate_single": [
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
            ],
            "extrapolate_double": [3.0, 5.0, 7.0, 18.0, 20.0, 22.0],
        },
    ],
    "dmc_walker": [
        {
            "context": "gravity",
            "train_range": WALKER_TRAIN_GRAVITY_RANGE,
            "interpolate_single": [4.9, 7.35, 9.81, 12.25, 14.7],
            "interpolate_double": [4.9, 9.81, 14.7],
            "extrapolate_single": [
                0.98,
                1.715,
                2.45,
                3.185,
                3.92,
                15.68,
                16.66,
                17.64,
                18.62,
                19.6,
            ],
            "extrapolate_double": [0.98, 2.45, 3.92, 15.68, 17.64, 19.6],
        },
        {
            "context": "actuator_strength",
            "train_range": WALKER_TRAIN_ACTUATOR_STRENGTH_RANGE,
            "interpolate_single": [0.5, 0.75, 1.0, 1.25, 1.5],
            "interpolate_double": [0.5, 1.0, 1.5],
            "extrapolate_single": [0.1, 0.2, 0.3, 0.4, 1.6, 1.7, 1.8, 1.9, 2.0],
            "extrapolate_double": [0.1, 0.3, 1.6, 1.8, 2.0],
        },
    ],
}

# Benchmark-aligned discrete context value sets for fair DRAMA comparison
_BENCHMARK_CONTEXTS = {
    "dmc_walker": {
        "gravity": {
            "train":         [7.848, 8.829, 9.810, 10.791, 11.772],
            "test_iid":      [8.339, 9.320, 10.301, 11.282],
            "test_ood_mild": [6.377, 6.867, 13.244, 13.734],
            "test_ood_hard": [4.905, 5.396, 15.696, 17.168],
        },
        "actuator_strength": {
            "train":         [0.75, 0.85, 1.00, 1.15, 1.25],
            "test_iid":      [0.80, 0.90, 1.10, 1.20],
            "test_ood_mild": [0.50, 0.60, 1.40, 1.50],
            "test_ood_hard": [0.35, 0.40, 1.75, 2.00],
        },
    },
}


if CARLBipedalWalker is not None:
    class CARLBipedalWalkerNew(CARLBipedalWalker):
        @classmethod
        def get_default_context(cls) -> Context:
            default_context = super().get_default_context()
            default_context["LEG_H"] = 2.0
            default_context["LEG_W"] = 0.5
            return default_context
else:
    CARLBipedalWalkerNew = None


_TASK2ENV = {
    "classic_cartpole": CARLCartPole,
    "dmc_walker": CARLDmcWalkerEnv,
    "classic_pendulum": CARLPendulum,
    "box2d_bipedal_walker": CARLBipedalWalker,
    "box2d_bipedal_walker_new": CARLBipedalWalkerNew,
    "box2d_lunar_lander": CARLLunarLander,
}


def make_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != "none":
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(embodied.wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != "none"))


def make_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    if suite == "carl":
        return make_carl_env(config, **overrides)
    elif suite == "atari" and hasattr(config.env, "atari") and "/" in str(config.env.atari.task_sequence):
        return make_atari_context_env(config, **overrides)
    elif suite == "procgen" and hasattr(config.env, "procgen") and "/" in str(config.env.procgen.task_sequence):
        return make_procgen_context_env(config, **overrides)
    else:
        return dreamerv3.train.make_env(config, **overrides)


def _parse_task_sequence(seq_str):
    """Parse 'mode_diff/mode_diff/...' string into [[mode, diff], ...]."""
    if isinstance(seq_str, list):
        return seq_str
    return [[int(x) for x in pair.split("_")] for pair in seq_str.split("/")]


def _parse_procgen_task_sequence(seq_str):
    """Parse 'start_num_mode/...' string into [[start, num, mode], ...]."""
    if isinstance(seq_str, list):
        return seq_str
    result = []
    for entry in seq_str.split("/"):
        parts = entry.split("_")
        result.append([int(parts[0]), int(parts[1]), parts[2]])
    return result


def make_atari_context_env(config, **overrides):
    """Create Atari env with CARL-like context features for benchmark."""
    from benchmark.wrappers.atari_context import AtariContextWrapper
    _, game = config.task.split("_", 1)
    task_seq = _parse_task_sequence(config.env.atari.task_sequence)
    try:
        _proc_id = int(current_process().name.split("-")[-1])
    except ValueError:
        _proc_id = 0
    seed = _proc_id + int(config.seed)
    switch_mode = getattr(config.env.atari, "switch_mode", "round_robin")
    atari_kwargs = {
        "repeat": getattr(config.env.atari, "repeat", 4),
        "size": tuple(getattr(config.env.atari, "size", (64, 64))),
        "gray": getattr(config.env.atari, "gray", True),
        "noops": getattr(config.env.atari, "noops", 0),
        "lives": getattr(config.env.atari, "lives", "unused"),
        "sticky": getattr(config.env.atari, "sticky", False),
        "actions": getattr(config.env.atari, "actions", "needed"),
        "length": getattr(config.env.atari, "length", 108000),
        "resize": getattr(config.env.atari, "resize", "opencv"),
    }
    env = AtariContextWrapper(
        game=game,
        task_sequence=task_seq,
        seed=seed,
        switch_mode=switch_mode,
        atari_kwargs=atari_kwargs,
    )
    # AtariContextWrapper outputs DreamerV3-native format, no FromGymnasium needed
    env = ResizeImage(env)
    return dreamerv3.wrap_env(env, config)


def make_procgen_context_env(config, **overrides):
    """Create Procgen env with CARL-like context features for benchmark."""
    from benchmark.wrappers.procgen_context import ProcgenContextWrapper
    _, game = config.task.split("_", 1)
    task_seq = _parse_procgen_task_sequence(config.env.procgen.task_sequence)
    try:
        _proc_id = int(current_process().name.split("-")[-1])
    except ValueError:
        _proc_id = 0
    seed = _proc_id + int(config.seed)
    switch_mode = getattr(config.env.procgen, "switch_mode", "round_robin")
    image_size = tuple(getattr(config.env.procgen, "size", (64, 64)))
    gray = getattr(config.env.procgen, "gray", False)
    env = ProcgenContextWrapper(
        game=game,
        task_sequence=task_seq,
        seed=seed,
        switch_mode=switch_mode,
        image_size=image_size,
        gray=gray,
    )
    # ProcgenContextWrapper outputs DreamerV3-native format, no FromGymnasium needed
    env = ResizeImage(env)
    return dreamerv3.wrap_env(env, config)


class NormalizeContextWrapper(Wrapper):
    _CLASS_TO_CTX_NORMALIZER = {
        CARLCartPole: {
            "gravity": CARTPOLE_TRAIN_GRAVITY_RANGE,
            "length": CARTPOLE_TRAIN_LENGTH_RANGE,
        },
        CARLDmcWalkerEnv: {
            "gravity": WALKER_TRAIN_GRAVITY_RANGE,
            "actuator_strength": WALKER_TRAIN_ACTUATOR_STRENGTH_RANGE,
        },
        CARLPendulum: {
            "l": PENDULUM_LENGTH_RANGE,
            "m": PENDULUM_TRAIN_MASS_RANGE,
        },
        CARLBipedalWalker: {
            "GRAVITY_Y": BIPEDAL_WALKER_TRAIN_GRAVITY_RANGE,
            "SPEED_KNEE": BIPEDAL_WALKER_TRAIN_SPEED_KNEE_RANGE,
        },
        CARLBipedalWalkerNew: {
            "LEG_H": BIPEDAL_WALKER_TRAIN_LEG_H_RANGE,
            "LEG_W": BIPEDAL_WALKER_TRAIN_LEG_W_RANGE,
        },
        CARLLunarLander: {
            "GRAVITY_Y": LUNAR_LANDER_TRAIN_GRAVITY_RANGE,
            "MAIN_ENGINE_POWER": LUNAR_LANDER_MAIN_ENGINE_POWER_RANGE,
        },
    }

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, CARLEnv)
        assert "context" in env.observation_space.keys()
        assert isinstance(env.observation_space["context"], spaces.Box)
        assert not env.obs_context_as_dict
        # Normalize context space to [0, 1]
        self.observation_space["context"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space["context"].shape,
            dtype=np.float32,
        )
        env_klass = type(env)
        self._train_low = np.array(
            [
                self._CLASS_TO_CTX_NORMALIZER[env_klass][k][0]
                for k in env.obs_context_features
            ]
        )
        self._train_high = np.array(
            [
                self._CLASS_TO_CTX_NORMALIZER[env_klass][k][1]
                for k in env.obs_context_features
            ]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if CARLLunarLander is not None and isinstance(self.env, CARLLunarLander):
            # Randomize the wind and torque start index in each episode
            # to make the environment more stochastic
            self.env.env.wind_idx = np.random.randint(-9999, 9999)
            self.env.env.torque_idx = np.random.randint(-9999, 9999)
        obs["context"] = self._normalize_context(obs["context"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["context"] = self._normalize_context(obs["context"])
        return obs, reward, terminated, truncated, info

    def _normalize_context(self, context):
        return (context - self._train_low) / (
            self._train_high - self._train_low
        ) * 2 - 1


class RandomizedRoundRobinSelector(AbstractSelector):
    """
    Round robin context selector.

    Iterate through all contexts and then start at the first again.
    """

    def __init__(self, seed: int, contexts: Contexts):
        super().__init__(contexts)
        self.rand = random.Random(seed)

    def _select(self) -> Tuple[Context, int]:
        if self.context_id is None:
            self.context_id = -1
        self.context_id = (self.context_id + 1) % len(self.contexts)
        if self.context_id == 0:
            self.rand.shuffle(self.contexts)
        context = self.contexts[self.contexts_keys[self.context_id]]
        return context, self.context_id


class ResizeImage(embodied.wrappers.ResizeImage):
    """Change interpolation to BILINEAR"""

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.BILINEAR)
        image = np.array(image)
        return image


def make_carl_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite

    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = {}

    if config.env.carl.context == "default":
        contexts = {0: env_cls.get_default_context()}
    elif "single" in config.env.carl.context:
        index = int(config.env.carl.context.split("_")[-1])
        context_name = _TASK2CONTEXTS[task][index]["context"]
        # since there is only one context we can only sample independent
        train_range = _TASK2CONTEXTS[task][index]["train_range"]
        ctx_dists = [
            UniformFloatContextFeature(context_name, train_range[0], train_range[1])
        ]
        # All workers sample same contexts based on experiment seed
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    elif "double_box" in config.env.carl.context:
        ctx_dists = []
        for index in range(2):
            context_name = _TASK2CONTEXTS[task][index]["context"]
            train_range = _TASK2CONTEXTS[task][index]["train_range"]
            ctx_dists.append(
                UniformFloatContextFeature(context_name, train_range[0], train_range[1])
            )
        # All workers sample same contexts based on experiment seed
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    elif "benchmark_discrete" in config.env.carl.context:
        # Benchmark mode: use discrete aligned value sets for fair comparison.
        # Format: "benchmark_discrete" or "benchmark_discrete_test_iid" etc.
        parts = config.env.carl.context.split("_", 2)
        split = parts[2] if len(parts) > 2 else "train"
        bm = _BENCHMARK_CONTEXTS[task]
        default_ctx = env_cls.get_default_context()
        # Generate all combinations of discrete values
        import itertools as _it
        factor_names = list(bm.keys())
        value_lists = [bm[f][split] for f in factor_names]
        for combo in _it.product(*value_lists):
            c = dict(default_ctx)
            for fname, val in zip(factor_names, combo):
                c[fname] = val
            contexts[len(contexts)] = c
    elif "specific" in config.env.carl.context:
        ctx_vals = {
            int(v.split("-")[0]): float(v.split("-")[-1])
            for v in config.env.carl.context.split("_")[1:]
        }
        specific_context = env_cls.get_default_context()
        for i, val in ctx_vals.items():
            context_name = _TASK2CONTEXTS[task][i]["context"]
            specific_context[context_name] = val
        contexts[0] = specific_context
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")

    return create_wrapped_carl_env(env_cls, contexts, config)


def gen_carl_val_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    env_cls: CARLEnv = _TASK2ENV[task]
    ctx_0_name = _TASK2CONTEXTS[task][0]["context"]
    ctx_1_name = _TASK2CONTEXTS[task][1]["context"]
    ctx_default = env_cls.get_default_context()
    ctx_0_default = ctx_default[ctx_0_name]
    ctx_1_default = ctx_default[ctx_1_name]
    contexts = []

    if "specific" in config.env.carl.context:
        ctx_vals = {
            int(v.split("-")[0]): float(v.split("-")[-1])
            for v in config.env.carl.context.split("_")[1:]
        }
        c = env_cls.get_default_context()
        changed = []
        for i, val in ctx_vals.items():
            context_name = _TASK2CONTEXTS[task][i]["context"]
            c[context_name] = val
            changed.append(context_name)
        contexts.append({"context": c, "changed": changed})
    else:

        ctx_0_single = (
            _TASK2CONTEXTS[task][0]["interpolate_single"]
            + _TASK2CONTEXTS[task][0]["extrapolate_single"]
        )
        assert ctx_0_default in ctx_0_single
        for v0 in ctx_0_single:
            c = env_cls.get_default_context()
            c[ctx_0_name] = v0
            changed = []
            if v0 != ctx_0_default:
                changed.append(ctx_0_name)
            contexts.append({"context": c, "changed": changed})

        ctx_1_single = (
            _TASK2CONTEXTS[task][1]["interpolate_single"]
            + _TASK2CONTEXTS[task][1]["extrapolate_single"]
        )
        assert ctx_1_default in ctx_1_single
        for v1 in ctx_1_single:
            c = env_cls.get_default_context()
            c[ctx_1_name] = v1
            if v1 == ctx_1_default:
                continue
            contexts.append({"context": c, "changed": [ctx_1_name]})

        ctx_0_double = (
            _TASK2CONTEXTS[task][0]["interpolate_double"]
            + _TASK2CONTEXTS[task][0]["extrapolate_double"]
        )
        assert ctx_0_default in ctx_0_double
        ctx_1_double = (
            _TASK2CONTEXTS[task][1]["interpolate_double"]
            + _TASK2CONTEXTS[task][1]["extrapolate_double"]
        )
        assert ctx_1_default in ctx_1_double
        for v0, v1 in itertools.product(ctx_0_double, ctx_1_double):
            c = env_cls.get_default_context()
            c[ctx_0_name] = v0
            c[ctx_1_name] = v1
            # We make sure that default context is already covered in interpolate single case
            if v0 == ctx_0_default or v1 == ctx_1_default:
                continue
            contexts.append({"context": c, "changed": [ctx_0_name, ctx_1_name]})

    for context_info in contexts:
        ctors = []
        for index in range(config.envs.amount):
            ctor = lambda: create_wrapped_carl_env(
                env_cls, contexts={0: context_info["context"]}, config=config
            )
            if config.envs.parallel != "none":
                ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
            if config.envs.restart:
                ctor = bind(embodied.wrappers.RestartOnException, ctor)
            ctors.append(ctor)
        envs = [ctor() for ctor in ctors]
        yield embodied.BatchEnv(
            envs, parallel=(config.envs.parallel != "none")
        ), context_info


def create_wrapped_carl_env(env_cls: CARLEnv, contexts, config):
    _, task = config.task.split("_", 1)
    # Only the context features that might change in training or evaluation are # added to the observation space
    context_features = [o["context"] for o in _TASK2CONTEXTS[task]]
    try:
        _proc_id = int(current_process().name.split("-")[-1])
    except ValueError:
        _proc_id = 0
    seed = _proc_id + int(config.seed)
    if task == "box2d_lunar_lander":
        env_cls = bind(
            env_cls,
            env=gym.make(
                "LunarLander-v2",
                enable_wind=True,
                continuous=True,
                render_mode="rgb_array",
            ),
        )
    env = env_cls(
        obs_context_as_dict=False,
        obs_context_features=context_features,
        context_selector=RandomizedRoundRobinSelector(seed, contexts),
        kwargs={"render_mode": "rgb_array"},
    )  # Replace this with your Gym env.
    # reset once for paranoia
    env.reset(seed=seed)

    if "dmc" in task:
        env.env.render_mode = "rgb_array"
    if task == "classic_cartpole":
        env.env.screen_width = 128
        env.env.screen_height = 128
    env = NormalizeContextWrapper(env)

    # Regime B: intra-episode context switching (benchmark only)
    if getattr(config.env.carl, "regime", "A") == "B":
        from benchmark.wrappers.carl_intra_episode import make_walker_regime_b_wrapper
        regime_b_split = getattr(config.env.carl, "regime_b_split", "train")
        regime_b_lambda = float(getattr(config.env.carl, "regime_b_lambda", 0.005))
        env = make_walker_regime_b_wrapper(
            env, seed=seed, lambda_switch=regime_b_lambda, split=regime_b_split,
        )

    if "classic" in task:
        env = TimeLimit(env, max_episode_steps=500)

    env = from_gymnasium.FromGymnasium(env, obs_key="obs")
    env = embodied.core.wrappers.RenderImage(env, key="image")
    env = ResizeImage(env)
    return dreamerv3.wrap_env(env, config)
