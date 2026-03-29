import sys
import os
import time
import warnings

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # EGL for headless GPU rendering (DreamerV3 default)
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ.get("MUJOCO_GL", "egl"))

# Import dm_control BEFORE JAX to avoid LLVM conflicts (OSMesa LLVM vs JAX LLVM)
import dm_control  # noqa: F401 — must be imported before JAX

import re
import warnings

import dreamerv3
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied

from contextual_mbrl.dreamer.envs import make_envs

warnings.filterwarnings("ignore")


def _get_context_factor_names(config):
    """Return ordered context factor names for Regime B runs, else None."""
    carl_cfg = getattr(getattr(config, "env", None), "carl", None)
    if carl_cfg is None or getattr(carl_cfg, "regime", "A") != "B":
        return None
    task = getattr(config, "task", "")
    factors = getattr(carl_cfg, "regime_b_factors", "K2")
    base = ["gravity", "joint_damping"] if "quadruped" in task else ["actuator_strength", "gravity"]
    if factors == "K3":
        base = sorted(base + ["wind_x"])
    return base


def train(agent, env, replay, logger, args, eval_env=None, factor_names=None):
    # copied from embodied.run.train and modified
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    should_eval = embodied.when.Every(args.eval_every) if eval_env is not None else None
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        if "context" in ep and factor_names is not None:
            ctx = ep["context"]  # (T+1, K)
            for i, fname in enumerate(factor_names):
                stats[f"gt_context_{fname}_mean"] = float(ctx[:, i].mean())
                stats[f"gt_context_{fname}_std"] = float(ctx[:, i].std())
        metrics.add(stats, prefix="stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)

    # Eval driver: separate env, greedy policy, not added to replay
    eval_metrics = embodied.Metrics()
    if eval_env is not None:
        eval_driver = embodied.Driver(eval_env)

        def per_eval_episode(ep):
            score = float(ep["reward"].astype(np.float64).sum())
            length = len(ep["reward"]) - 1
            eval_metrics.add({"score": score, "length": length}, prefix="eval")
            print(f"Eval episode: length={length}, return={score:.1f}.")

        eval_driver.on_episode(lambda ep, worker: per_eval_episode(ep))

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")
            if "priority" in outs:
                replay.prioritize(outs["key"], outs["priority"])
            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print("Start training loop.")
    policy = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    while step < args.steps:
        driver(policy, steps=100)
        if should_eval is not None and should_eval(step):
            eval_policy = lambda *a, **kw: agent.policy(*a, mode="eval")
            eval_driver(eval_policy, episodes=args.eval_eps)
            logger.add(eval_metrics.result())
            logger.write()
        if should_save(step):
            checkpoint.save()
    # save the final checkpoint
    checkpoint.save()
    if checkpoint._parallel:
        checkpoint._worker.shutdown(wait=True)
    logger.write()


def main():

    # import sys

    # add configurations
    # ontextual_mbrl.dreamer.train --configs carl enc_img_ctx_dec_img_ctx --task carl_classic_cartpole --env.carl.context default --seed 10 --logdir logs/$group_name/$seed --wandb.group $group_name --jax.policy_devices 0 --jax.train_devices 1 --run.steps 50000
    # group_name="carl_classic_cartpole_default_enc_img_ctx_dec_img_ctx"
    # seed=0
    # sys.argv[:1] = (
    #     f"--configs carl enc_img_ctx_dec_img_ctx --task carl_classic_cartpole --env.carl.context default --seed 10 --logdir logs/{group_name}/{seed} --wandb.group $group_name --jax.policy_devices 0 --jax.train_devices 1 --run.steps 50000"
    # ).split()
    # warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    # warnings.filterwarnings("once", ".*If you want to use these environments.*")

    parsed, other = embodied.Flags(configs=["defaults"]).parse_known()
    master_config = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "configs.yaml").read()
    )

    config = embodied.Config(master_config["defaults"])
    for name in parsed.configs:
        config = config.update(master_config[name])
    config = embodied.Flags(config).parse(other)
    logdir = embodied.Path(config.logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    step = embodied.Counter()

    loggers = [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
        # TensorBoardOutput triggers tf.summary which fails in this env.
        # Metrics available via JSONLOutput and WandB instead.
    ]
    if config.wandb.project != "":
        import wandb as _wandb
        loggers.append(
            embodied.logger.WandBOutput(
                ".*",
                dict(
                    **config.wandb,
                    name=logdir.name,
                    config=dict(config),
                    resume="allow",
                    dir=str(logdir),
                    settings=_wandb.Settings(init_timeout=300),
                ),
            )
        )

    logger = embodied.Logger(step, loggers)

    env = make_envs(config)
    eval_env = make_envs(config)  # same distribution as training for fair comparison
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, None
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    train(agent, env, replay, logger, args, eval_env=eval_env,
          factor_names=_get_context_factor_names(config))


if __name__ == "__main__":
    main()
