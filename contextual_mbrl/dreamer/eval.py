import csv
import logging
import os
import re
import warnings

import dreamerv3
import jsonlines
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied

from contextual_mbrl.dreamer.envs import gen_carl_val_envs

logging.captureWarnings(True)
os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def eval(policy, env, args, episodes=10):
    lengths = []
    rewards = []
    # Per-episode reward prediction tracking
    ep_reward_hats = {}   # worker_id -> list of reward_hat per step
    ep_reward_actuals = {}
    all_reward_mses = []

    def per_step(tran, worker):
        """Collect per-step reward predictions if available."""
        if "reward_hat" in tran:
            ep_reward_hats.setdefault(worker, []).append(
                float(tran["reward_hat"]))
            ep_reward_actuals.setdefault(worker, []).append(
                float(tran["reward"]))

    def per_episode(ep, worker):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        lengths.append(length)
        rewards.append(score)
        # Compute per-episode reward MSE if predictions were collected
        if worker in ep_reward_hats and len(ep_reward_hats[worker]) > 0:
            rh = np.array(ep_reward_hats[worker])
            ra = np.array(ep_reward_actuals[worker])
            mse = float(np.mean((rh - ra) ** 2))
            all_reward_mses.append(mse)
            ep_reward_hats[worker] = []
            ep_reward_actuals[worker] = []

    driver = embodied.Driver(env)
    driver.on_step(per_step)
    driver.on_episode(per_episode)

    print("Start evaluation loop.")
    driver(policy, episodes=episodes)
    metrics = {
        "length": np.mean(lengths).astype(float),
        "length_std": np.std(lengths).astype(float),
        "length_min": np.min(lengths).astype(float),
        "length_max": np.max(lengths).astype(float),
        "return": np.mean(rewards).astype(float),
        "return_std": np.std(rewards).astype(float),
        "return_min": np.min(rewards).astype(float),
        "return_max": np.max(rewards).astype(float),
        "returns": rewards,
        "lengths": lengths,
    }
    if all_reward_mses:
        metrics["reward_mse"] = float(np.mean(all_reward_mses))
        metrics["reward_mse_std"] = float(np.std(all_reward_mses))

    return metrics


def create_random_policy(act_space):
    def policy(*args):
        bs = args[0]["is_first"].shape[0]
        actions_space = act_space["action"]
        actions = np.stack([actions_space.sample() for _ in range(bs)])
        return {"action": actions}, None

    return policy


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")

    # create argparse with logdir
    parsed, other = embodied.Flags(logdir="", random_policy=False).parse_known()
    logdir = embodied.Path(parsed.logdir)
    is_random_policy = parsed.random_policy
    # load the config from the logdir
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config = embodied.Config(config)
    # parse the overrides for eval
    config = embodied.Flags(config).parse(other)

    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint
    config = config.update({"run.from_checkpoint": str(checkpoint)})

    # Just load the step counter from the checkpoint, as there is
    # a circular dependency to load the agent.
    step = 0
    if not is_random_policy:
        ckpt = embodied.Checkpoint()
        ckpt.step = embodied.Counter()
        ckpt.load(checkpoint, keys=["step"])
        step = ckpt._values["step"]

    policy = None
    returns = []
    lengths = []
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    for env, ctx_info in gen_carl_val_envs(config):

        if policy is None:
            if is_random_policy:
                policy = create_random_policy(env.act_space)
            else:
                agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
                checkpoint = embodied.Checkpoint()
                checkpoint.agent = agent
                checkpoint.load(args.from_checkpoint, keys=["agent"])
                policy = lambda *args: agent.policy(*args, mode="eval")
        metrics = eval(policy, env, args, episodes=50)
        env.close()
        returns.extend(metrics["returns"])
        lengths.extend(metrics["lengths"])
        metrics["ctx"] = {**ctx_info}
        metrics["aggregated_context_metric"] = False
        metrics["checkpoint_step"] = int(step)

        # Write metrics to eval.jsonl
        log_file = logdir / "eval.jsonl"
        if is_random_policy:
            log_file = logdir / "eval_random_policy.jsonl"

        with jsonlines.open(log_file, mode="a") as writer:
            writer.write(metrics)


if __name__ == "__main__":
    main()
