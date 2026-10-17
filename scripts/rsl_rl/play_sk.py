# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""
import sk_franka.tasks # noqa: F401
import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_frontend_args,
    apply_video_recording,
    create_isaaclab_env,
    pre_launch_video_config,
    request_determinism,
    resolve_checkpoint_selector,
    resolve_play_task_name,
    show_run_summary,
    startup_screen,
)
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Length of each recorded video clip in env steps. Overrides the value in VideoRecorderCfg.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=None,
    help="Interval between video clips in env steps. Overrides the value in VideoRecorderCfg.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Play with the training environment configuration as-is, skipping play-mode overrides.",
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")

# checkpoint 1 and checkpoint 2
parser.add_argument("--checkpoint_1", type=str, required = True)
parser.add_argument("--checkpoint_2", type=str, required = True)
parser.add_argument("--episodes", type=int, default = None,) 


cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
args_cli.task = resolve_play_task_name(args_cli.task)

if args_cli.video:
    args_cli.enable_cameras = True
    

# an external callback lets downstream code register its environments; it returns
# the arguments it did not consume
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# hand the arguments consumed by neither this parser nor the callback over to Hydra
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")

# P1 - cube 2 na cube 1, P2 - cube 3 na cube 2, RELEASE 
P1, P2, RELEASE = 0, 1, 2


#### load policy =============================================================
    
def policy_load(env,agent_cfg,path):
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if version.parse(installed_version) >= version.parse("4.0.0"):
        reset_fn = policy.reset
    else:
        reset_fn = runner.alg.policy.reset
    return policy,reset_fn

### ==============================================================

@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    pre_launch_video_config(env_cfg, args_cli=args_cli)
    with startup_screen(args_cli, num_stages=3) as screen:
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
            # Warp reads its determinism mode at module build time, so request it before the env exists.
            request_determinism(args_cli, env_cfg)

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

            # note: certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
            env_cfg.episode_length_s = 30.0  
            

            # przekazanie checkpointow
            path_1 = retrieve_file_path(args_cli.checkpoint_1)
            path_2 = retrieve_file_path(args_cli.checkpoint_2)

            log_dir = os.path.dirname(path_1)

            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=path_1)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            print(f"[INFO]: Loading policy 1 from: {path_1}")
            policy_1, reset_1 = policy_load(env, agent_cfg, path_1)
            print(f"[INFO]: Loading policy 2 from: {path_2}")
            policy_2, reset_2 = policy_load(env, agent_cfg, path_2)
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            dt = env.unwrapped.step_dt

            base = env.unwrapped
            device = base.device
            num_envs = base.num_envs   
            robot = base.scene["robot"]  
            cube_1= base.scene["cube_1"]  
            cube_2= base.scene["cube_2"]  
            cube_3= base.scene["cube_3"]
            arm_ids, _ = robot.find_joints(["panda_joint.*"])
            arm_scale = base.cfg.actions.arm_action.scale

            
            from sk_franka.tasks.stack.mdp.terminations import cubes_stacked

            
            phase = torch.zeros(num_envs, dtype=torch.long, device=device)  #aktualna faza 
            cube_2_on_cube_1_steps = torch.zeros(num_envs, dtype=torch.long, device=device)  # ile krokow cube_2 stoi nieruchomo na cube_1
            tower_collapse_cnt= torch.zeros(num_envs, dtype=torch.long, device=device)  # ile korkow wieza zburzona
            release_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)  # ile krokow klocke jest puszczany 
            P1_comeback_cnt= torch.zeros(num_envs, dtype=torch.long, device=device)  # powroty do P1
            position_hold = torch.zeros(num_envs, len(arm_ids), device=device)  # pozycja ramienia trzymana podczas puszczania klocka
            success_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)  # ile krokow wieza z 3 kostek stoi nieruchomo
            success = torch.zeros(num_envs, dtype=torch.bool, device=device)  # sukces w tym epizodzie 

            finished = 0  # zakonczone epizody
            n_success = 0
            print("==================== OK ==================")
            screen.close()
            obs = env.get_observations()
            timestep = 0
            try:
                while True:
                    start_time = time.time()
                    with torch.inference_mode():
                        actions_1 = policy_1(obs)
                        actions_2 = policy_2(obs)
                        q_default = robot.data.default_joint_pos.torch[:, arm_ids]
                        arm_action = (position_hold - q_default) / arm_scale
                        gripper_open = torch.ones(num_envs, 1, device=device)
                        release_action = torch.cat([arm_action, gripper_open], dim=-1)

                        ph = phase.unsqueeze(-1)
                        actions = torch.where(ph == P1, actions_1, torch.where(ph == P2, actions_2, release_action))
                        obs, _, dones, _ = env.step(actions)
                        # reset recurrent states for episodes that have terminated
                        reset_1(dones)
                        reset_2(dones)
                        dones = dones.bool()
                        alive = ~dones

                        cube_2_on_cube_1 = cubes_stacked(base, cube_3_cfg = None, xy_threshold = 0.02, height_diff = 0.05, atol = 0.005, rtol = 0.0)
                        cube_2_speed = torch.linalg.norm(cube_2.data.root_lin_vel_w.torch, dim=1)

                        cube_2_still = cube_2_on_cube_1 & (cube_2_speed < 0.02) & (phase == P1) & alive
                        cube_2_on_cube_1_steps = torch.where(
                            cube_2_still, cube_2_on_cube_1_steps + 1, torch.zeros_like(cube_2_on_cube_1_steps)
                        )
                        arm_err = torch.linalg.norm(robot.data.joint_pos.torch[:, arm_ids] - q_default, dim=1)
                        switch = (phase == P1) & (cube_2_on_cube_1_steps >= 20) & (arm_err < 0.3) & alive
                        if switch.any():
                            phase = torch.where(switch, P2, phase)
                            cube_2_on_cube_1_steps = torch.where(
                                switch, torch.zeros_like(cube_2_on_cube_1_steps), cube_2_on_cube_1_steps
                            )

                        c2_c1_dist = cube_2.data.root_pos_w.torch - cube_1.data.root_pos_w.torch
                        tower_ok = (torch.linalg.norm(c2_c1_dist[:, :2], dim=1) < 0.04) & (
                            (c2_c1_dist[:, 2] - 0.05).abs() < 0.02
                        )
                        tower_collapse_cnt = torch.where(
                            (phase == P2) & alive & ~tower_ok, tower_collapse_cnt + 1, torch.zeros_like(tower_collapse_cnt)
                        )
                        collapse = (tower_collapse_cnt >= 10) & (P1_comeback_cnt < 2) & alive
                        if collapse.any():
                            position_hold[collapse] = robot.data.joint_pos.torch[:, arm_ids][collapse]
                            phase = torch.where(collapse, RELEASE, phase)
                            P1_comeback_cnt = torch.where(collapse, P1_comeback_cnt + 1, P1_comeback_cnt)
                            tower_collapse_cnt = torch.where(
                                collapse, torch.zeros_like(tower_collapse_cnt), tower_collapse_cnt
                            )

                        release_cnt = torch.where(
                            (phase == RELEASE) & alive, release_cnt + 1, torch.zeros_like(release_cnt)
                        )
                        back = (phase == RELEASE) & (release_cnt >= 25) & alive
                        if back.any():
                            phase = torch.where(back, P1, phase)

                        tower_3 = cubes_stacked(base, xy_threshold=0.02, height_diff=0.05, atol=0.005, rtol=0.0)
                        cube_3_speed = torch.linalg.norm(cube_3.data.root_lin_vel_w.torch, dim=1)
                        success_cnt = torch.where(
                            tower_3 & (cube_3_speed < 0.02) & alive, success_cnt + 1, torch.zeros_like(success_cnt)
                        )
                        new_success = (success_cnt >= 10) & alive & ~success
                        if new_success.any():
                            success |= new_success

                        if dones.any():
                            finished += int(dones.sum().item())
                            n_success += int(success[dones].sum().item())

                        phase[dones] = P1
                        cube_2_on_cube_1_steps[dones] =0
                        tower_collapse_cnt[dones] = 0
                        release_cnt[dones] = 0
                        P1_comeback_cnt[dones] = 0
                        success_cnt[dones] = 0
                        success[dones] = False

                    if args_cli.episodes is not None and finished >= args_cli.episodes:
                        break


                    if args_cli.video:
                        timestep += 1
                        video_stop = args_cli.video_length
                        if video_stop is None:
                            recorders = getattr(env_cfg, "video_recorders", [])
                            video_stop = recorders[0].video_length + recorders[0].step_offset if recorders else None
                        if video_stop is not None and timestep >= video_stop:
                            break

                    sleep_time = dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0:
                        time.sleep(sleep_time)

                env.close()
            except KeyboardInterrupt:
                pass

            if finished > 0:
                print("=" * 50)
                print(f"Epizody: {finished}")
                print(f"Sukces lacznie: {n_success}  ({100.0 * n_success / finished:.1f}%)")
                print("=" * 50)


if __name__ == "__main__":
    main()
