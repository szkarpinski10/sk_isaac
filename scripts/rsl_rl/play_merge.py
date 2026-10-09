# Laczy dwie polityki w jednym srodowisku (task sk_franka_merge):
#   faza 1: polityka 1 uklada cube_2 na cube_1
#   faza 2: gdy para stoi (chwytak otwarty, kostka nieruchoma ~hold_steps krokow) -> polityka 2 uklada cube_3 na cube_2
# Oparty na play_rsl_rl.py z Twojej wersji Isaac Lab (hydra_task_config, 255 linii).
# Uruchamiac BEZ --rl_library (ten plik sam jest backendem rsl_rl).
"""Chain two RSL-RL policies with an external supervisor and report success rates."""
 
import sk_franka.tasks  # noqa: F401  (rejestruje taski)
 
import warp as wp
 
wp.config.enable_backward = False
 
import argparse  # noqa: E402
import contextlib  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
 
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
 
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab.utils.string import list_intersection, string_to_callable  # noqa: E402
 
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.common import (  # noqa: E402
    add_frontend_args,
    create_isaaclab_env,
    request_determinism,
    resolve_play_task_name,
)
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
 
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
 
# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
 
# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chain two RSL-RL policies (cube_2 on cube_1, then cube_3 on cube_2).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--train_env_cfg",
    action="store_true",
    default=False,
    help="Play with the training environment configuration as-is, skipping play-mode overrides.",
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument("--checkpoint_1", required=True, help="Checkpoint polityki 1 (cube_2 na cube_1).")
parser.add_argument("--checkpoint_2", required=True, help="Checkpoint polityki 2 (cube_3 na cube_2).")
parser.add_argument("--episodes", type=int, default=100, help="Ile epizodow ocenic (po tylu konczy sie skrypt).")
parser.add_argument("--hold_steps", type=int, default=20, help="Ile krokow para musi stac nieruchomo przed przelaczeniem.")
parser.add_argument("--success_hold", type=int, default=10, help="Ile krokow wieza z 3 kostek musi stac, by uznac sukces.")
parser.add_argument("--max_speed", type=float, default=0.02, help="Prog predkosci kostki [m/s] uznawany za 'nieruchoma'.")
parser.add_argument(
    "--no_home", action="store_true", default=False, help="Pomin przejazd do pozycji domowej miedzy politykami."
)
parser.add_argument("--home_steps", type=int, default=60, help="Dlugosc plynnego przejazdu do pozycji domowej (w krokach; 50 krokow = 1 s).")
parser.add_argument("--home_max_steps", type=int, default=200, help="Maksymalna liczba krokow fazy powrotu (potem i tak startuje polityka 2).")
parser.add_argument("--home_tol", type=float, default=0.03, help="Tolerancja [rad] bledu stawow uznawana za 'jestem w domu'.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
args_cli.task = resolve_play_task_name(args_cli.task)
 
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()
 
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args
 
installed_version = metadata.version("rsl-rl-lib")
 
 
def _load_policy(env, agent_cfg, path: str):
    """Zwraca (policy, reset_fn) dla podanego checkpointu."""
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(path))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if version.parse(installed_version) >= version.parse("4.0.0"):
        reset_fn = policy.reset
    elif version.parse(installed_version) >= version.parse("2.3.0"):
        reset_fn = runner.alg.policy.reset
    else:
        reset_fn = runner.alg.actor_critic.reset
    return policy, reset_fn
 
 
@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        request_determinism(args_cli, env_cfg)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
 
        env_cfg.log_dir = os.path.dirname(retrieve_file_path(args_cli.checkpoint_1))
 
        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
        )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
 
        policy_1, reset_1 = _load_policy(env, agent_cfg, args_cli.checkpoint_1)
        policy_2, reset_2 = _load_policy(env, agent_cfg, args_cli.checkpoint_2)
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)
 
        # import po starcie symulacji (tak jak w env_cfg.py)
        from sk_franka.tasks.stack.mdp.terminations import cubes_stacked
 
        base = env.unwrapped
        device = base.device
        n = base.num_envs
 
        # fazy: 0 = polityka 1, 1 = przejazd do pozycji domowej, 2 = polityka 2
        phase = torch.zeros(n, dtype=torch.long, device=device)
        home_cnt = torch.zeros(n, dtype=torch.long, device=device)
        ep_step = torch.zeros(n, dtype=torch.long, device=device)
        robot = base.scene["robot"]
        arm_ids, _ = robot.find_joints(["panda_joint.*"])
        arm_scale = float(base.cfg.actions.arm_action.scale)
        q_start = torch.zeros(n, len(arm_ids), device=device)
        # statystyki czasu: srednia liczba krokow do przelaczenia i na powrot do domu
        sum_p0 = 0
        cnt_p0 = 0
        sum_home = 0
        cnt_home = 0
        hold_cnt = torch.zeros(n, dtype=torch.long, device=device)  # ile krokow pod rząd para stoi
        succ_cnt = torch.zeros(n, dtype=torch.long, device=device)  # ile krokow pod rząd stoją 3 kostki
        success = torch.zeros(n, dtype=torch.bool, device=device)  # sukces osiagniety w tym epizodzie
        handed = torch.zeros(n, dtype=torch.bool, device=device)  # w tym epizodzie nastapilo przelaczenie
 
        finished = 0
        n_handoff = 0
        n_success = 0
 
        obs = env.get_observations()
        try:
            with torch.inference_mode():
                while finished < args_cli.episodes:
                    a1 = policy_1(obs)
                    a2 = policy_2(obs)
                    q_default = robot.data.default_joint_pos.torch[:, arm_ids]
                    alpha = (home_cnt.float() / max(args_cli.home_steps, 1)).clamp(0.0, 1.0).unsqueeze(-1)
                    q_cmd = q_start + (q_default - q_start) * alpha
                    home_action = torch.cat(
                        [(q_cmd - q_default) / arm_scale, torch.ones(n, 1, device=device)], dim=-1
                    )
                    ph = phase.unsqueeze(-1)
                    actions = torch.where(ph == 0, a1, torch.where(ph == 1, home_action, a2))
                    obs, _, dones, _ = env.step(actions)
                    reset_1(dones)
                    reset_2(dones)
                    dones = dones.bool()
 
                    # Dla srodowisk, ktore wlasnie sie zresetowaly (dones), stan sceny jest juz nowy,
                    # wiec liczniki i flagi aktualizujemy tylko dla srodowisk 'zywych'.
                    # Wynik zakonczonego epizodu to flagi z poprzednich krokow.
                    alive = ~dones
                    ep_step = torch.where(alive, ep_step + 1, torch.zeros_like(ep_step))
 
                    pair = cubes_stacked(
                        base, cube_3_cfg=None, xy_threshold=0.02, height_diff=0.05, atol=0.005, rtol=0.0
                    )
                    tower = cubes_stacked(base, xy_threshold=0.02, height_diff=0.05, atol=0.005, rtol=0.0)
                    cube_2_speed = torch.linalg.norm(base.scene["cube_2"].data.root_lin_vel_w.torch, dim=1)
                    cube_3_speed = torch.linalg.norm(base.scene["cube_3"].data.root_lin_vel_w.torch, dim=1)
 
                    # faza 1: licznik 'para cube_2/cube_1 stoi nieruchomo z otwartym chwytakiem'
                    pair_still = pair & (cube_2_speed < args_cli.max_speed) & (phase == 0) & alive
                    hold_cnt = torch.where(pair_still, hold_cnt + 1, torch.zeros_like(hold_cnt))
                    switch = (phase == 0) & (hold_cnt >= args_cli.hold_steps) & alive
                    next_phase = 2 if args_cli.no_home else 1
                    phase = torch.where(switch, torch.full_like(phase, next_phase), phase)
                    handed |= switch
                    if switch.any():
                        q_start[switch] = robot.data.joint_pos.torch[:, arm_ids][switch]
                        sum_p0 += int(ep_step[switch].sum().item())
                        cnt_p0 += int(switch.sum().item())
 
                    # faza 1: przejazd do pozycji domowej, potem polityka 2
                    homing = (phase == 1) & alive
                    home_cnt = torch.where(homing, home_cnt + 1, torch.zeros_like(home_cnt))
                    q_err = (robot.data.joint_pos.torch[:, arm_ids] - robot.data.default_joint_pos.torch[:, arm_ids]).abs().amax(dim=1)
                    q_vel = robot.data.joint_vel.torch[:, arm_ids].abs().amax(dim=1)
                    at_home = (home_cnt >= args_cli.home_steps) & (q_err < args_cli.home_tol) & (q_vel < 0.05)
                    timeout_home = home_cnt >= args_cli.home_max_steps
                    go_p2 = homing & (at_home | timeout_home)
                    if go_p2.any():
                        sum_home += int(home_cnt[go_p2].sum().item())
                        cnt_home += int(go_p2.sum().item())
                    phase = torch.where(go_p2, torch.full_like(phase, 2), phase)
 
                    # sukces: 3 kostki w wiezy, chwytak otwarty, nieruchomo przez success_hold krokow
                    tower_still = (
                        tower & (cube_3_speed < args_cli.max_speed) & (cube_2_speed < args_cli.max_speed) & alive
                    )
                    succ_cnt = torch.where(tower_still, succ_cnt + 1, torch.zeros_like(succ_cnt))
                    success |= (succ_cnt >= args_cli.success_hold) & alive
 
                    # koniec epizodu: zapisz wynik i wyzeruj stan
                    if dones.any():
                        finished += int(dones.sum().item())
                        n_handoff += int(handed[dones].sum().item())
                        n_success += int(success[dones].sum().item())
                        phase[dones] = 0
                        ep_step[dones] = 0
                        home_cnt[dones] = 0
                        hold_cnt[dones] = 0
                        succ_cnt[dones] = 0
                        success[dones] = False
                        handed[dones] = False
                        print(
                            f"[chain] epizody: {finished}/{args_cli.episodes}  "
                            f"przelaczenia: {n_handoff}  sukcesy: {n_success}",
                            flush=True,
                        )
        except KeyboardInterrupt:
            pass
 
        print("=" * 70)
        print(f"Epizody ocenione:        {finished}")
        print(f"Przelaczenia 1 -> 2:     {n_handoff}  ({100.0 * n_handoff / max(finished, 1):.1f}%)")
        print(f"Pelny sukces (3 kostki): {n_success}  ({100.0 * n_success / max(finished, 1):.1f}%)")
        if cnt_p0:
            print(f"Sredni czas do przelaczenia (polityka 1): {sum_p0 / cnt_p0:.0f} krokow = {sum_p0 / cnt_p0 * 0.02:.1f} s")
        if cnt_home:
            print(f"Sredni czas powrotu do domu:               {sum_home / cnt_home:.0f} krokow = {sum_home / cnt_home * 0.02:.1f} s")
        print("=" * 70)
        env.close()
 
 
if __name__ == "__main__":
    main()