import sk_franka.tasks  # noqa: F401  (rejestruje taski)
 
import warp as wp
 
wp.config.enable_backward = False
 
import argparse  # noqa: E402
import contextlib  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
import math  # noqa: E402
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
parser.add_argument(
    "--episodes",
    type=int,
    default=100,
    help="Ile epizodow ocenic. Uzyj num_envs <= episodes/4, inaczej wynik jest przeklamany w strone krotkich epizodow.",
)
parser.add_argument("--hold_steps", type=int, default=20, help="Ile krokow para musi stac nieruchomo przed przelaczeniem 1 -> 2.")
parser.add_argument("--success_hold", type=int, default=10, help="Ile krokow wieza z 3 kostek musi stac, by uznac sukces.")
parser.add_argument("--max_speed", type=float, default=0.02, help="Prog predkosci kostki [m/s] uznawany za 'nieruchoma'.")
parser.add_argument(
    "--arm_home_tol",
    type=float,
    default=0.3,
    help="Jesli > 0: przelacz na polityke 2 dopiero gdy norma bledu stawow ramienia wzgledem pozycji bazowej "
    "jest mniejsza niz ta wartosc [rad] (jak w nagrodzie home_return). Szum startowy 0.1 na 7 stawow daje w treningu "
    "norme ok. 0.26, stad domyslne 0.3. 0 = wylaczone (polityka 2 dostaje ramie tam, gdzie je zostawila polityka 1).",
)
parser.add_argument(
    "--arm_wait_max",
    type=int,
    default=150,
    help="Przy arm_home_tol > 0: po tylu dodatkowych krokach przelacz mimo wszystko (zabezpieczenie przed zawieszeniem).",
)
parser.add_argument(
    "--clean_handoff",
    action="store_true",
    default=False,
    help="DIAGNOSTYKA: w chwili przelaczenia 1 -> 2 ustaw ramie dokladnie w pozie bazowej z zerowymi predkosciami "
    "i wyzeruj last_action (chwytak otwarty). Sprawdza, czy porazki wynikaja z obserwacji przy przejeciu.",
)
parser.add_argument(
    "--p2_yaw_rand",
    type=float,
    default=math.pi,
    help="Tylko z --p2_only: zakres losowego obrotu cube_2 wzgledem cube_1 [rad] (+-). Ustaw taki, jak w treningu polityki 2.",
)
parser.add_argument("--max_recoveries", type=int, default=2, help="Maks. liczba odzyskiwan po zburzeniu wiezy w jednym epizodzie.")
parser.add_argument("--knock_steps", type=int, default=10, help="Ile krokow pod rzad cube_2 musi byc poza cube_1, by uznac wieze za zburzona.")
parser.add_argument("--release_steps", type=int, default=25, help="Ile krokow trwa puszczenie cube_3 (otwarcie chwytaka) przed polityka 1.")
parser.add_argument(
    "--p2_only",
    action="store_true",
    default=False,
    help="Pomin polityke 1: wieze buduje event z treningu polityki 2, od razu dziala polityka 2 (pomiar bazowy, bez odzyskiwania).",
)
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
 
 
def _wilson(k: int, n: int, z: float = 1.96):
    """Przedzial ufnosci Wilsona 95% dla proporcji k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half
 
 
def _fmt(k: int, n: int) -> str:
    lo, hi = _wilson(k, n)
    return f"{k}/{n} = {100.0 * k / max(n, 1):.1f}%  (95% CI: {100.0 * lo:.1f}-{100.0 * hi:.1f}%)"
 
 
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
        dt = float(base.step_dt)
 
        robot = base.scene["robot"]
        arm_ids, _ = robot.find_joints(["panda_joint.*"])
        arm_scale = float(base.cfg.actions.arm_action.scale)
 
        if args_cli.p2_only:
            args_cli.max_recoveries = 0  # w trybie bazowym bez odzyskiwania
 
        def _rebuild_tower(env_ids):
            """Ustawia wieze cube_2 na cube_1 i cube_3 tak jak w treningu polityki 2."""
            from sk_franka.tasks.stack import mdp as stack_mdp
 
            stack_mdp.rand_tower_pos_and_cube_3(
                base,
                env_ids,
                min_separation=0.12,
                pose_range={"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.025, 0.025), "yaw": (-1.0, 1.0)},
                yaw_rand=args_cli.p2_yaw_rand,
            )
 
        # fazy: 0 = polityka 1, 1 = polityka 2, 2 = puszczenie cube_3 (odzyskiwanie)
        phase = torch.zeros(n, dtype=torch.long, device=device)
        ep_step = torch.zeros(n, dtype=torch.long, device=device)
        hold_cnt = torch.zeros(n, dtype=torch.long, device=device)  # ile krokow pod rzad para stoi
        succ_cnt = torch.zeros(n, dtype=torch.long, device=device)  # ile krokow pod rzad stoja 3 kostki
        bad_cnt = torch.zeros(n, dtype=torch.long, device=device)  # ile krokow pod rzad cube_2 jest poza cube_1 (faza 1)
        rel_cnt = torch.zeros(n, dtype=torch.long, device=device)  # licznik puszczenia
        recov = torch.zeros(n, dtype=torch.long, device=device)  # ile odzyskiwan w tym epizodzie
        succ_rec = torch.full((n,), -1, dtype=torch.long, device=device)  # ile odzyskiwan bylo w chwili sukcesu
        success = torch.zeros(n, dtype=torch.bool, device=device)
        handed = torch.zeros(n, dtype=torch.bool, device=device)  # w epizodzie byl choc raz przelaczenie 1 -> 2
        q_hold = torch.zeros(n, len(arm_ids), device=device)  # pozycja ramienia przytrzymana podczas puszczenia
        last_intact = torch.ones(n, dtype=torch.bool, device=device)  # czy w ostatnim kroku cube_2 lezala na cube_1
        c3_start = torch.zeros(n, 3, device=device)  # pozycja cube_3 na poczatku epizodu
        c3_last = torch.zeros(n, 3, device=device)  # ostatnia znana pozycja cube_3
        sw_err = torch.zeros(n, device=device)  # norma bledu stawow ramienia w chwili pierwszego przelaczenia
        sw_dq = torch.zeros(n, len(arm_ids), device=device)  # odchylenie kazdego stawu od pozycji bazowej przy przelaczeniu
        sw_fb = torch.zeros(n, dtype=torch.bool, device=device)  # przelaczono przez zabezpieczenie arm_wait_max
        grp_err = {"succ": [], "idle": [], "other": []}
        grp_dq = {"succ": [], "idle": [], "other": []}
        grp_fb = {"succ": [], "idle": [], "other": []}
        sw_ang = torch.zeros(n, device=device)  # kat obrotu cube_2 wzgledem cube_1 [rad] w chwili pierwszego przelaczenia
        sw_tilt = torch.zeros(n, device=device)  # przechyl cube_2 (kat osi z kostki od pionu) [rad] przy przelaczeniu
        tilt_succ, tilt_idle, tilt_other = [], [], []
        ang_succ, ang_idle, ang_other = [], [], []  # kat dla: sukcesow / bezczynnych porazek / innych porazek
        min_d = torch.full((n,), 10.0, device=device)  # min. odleglosc chwytak-cube_3 w fazie polityki 2
        n_idle_far = 0  # nie ruszona i chwytak nigdy nie zblizyl sie na < 8 cm
        n_idle_near = 0  # nie ruszona, ale chwytak byl blisko (chwyt sie nie udal)
        n_c3_idle = 0  # cube_3 na stole, nie ruszona (polityka 2 jej nie podniosla)
        n_c3_dropped = 0  # cube_3 na stole, ale ruszona (upuszczona / zrzucona)
        n_c3_other = 0  # cube_3 gdzie indziej (w powietrzu, na wiezy niestabilnie, itp.)
        n_fail_off = 0  # porazka po przelaczeniu: cube_2 nie na cube_1 na koncu
        n_fail_c3 = 0  # porazka po przelaczeniu: wieza 1+2 stoi, ale cube_3 nie na gorze
 
        finished = 0
        n_handoff = 0
        n_success = 0
        n_success_first = 0
        n_success_rec = 0
        n_used_rec = 0
        n_no_handoff = 0
        sum_recov = 0
        sum_p0 = 0
        cnt_p0 = 0
 
        obs = env.get_observations()
        if args_cli.p2_only:
            _rebuild_tower(torch.arange(n, device=device))
            phase[:] = 1
            handed[:] = True
            obs = env.get_observations()
        try:
            with torch.inference_mode():
                while finished < args_cli.episodes:
                    a1 = policy_1(obs)
                    a2 = policy_2(obs)
                    q_default = robot.data.default_joint_pos.torch[:, arm_ids]
                    # puszczenie: ramie trzyma pozycje q_hold, chwytak otwarty (+1 = open)
                    rel_action = torch.cat(
                        [(q_hold - q_default) / arm_scale, torch.ones(n, 1, device=device)], dim=-1
                    )
                    ph = phase.unsqueeze(-1)
                    actions = torch.where(ph == 0, a1, torch.where(ph == 1, a2, rel_action))
                    obs, _, dones, _ = env.step(actions)
                    reset_1(dones)
                    reset_2(dones)
                    dones = dones.bool()
 
                    # Dla srodowisk, ktore wlasnie sie zresetowaly (dones), stan sceny jest juz nowy,
                    # wiec liczniki i flagi aktualizujemy tylko dla srodowisk 'zywych'.
                    alive = ~dones
                    ep_step = torch.where(alive, ep_step + 1, torch.zeros_like(ep_step))
 
                    pair = cubes_stacked(
                        base, cube_3_cfg=None, xy_threshold=0.02, height_diff=0.05, atol=0.005, rtol=0.0
                    )
                    tower = cubes_stacked(base, xy_threshold=0.02, height_diff=0.05, atol=0.005, rtol=0.0)
                    cube_2_speed = torch.linalg.norm(base.scene["cube_2"].data.root_lin_vel_w.torch, dim=1)
                    cube_3_speed = torch.linalg.norm(base.scene["cube_3"].data.root_lin_vel_w.torch, dim=1)
 
                    # --- faza 0 -> 1: para cube_2/cube_1 stoi nieruchomo z otwartym chwytakiem ---
                    pair_still = pair & (cube_2_speed < args_cli.max_speed) & (phase == 0) & alive
                    hold_cnt = torch.where(pair_still, hold_cnt + 1, torch.zeros_like(hold_cnt))
                    arm_err = torch.linalg.norm(
                        robot.data.joint_pos.torch[:, arm_ids] - robot.data.default_joint_pos.torch[:, arm_ids], dim=1
                    )
                    if args_cli.clean_handoff:
                        arm_ok = torch.ones_like(pair)
                    elif args_cli.arm_home_tol > 0:
                        arm_ok = (arm_err < args_cli.arm_home_tol) | (
                            hold_cnt >= args_cli.hold_steps + args_cli.arm_wait_max
                        )
                    else:
                        arm_ok = torch.ones_like(pair)
                    switch = (phase == 0) & (hold_cnt >= args_cli.hold_steps) & arm_ok & alive
                    if switch.any():
                        first = switch & ~handed
                        if first.any():
                            sum_p0 += int(ep_step[first].sum().item())
                            cnt_p0 += int(first.sum().item())
                            q1 = base.scene["cube_1"].data.root_quat_w.torch
                            q2 = base.scene["cube_2"].data.root_quat_w.torch
                            ang_c12 = 2.0 * torch.acos((q1 * q2).sum(dim=-1).abs().clamp(max=1.0))
                            sw_ang = torch.where(first, ang_c12, sw_ang)
                            from isaaclab.utils import math as math_utils
 
                            r33 = math_utils.matrix_from_quat(q2)[:, 2, 2]
                            sw_tilt = torch.where(first, torch.acos(r33.clamp(-1.0, 1.0)), sw_tilt)
                            dq_now = (
                                robot.data.joint_pos.torch[:, arm_ids] - robot.data.default_joint_pos.torch[:, arm_ids]
                            )
                            sw_err = torch.where(first, arm_err, sw_err)
                            sw_dq = torch.where(first.unsqueeze(-1), dq_now, sw_dq)
                            if args_cli.arm_home_tol > 0:
                                fb_now = hold_cnt >= args_cli.hold_steps + args_cli.arm_wait_max
                            else:
                                fb_now = torch.zeros_like(first)
                            sw_fb = torch.where(first, fb_now, sw_fb)
                        if args_cli.clean_handoff:
                            sw_ids = switch.nonzero(as_tuple=False).squeeze(-1)
                            q_def = robot.data.default_joint_pos.torch[sw_ids][:, arm_ids]
                            robot.write_joint_state_to_sim(
                                q_def, torch.zeros_like(q_def), joint_ids=arm_ids, env_ids=sw_ids
                            )
                            robot.set_joint_position_target(q_def, joint_ids=arm_ids, env_ids=sw_ids)
                            act = base.action_manager.action
                            act[sw_ids] = 0.0  # 0 = chwytak otwarty (>= 0), ramie = poza bazowa
                            obs = env.get_observations()
                        phase = torch.where(switch, torch.ones_like(phase), phase)
                        handed |= switch
                        hold_cnt = torch.where(switch, torch.zeros_like(hold_cnt), hold_cnt)
 
                    # --- faza 1 -> 2: wieza zburzona (cube_2 poza cube_1 przez >= knock_steps krokow) ---
                    d12 = base.scene["cube_2"].data.root_pos_w.torch - base.scene["cube_1"].data.root_pos_w.torch
                    intact = (torch.linalg.norm(d12[:, :2], dim=1) < 0.04) & ((d12[:, 2] - 0.05).abs() < 0.02)
                    last_intact = torch.where(alive, intact, last_intact)
                    c3_now = base.scene["cube_3"].data.root_pos_w.torch
                    c3_start = torch.where((alive & (ep_step == 1)).unsqueeze(-1), c3_now, c3_start)
                    c3_last = torch.where(alive.unsqueeze(-1), c3_now, c3_last)
                    ee_now = base.scene["ee_frame"].data.target_pos_w.torch[..., 0, :]
                    d_ee_c3 = torch.linalg.norm(c3_now - ee_now, dim=1)
                    min_d = torch.where((phase == 1) & alive, torch.minimum(min_d, d_ee_c3), min_d)
                    track = (phase == 1) & alive & ~success
                    bad_cnt = torch.where(track & ~intact, bad_cnt + 1, torch.zeros_like(bad_cnt))
                    recover = (bad_cnt >= args_cli.knock_steps) & (recov < args_cli.max_recoveries) & alive
                    if recover.any():
                        q_hold[recover] = robot.data.joint_pos.torch[:, arm_ids][recover]
                        phase = torch.where(recover, torch.full_like(phase, 2), phase)
                        rel_cnt = torch.where(recover, torch.zeros_like(rel_cnt), rel_cnt)
                        recov = torch.where(recover, recov + 1, recov)
                        bad_cnt = torch.where(recover, torch.zeros_like(bad_cnt), bad_cnt)
                        hold_cnt = torch.where(recover, torch.zeros_like(hold_cnt), hold_cnt)
 
                    # --- faza 2 -> 0: po puszczeniu cube_3 wraca polityka 1 ---
                    releasing = (phase == 2) & alive
                    rel_cnt = torch.where(releasing, rel_cnt + 1, torch.zeros_like(rel_cnt))
                    back = releasing & (rel_cnt >= args_cli.release_steps)
                    phase = torch.where(back, torch.zeros_like(phase), phase)
 
                    # --- sukces: 3 kostki w wiezy, chwytak otwarty, nieruchomo przez success_hold krokow ---
                    tower_still = (
                        tower & (cube_3_speed < args_cli.max_speed) & (cube_2_speed < args_cli.max_speed) & alive
                    )
                    succ_cnt = torch.where(tower_still, succ_cnt + 1, torch.zeros_like(succ_cnt))
                    new_s = (succ_cnt >= args_cli.success_hold) & alive & ~success
                    if new_s.any():
                        succ_rec = torch.where(new_s, recov, succ_rec)
                        success |= new_s
 
                    # --- koniec epizodu: zapisz wynik i wyzeruj stan ---
                    if dones.any():
                        d = dones
                        finished += int(d.sum().item())
                        n_handoff += int(handed[d].sum().item())
                        n_no_handoff += int((~handed)[d].sum().item())
                        n_success += int(success[d].sum().item())
                        n_success_first += int((success & (succ_rec == 0))[d].sum().item())
                        n_success_rec += int((success & (succ_rec > 0))[d].sum().item())
                        n_used_rec += int((recov > 0)[d].sum().item())
                        sum_recov += int(recov[d].sum().item())
                        post_fail = d & handed & ~success
                        n_fail_off += int((post_fail & ~last_intact).sum().item())
                        n_fail_c3 += int((post_fail & last_intact).sum().item())
                        c3_fail = post_fail & last_intact
                        on_table = c3_last[:, 2] < 0.035
                        moved = torch.linalg.norm(c3_last - c3_start, dim=1) > 0.03
                        idle = c3_fail & on_table & ~moved
                        n_c3_idle += int(idle.sum().item())
                        n_idle_far += int((idle & (min_d >= 0.08)).sum().item())
                        n_idle_near += int((idle & (min_d < 0.08)).sum().item())
                        idle_far = idle & (min_d >= 0.08)
                        ang_succ += sw_ang[d & success].tolist()
                        ang_idle += sw_ang[idle_far].tolist()
                        ang_other += sw_ang[post_fail & ~idle_far].tolist()
                        for key, mask in (("succ", d & success), ("idle", idle_far), ("other", post_fail & ~idle_far)):
                            grp_err[key] += sw_err[mask].tolist()
                            grp_dq[key] += sw_dq[mask].abs().tolist()
                            grp_fb[key] += sw_fb[mask].tolist()
                        tilt_succ += sw_tilt[d & success].tolist()
                        tilt_idle += sw_tilt[idle_far].tolist()
                        tilt_other += sw_tilt[post_fail & ~idle_far].tolist()
                        sw_tilt[d] = 0.0
                        sw_ang[d] = 0.0
                        min_d[d] = 10.0
                        n_c3_dropped += int((c3_fail & on_table & moved).sum().item())
                        n_c3_other += int((c3_fail & ~on_table).sum().item())
                        last_intact[d] = True
                        for t in (ep_step, hold_cnt, succ_cnt, bad_cnt, rel_cnt, recov, phase):
                            t[d] = 0
                        succ_rec[d] = -1
                        success[d] = False
                        handed[d] = False
                        if args_cli.p2_only:
                            phase[d] = 1
                            handed[d] = True
                            _rebuild_tower(d.nonzero(as_tuple=False).squeeze(-1))
                        print(
                            f"[chain] epizody: {finished}/{args_cli.episodes}  "
                            f"sukcesy: {n_success} (z pierwszej proby: {n_success_first})  "
                            f"odzyskiwania uzyte: {n_used_rec}",
                            flush=True,
                        )
        except KeyboardInterrupt:
            pass
 
        print("=" * 70)
        print(f"Epizody ocenione: {finished}   (max_recoveries={args_cli.max_recoveries})")
        print(f"Sukces od pierwszej proby (bez odzyskiwania): {_fmt(n_success_first, finished)}")
        print(f"Sukces lacznie (z odzyskiwaniem):             {_fmt(n_success, finished)}")
        print(f"  w tym uratowane przez odzyskiwanie:         {n_success_rec}")
        print("-" * 70)
        print(f"Epizody, w ktorych uzyto odzyskiwania:        {n_used_rec}  (lacznie {sum_recov} odzyskiwan)")
        print(f"Nigdy nie doszlo do przelaczenia 1 -> 2:      {n_no_handoff}  (polityka 1 nie zbudowala pary)")
        print(f"Przelaczenia 1 -> 2 (min. raz):               {n_handoff}")
        print(f"Porazki lacznie:                              {finished - n_success}")
        print(f"  po przelaczeniu, cube_2 NIE na cube_1:      {n_fail_off}")
        print(f"  po przelaczeniu, wieza 1+2 stoi, cube_3 nie: {n_fail_c3}")
        print(f"      cube_3 na stole, NIE ruszona (nie podniesiona): {n_c3_idle}")
        print(f"          z tego chwytak nigdy nie podszedl (< 8 cm):  {n_idle_far}")
        print(f"          z tego chwytak byl blisko, chwyt nieudany:   {n_idle_near}")
        print(f"      cube_3 na stole, ruszona (upuszczona/zrzucona): {n_c3_dropped}")
        print(f"      cube_3 gdzie indziej (powietrze, na wiezy):     {n_c3_other}")
        if not args_cli.p2_only:
            def _ang(xs):
                if not xs:
                    return "brak"
                return f"srednio {sum(xs) / len(xs):.2f} rad, {100.0 * sum(x > 0.6 for x in xs) / len(xs):.0f}% powyzej 0.6 rad (n={len(xs)})"
 
            print("-" * 70)
            print("Obrot cube_2 wzgledem cube_1 w chwili przelaczenia (trening polityki 2: <= 0.5 rad):")
            print(f"  sukcesy:               {_ang(ang_succ)}")
            print(f"  bezczynne porazki:     {_ang(ang_idle)}")
            print(f"  pozostale porazki:     {_ang(ang_other)}")
            def _tilt(xs):
                if not xs:
                    return "brak"
                return (
                    f"srednio {sum(xs) / len(xs):.2f} rad, max {max(xs):.2f}, "
                    f"{100.0 * sum(x > 0.3 for x in xs) / len(xs):.0f}% powyzej 0.3 rad (n={len(xs)})"
                )
 
            print("Przechyl cube_2 od pionu w chwili przelaczenia (0 = lezy plasko; trening polityki 2: tylko yaw, czyli 0):")
            print(f"  sukcesy:               {_tilt(tilt_succ)}")
            print(f"  bezczynne porazki:     {_tilt(tilt_idle)}")
            print(f"  pozostale porazki:     {_tilt(tilt_other)}")
            print("Stan ramienia w chwili przelaczenia (blad = norma odchylenia od pozycji bazowej; |dq| = staw 1..7):")
            for key, name in (("succ", "sukcesy"), ("idle", "bezczynne porazki"), ("other", "pozostale porazki")):
                if not grp_err[key]:
                    print(f"  {name}: brak")
                    continue
                m_err = sum(grp_err[key]) / len(grp_err[key])
                m_dq = torch.tensor(grp_dq[key]).mean(dim=0).tolist()
                fb = 100.0 * sum(grp_fb[key]) / len(grp_fb[key])
                print(
                    f"  {name}: blad {m_err:.2f} rad, |dq| = "
                    + " ".join(f"{x:.2f}" for x in m_dq)
                    + f", przelaczone zabezpieczeniem {fb:.0f}% (n={len(grp_err[key])})"
                )
        if cnt_p0:
            print(f"Sredni czas do pierwszego przelaczenia:       {sum_p0 / cnt_p0:.0f} krokow = {sum_p0 / cnt_p0 * dt:.1f} s")
        print("=" * 70)
        env.close()
 
 
if __name__ == "__main__":
    main()