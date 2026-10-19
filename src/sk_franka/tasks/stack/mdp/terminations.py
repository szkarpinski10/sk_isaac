# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

import math
import isaaclab.utils.math as math_utils

def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg | None = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    atol: float = 0.0001,
    rtol: float = 0.0001,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w.torch - cube_2.data.root_pos_w.torch

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.linalg.norm(pos_diff_c12[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.linalg.norm(pos_diff_c12[:, 2:], dim=1)

    # Check cube positions
    stacked = xy_dist_c12 < xy_threshold
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)

    if cube_3_cfg is not None:
        cube_3: RigidObject = env.scene[cube_3_cfg.name]
        pos_diff_c23 = cube_2.data.root_pos_w.torch - cube_3.data.root_pos_w.torch

        # Compute cube position difference in x-y plane
        xy_dist_c23 = torch.linalg.norm(pos_diff_c23[:, :2], dim=1)

        # Compute cube height difference
        h_dist_c23 = torch.linalg.norm(pos_diff_c23[:, 2:], dim=1)

        # Check cube positions
        stacked = torch.logical_and(xy_dist_c23 < xy_threshold, stacked)
        stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
        stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = wp.to_torch(surface_gripper.state).view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) >= 1, "Terminations require at least one gripper joint"
            # Success also requires the gripper to be released (every jaw back at the open value).
            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            for joint_id in gripper_joint_ids:
                stacked = torch.logical_and(
                    torch.isclose(
                        robot.data.joint_pos.torch[:, joint_id],
                        open_val,
                        atol=atol,
                        rtol=rtol,
                    ),
                    stacked,
                )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def debug_print_obs(env: ManagerBasedRLEnv, env_ids):
    """Wydruk do porównania z MuJoCo (tylko kroki 20-45)."""
    step = env.common_step_counter
    if step < 20 or step > 45:
        return
    origin = env.scene.env_origins[0].cpu().numpy()
    a = env.action_manager.action[0].cpu().numpy()
    finger = env.scene["robot"].data.joint_pos.torch[0, 7].item()
    cube = env.scene["cube_2"].data.root_pos_w.torch[0].cpu().numpy() - origin
    tcp = env.scene["ee_frame"].data.target_pos_w.torch[0, 0].cpu().numpy() - origin
    print(f"[ISAAC {step}] grip_a={a[7]:6.2f}  palce={finger:.4f}  "
        f"cube2={cube.round(3)}  tcp={tcp.round(3)}")

    robot = env.scene["robot"]
    hid = robot.find_bodies("panda_hand")[0][0]
    qh = robot.data.body_quat_w.torch[0, hid].unsqueeze(0)
    qc = env.scene["cube_2"].data.root_quat_w.torch[0].unsqueeze(0)
    dev = qh.device
    yh = math_utils.quat_apply(qh, torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]   # oś otwierania palców
    xc = math_utils.quat_apply(qc, torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]   # ściana kostki
    zc = math_utils.quat_apply(qc, torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]   # pion kostki
    a_h = math.degrees(math.atan2(yh[1].item(), yh[0].item()))
    a_c = math.degrees(math.atan2(xc[1].item(), xc[0].item()))
    d = (a_h - a_c + 45) % 90 - 45
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, zc[2].item()))))
    print(f"   yaw_roznica={d:6.1f}°  przechyl={tilt:5.1f}°")