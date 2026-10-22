# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the rigid lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from .terminations import cubes_stacked
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w.torch
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


class object_goal_distance(ManagerTermBase):
    """Reward the agent for tracking the object-to-goal pose using a tanh kernel.

    If ``success_threshold`` is provided in the term params, this also tracks per-episode
    success (sticky binary: object ever within ``success_threshold`` of the commanded goal
    while lifted above ``minimal_height``) and logs the mean across environments under
    ``Metrics/success_rate`` on reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        object_pos_w = obj.data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
        is_lifted = object_pos_w[:, 2] > minimal_height
        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)
        return is_lifted.float() * (1 - torch.tanh(distance / std))



#funkcja naprowadzająca cube_2 nad cube_1 
class object_goal_distance_above_target(ManagerTermBase):
    

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False
    def __call__(
            self,
            env:ManagerBasedRLEnv,
            std:float,
            minimal_height:float,
            place_offset: float = 0.05,
            object_cfg:SceneEntityCfg = SceneEntityCfg("cube_2"),
            target_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
            success_threshold: float | None = None,
    )-> torch.Tensor:

        obj : RigidObject = env.scene[object_cfg.name]
        target : RigidObject = env.scene[target_cfg.name]

        object_pos_w = obj.data.root_pos_w.torch
        target_pos_w = target.data.root_pos_w.torch.clone()
        target_pos_w[:,2] += place_offset

        distance = torch.linalg.norm(object_pos_w - target_pos_w,dim = 1)

        is_lifted = object_pos_w[:, 2] > minimal_height
        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)
        return is_lifted.float() * (1 - torch.tanh(distance / std))



def cube_2_on_cube_1(
        env:ManagerBasedRLEnv,
        xy_threshold:float = 0.02,
        height_diff: float = 0.05,
        atol: float = 0.005,
        speed_std: float = 0.05,
        cube_2_cfg:SceneEntityCfg = SceneEntityCfg("cube_2"),


) ->torch.Tensor: 
    stacked = cubes_stacked(
        env,
        cube_3_cfg = None,
        xy_threshold = xy_threshold,
        height_diff = height_diff,
        atol = atol,
        rtol = 0.0,
    )

    cube_2_speed = env.scene[cube_2_cfg.name]
    cube_2_velocity = cube_2_speed.data.root_lin_vel_w.torch
    
    speed = torch.linalg.norm(cube_2_velocity,dim=1)
    cube_2_still = 1-torch.tanh(speed/speed_std)

    return stacked.float() * cube_2_still


def release_cube(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.02,
    height_diff: float = 0.05,
    height_tolerance: float = 0.01,
    grip_width: float = 0.037,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    finger_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

    cube_1_pos = cube_1.data.root_pos_w.torch
    cube_2_pos = cube_2.data.root_pos_w.torch
    cubes_diff = cube_2_pos - cube_1_pos

    xy_dist = torch.linalg.norm(cubes_diff[:, [0, 1]], dim=1)
    height_error = cubes_diff[:, 2] - height_diff
    dist = torch.sqrt(xy_dist**2 + height_error**2)
    near = 1.0 - torch.tanh(dist / 0.02)

    finger_pos = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1)
    open_amount = torch.clamp((finger_pos - grip_width) / (env.cfg.gripper_open_val - grip_width), 0.0, 1.0)


    grip_action = env.action_manager.action[:, 7]
    open_gripper_action = torch.sigmoid(3.0 * grip_action)
    return near * (0.7 * open_amount + 0.3 * open_gripper_action)


def home_return(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    xy_threshold: float = 0.02,
    height_diff: float = 0.05,
    atol: float = 0.005,
    clear_height: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    arm_ids, _ = robot.find_joints(["panda_joint.*"])
    joints_q = robot.data.joint_pos.torch[:, arm_ids]
    joints_q_home = robot.data.default_joint_pos.torch[:, arm_ids]
    err = torch.linalg.norm(joints_q - joints_q_home, dim=1)

    stacked = cubes_stacked(
        env, cube_3_cfg=None, xy_threshold=xy_threshold, height_diff=height_diff, atol=atol, rtol=0.0
    )

    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w.torch[:, 0, 2]
    c2_z = cube_2.data.root_pos_w.torch[:, 2]
    clear = torch.clamp((ee_z - c2_z) / clear_height, 0.0, 1.0)

    return stacked.float() * clear * (1 - torch.tanh(err / std))

class tower_knocked(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._placed[env_ids] = False
        self._hold[env_ids] = 0

    def __call__(
        self,
        env,
        hold_steps: int = 10,
        xy_threshold: float = 0.025,
        height_diff: float = 0.05,
        height_threshold: float = 0.01,
        cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
        cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ) -> torch.Tensor:
        diff = env.scene[cube_2_cfg.name].data.root_pos_w.torch - env.scene[cube_1_cfg.name].data.root_pos_w.torch
        on_top = (torch.linalg.norm(diff[:, :2], dim=1) < xy_threshold) & (
            torch.abs(diff[:, 2] - height_diff) < height_threshold
        )

        robot = env.scene["robot"]
        finger_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        is_open = (robot.data.joint_pos.torch[:, finger_ids] > env.cfg.gripper_open_val - 0.005).all(dim=1)

        self._hold = torch.where(on_top & is_open, self._hold + 1, torch.zeros_like(self._hold))
        self._placed |= self._hold >= hold_steps
        return self._placed & ~on_top

def arm_action_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action[:, :8]), dim=1)



def ee_cube_align(
    env: ManagerBasedRLEnv,
    std: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    cube_2: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_2_pos = cube_2.data.root_pos_w.torch
    ee_pos = ee_frame.data.target_pos_w.torch[:, 0, :]

    xy_dist = torch.norm(cube_2_pos[:, :2] - ee_pos[:, :2], dim=1)
    ee_close_cube = (ee_pos[:, 2] - cube_2_pos[:, 2]) < 0.10
    on_table = cube_2_pos[:, 2] < 0.04

    return (1 - torch.tanh(xy_dist / std)) * ee_close_cube.float() * on_table.float()



def grasp_height(
    env: ManagerBasedRLEnv,
    std: float = 0.03,
    xy_std: float = 0.02,
    xy_gate: float = 0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
 
    cube_pos = cube.data.root_pos_w.torch
    ee_pos = ee_frame.data.target_pos_w.torch[:, 0, :]
 
    dz = ee_pos[:, 2] - cube_pos[:, 2]
    dxy = torch.linalg.norm(ee_pos[:, :2] - cube_pos[:, :2], dim=1)
    near = (dxy < xy_gate) & (cube_pos[:, 2] < 0.04) & (dz < 0.08) & (dz > -0.03)
 
    h = torch.exp(-torch.square(dz / std)) * torch.exp(-torch.square(dxy / xy_std))
    closing = env.action_manager.action[:, 7] < 0          # BinaryJointPositionAction: < 0 = zamknij
 
    r = torch.where(closing, 2.0 * h - 1.0, 0.5 * h)
    return near.float() * r

class place_height(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        self._finger_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        self._prev_finger = torch.full((env.num_envs,), 0.04, device=env.device)
 
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev_finger[env_ids] = 0.04          # na starcie epizodu chwytak jest otwarty
 
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float = 0.03,
        xy_std: float = 0.02,
        height_diff: float = 0.05,
        xy_gate: float = 0.08,
        release_bonus: float = 10.0,
        open_thr: float = 0.035,
        cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
        cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        c1 = env.scene[cube_1_cfg.name].data.root_pos_w.torch
        c2 = env.scene[cube_2_cfg.name].data.root_pos_w.torch
        ee = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
 
        diff = c2 - c1
        xy = torch.linalg.norm(diff[:, :2], dim=1)
        herr = diff[:, 2] - height_diff
        ee_near = torch.linalg.norm(ee - c2, dim=1) < 0.03            # kostka jeszcze przy chwytaku
 
        gate = (xy < xy_gate) & (herr > -0.01) & (herr < 0.10) & ee_near & (c2[:, 2] > 0.04)
 
        finger = env.scene["robot"].data.joint_pos.torch[:, self._finger_ids].mean(dim=1)
        opened_now = (self._prev_finger < open_thr) & (finger >= open_thr)   # palce właśnie przeszły przez 3.5 cm
        self._prev_finger = finger.clone()
 
        h = torch.exp(-torch.square(herr / std)) * torch.exp(-torch.square(xy / xy_std))
        r = torch.where(gate & opened_now, release_bonus * (2.0 * h - 1.0), torch.zeros_like(h))
        return r

def _cube_horizontal_axis_yaw(R: torch.Tensor) -> torch.Tensor:
    up_idx = R[:, 2, :].abs().argmax(dim=1)                                    #
    idx_h = torch.where(up_idx == 0, torch.ones_like(up_idx), torch.zeros_like(up_idx))
    ax = R[torch.arange(R.shape[0], device=R.device), :, idx_h]                
    return torch.atan2(ax[:, 1], ax[:, 0])

def grasp_yaw_align(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
 
    cube_pos = cube.data.root_pos_w.torch
    ee_pos = ee_frame.data.target_pos_w.torch[:, 0, :]
 
    dz = ee_pos[:, 2] - cube_pos[:, 2]
    dxy = torch.linalg.norm(ee_pos[:, :2] - cube_pos[:, :2], dim=1)
    gate = (dxy < xy_threshold) & (dz < 0.10) & (dz > -0.03) & (cube_pos[:, 2] < 0.04)
 
    R_hand = math_utils.matrix_from_quat(ee_frame.data.target_quat_w.torch[:, 0, :])
    R_cube = math_utils.matrix_from_quat(cube.data.root_quat_w.torch)
 
    hand_yaw = torch.atan2(R_hand[:, 1, 1], R_hand[:, 0, 1])                    
    cube_yaw = _cube_horizontal_axis_yaw(R_cube)
    score = torch.square(torch.cos(2.0 * (hand_yaw - cube_yaw)))                
 
    down = torch.clamp(-R_hand[:, 2, 2], 0.0, 1.0) ** 4                         
 
    return gate.float() * down * score