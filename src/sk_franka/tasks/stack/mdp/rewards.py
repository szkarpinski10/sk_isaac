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
    open_intent = torch.sigmoid(3.0 * grip_action)
    return near * (0.7 * open_amount + 0.3 * open_intent)


def home_return(
    env:ManagerBasedRLEnv,
    std:float = 1.0,
    xy_threshold :float = 0.02,
    height_diff:float = 0.05,
    atol:float = 0.005,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) ->torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    arm_ids, _ = robot.find_joints(["panda_joint.*"])
    joints_q = robot.data.joint_pos.torch[:, arm_ids]
    joints_q_home = robot.data.default_joint_pos.torch[:,arm_ids]
    err = torch.linalg.norm(joints_q - joints_q_home,dim = 1)
    stacked = cubes_stacked(
        env,cube_3_cfg = None, xy_threshold = xy_threshold, height_diff= height_diff, atol = atol, rtol = 0.0
    )

    

    return stacked.float() * (1 - torch.tanh(err/std))

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



