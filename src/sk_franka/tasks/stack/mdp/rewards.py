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
            object_cfg:SceneEntityCfg = SceneEntityCfg("cube_3"),
            target_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
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



def cube_3_on_cube_2(
        env:ManagerBasedRLEnv,
        xy_threshold:float = 0.02,
        height_diff: float = 0.05,
        atol: float = 0.005,
        speed_std: float = 0.05,
        cube_3_cfg:SceneEntityCfg = SceneEntityCfg("cube_3"),


) ->torch.Tensor: 
    stacked = cubes_stacked(
        env,
        xy_threshold = xy_threshold,
        height_diff = height_diff,
        atol = atol,
        rtol = 0.0,
    )

    cube_3_speed = env.scene[cube_3_cfg.name]
    cube_3_velocity = cube_3_speed.data.root_lin_vel_w.torch
    
    speed = torch.linalg.norm(cube_3_velocity,dim=1)
    cube_3_still = 1-torch.tanh(speed/speed_std)

    return stacked.float() * cube_3_still


def release_cube(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.02,
    height_diff: float = 0.05,
    height_tolerance: float = 0.01,
    grip_width: float = 0.025,
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    finger_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

    cube_2_pos = cube_2.data.root_pos_w.torch
    cube_3_pos = cube_3.data.root_pos_w.torch
    cubes_diff = cube_3_pos - cube_2_pos

    xy_dist = torch.linalg.norm(cubes_diff[:, [0, 1]], dim=1)
    height_error = torch.abs(cubes_diff[:, 2] - height_diff)
    cubes_aligned = (xy_dist < xy_threshold) & (height_error < height_tolerance)

    
    finger_pos = robot.data.joint_pos.torch[:, finger_ids]
    fully_open_gripper = (finger_pos > env.cfg.gripper_open_val - 0.005).all(dim=1)
    return cubes_aligned.float() * fully_open_gripper 






